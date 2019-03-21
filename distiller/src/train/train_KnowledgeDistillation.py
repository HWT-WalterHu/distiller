import sys
import os
sys.path.append(os.getcwd())
from distiller.src.train.train_base import TrainBase
from distiller.src.roc_evaluate.run_time_evaluate import Run_Time_Evaluate
from tensorboardX import SummaryWriter
from distiller.src.models.model_map import model_map, loss_map
from distiller.src.models.model import sphere_plusLoss
from distiller.src.utils import get_time, separate_bn_paras, find_most_recent_model
from torch import optim
from distiller.src.utils import reinit_certain_layer_param
from pathlib import Path
from easydict import EasyDict as edict
from torch.nn import CrossEntropyLoss
from datetime import datetime
from distiller.src.utils import get_training_param_from_filename, make_if_not_exist
import shutil
import torch._utils
import distiller.src.train.myparser as myparser


import os
import traceback
import logging
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from distiller.data_loggers import *
import distiller.quantization as quantization
import examples.automated_deep_compression as adc
from distiller.models import ALL_MODEL_NAMES, create_model
import operator
import tqdm
import math
import time



OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'
msglogger = None


class TrainKnowledgeDistill(TrainBase):
    def __init__(self, conf, args):
        super(TrainKnowledgeDistill,self).__init__(conf)
        # init roc running time evalute
        self.roc_evaluate = Run_Time_Evaluate(conf)
        self.args = args
        self.conf = conf


    def define_teacher_network(self):
        print("teacher model:", self.conf.teacher_net_mode)
        net_mode = self.conf.teacher_net_mode
        if net_mode.find("MobileFaceNet") >=0 and "kernel" in self.conf.keys():
                model = model_map[net_mode](self.conf.embedding_size, self.conf.kernel).to(self.conf.device)
                #show network in tensorboard

        elif net_mode.find("FaceNet") >=0 or net_mode.find("FaceNetOrg")>=0:
            height = self.conf.input_size[0]
            width = self.conf.input_size[1]
            model = model_map[net_mode](self.conf.embedding_size, height, width).to(self.conf.device)

        elif net_mode.find("ResNet") >=0:
            assert self.conf.input_size[0] == self.conf.input_size[1]
            model = model_map[net_mode](self.conf.input_size[0]).to(self.conf.device)
        else:
            model = model_map[net_mode](self.conf.embedding_size).to(self.conf.device)
        self.teacher = model
        #init weight
        self.teacher.apply(self.weight_init)
        self.teacher = torch.nn.DataParallel(self.teacher, self.conf.device_ids)
        self.teacher.to(self.conf.device)

    def load_teacher_params(self, ):
        if self.conf.teacher_model_path != None:
            assert self.conf.opt_prefix == None

            print("Load Teacher Params: \n", self.conf.teacher_model_path)
            self.load_teacher_model(self.conf.teacher_model_path)



    def load_teacher_model(self, model_path):
        state_dict = torch.load(model_path)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key.split('module.')[-1]
                new_state_dict[name_key] = value
            self.teacher.module.load_state_dict(new_state_dict)

        elif self.conf.ignore_layer != None:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.split('.')[0] in self.conf.ignore_layer:
                    continue
                new_state_dict[key] = value

        else:
            self.teacher.module.load_state_dict(state_dict)

    def define_teacher_head(self):
        self.teacher_head = loss_map[self.conf.teacher_loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                      s=self.conf.teacher_scale, m=self.conf.teacher_bias).to(self.conf.device)
        model_path = self.conf.teacher_model_path.split('/')
        model_path[-1] = model_path[-1].replace('model','head')
        head_path = '/'.join(model_path)
        self.teacher_head.load_state_dict(torch.load(head_path))

    def train_model(self):
        args = self.args
        script_dir = os.path.dirname(__file__)
        module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        global msglogger
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

        # Log various details about the execution environment.  It is sometimes useful
        # to refer to past experiment executions and this information may be useful.
        apputils.log_execution_env_state(args.compress, msglogger.logdir, gitroot=module_path)
        msglogger.debug("Distiller: %s", distiller.__version__)
        start_epoch = 0
        perf_scores_history = []
        if args.deterministic:
            # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
            # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
            # In Pytorch, support for deterministic execution is still a bit clunky.
            if args.workers > 1:
                msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
                exit(1)
            # Use a well-known seed, for repeatability of experiments
            distiller.set_deterministic()
        else:
            # This issue: https://github.com/pytorch/pytorch/issues/3659
            # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
            # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
            cudnn.benchmark = True

        if args.cpu or not torch.cuda.is_available():
            # Set GPU index to -1 if using CPU
            args.device = 'cpu'
            args.gpus = -1
        else:
            args.device = 'cuda'
            if args.gpus is not None:
                try:
                    args.gpus = [int(s) for s in args.gpus.split(',')]
                except ValueError:
                    msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
                    exit(1)
                available_gpus = torch.cuda.device_count()
                for dev_id in args.gpus:
                    if dev_id >= available_gpus:
                        msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                        .format(dev_id, available_gpus))
                        exit(1)
                # Set default device in case the first one on the list != 0
                torch.cuda.set_device(args.gpus[0])

        # Infer the dataset from the model name
        # args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
        # args.num_classes = 10 if args.dataset == 'cifar10' else 1000

        if args.earlyexit_thresholds:
            args.num_exits = len(args.earlyexit_thresholds) + 1
            args.loss_exits = [0] * args.num_exits
            args.losses_exits = []
            args.exiterrors = []

        # Create the model
        self.define_network()
        self.loss_type()
        self.SGD_opt()
        self.fintune_model()

        compression_scheduler = None
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.

        tflogger = TensorBoardLogger(self.conf.log_path)
        # tflogger = TensorBoardLogger(msglogger.logdir)
        pylogger = PythonLogger(msglogger)

        # capture thresholds for early-exit training
        if args.earlyexit_thresholds:
            msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(args.device)

        msglogger.info('Optimizer Type: %s', type(self.optimizer))
        msglogger.info('Optimizer Args: %s', self.optimizer.defaults)
        if args.greedy:
            return self.greedy(self.model, criterion, self.optimizer, pylogger, args)

        # This sample application can be invoked to produce various summary reports.
        if args.summary:
            return self.summarize_model(self.model, args.dataset, which_summary=args.summary)

        activations_collectors = self.create_activation_stats_collectors(self.model, *args.activation_stats)

        if args.qe_calibration:
            msglogger.info('Quantization calibration stats collection enabled:')
            msglogger.info('\tStats will be collected for {:.1%} of test dataset'.format(args.qe_calibration))
            msglogger.info('\tSetting constant seeds and converting model to serialized execution')
            distiller.set_deterministic()
            self.model = distiller.make_non_parallel_copy(self.model)
            activations_collectors.update(self.create_quantization_stats_collector(self.model))
            args.evaluate = True
            args.effective_test_size = args.qe_calibration

        # Load the datasets: the dataset to load is inferred from the model name passed
        # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
        # substring "_cifar", then cifar10 is used.
        # train_loader, val_loader, test_loader, _ = apputils.load_data(
        #     args.dataset, os.path.expanduser(args.data), args.batch_size,
        #     args.workers, args.validation_split, args.deterministic,
        #     args.effective_train_size, args.effective_valid_size, args.effective_test_size)
        # msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
        #                len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

        if args.compress:
            # The main use-case for this sample application is CNN compression. Compression
            # requires a compression schedule configuration file in YAML.
            compression_scheduler = distiller.file_config(self.model, self.optimizer, args.compress, compression_scheduler)
            # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
            self.model.to(args.device)
        elif compression_scheduler is None:
            compression_scheduler = distiller.CompressionScheduler(self.model)

        if args.thinnify:
            #zeros_mask_dict = distiller.create_model_masks_dict(model)
            assert args.resume is not None, "You must use --resume to provide a checkpoint file to thinnify"
            distiller.remove_filters(self.model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
            apputils.save_checkpoint(0, args.arch, self.model, optimizer=None, scheduler=compression_scheduler,
                                     name="{}_thinned".format(args.resume.replace(".pth.tar", "")), dir=msglogger.logdir)
            print("Note: your model may have collapsed to random inference, so you may want to fine-tune")
            return

        self.args.kd_policy = None
        if self.conf.teacher_net_mode:
            self.define_teacher_network()
            self.load_teacher_params()
            self.define_teacher_head()
            dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
            args.kd_policy = distiller.KnowledgeDistillationPolicy(self.model, self.teacher, self.head, self.teacher_head, args.kd_temp, dlw)
            compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                             frequency=1)

            msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
            msglogger.info('\tTeacher Model: %s', args.kd_teacher)
            msglogger.info('\tTemperature: %s', args.kd_temp)
            msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                           ' | '.join(['{:.2f}'.format(val) for val in dlw]))
            msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

        for epoch in (range(start_epoch, start_epoch + args.epochs)):
            # if epoch == start_epoch:
            #     self.writer = SummaryWriter(self.conf.log_path)
            # #     # write graps in to tensorboard
            #     net_input = torch.FloatTensor(1, 3, self.conf.input_size[0], self.conf.input_size[1]).zero_()
            #     self.writer.add_graph(self.model_board, net_input)
            #
            # This is the main training loop.
            msglogger.info('\n')
            if compression_scheduler:
                compression_scheduler.on_epoch_begin(epoch)

            if epoch in self.milestones:
                self.schedule_lr()
            # Train for one epoch
            with collectors_context(activations_collectors["train"]) as collectors:
                self.train(self.loader, self.model, criterion, self.optimizer, epoch, compression_scheduler,
                      loggers=[tflogger, pylogger], args=args)
                distiller.log_weights_sparsity(self.model, epoch, loggers=[tflogger, pylogger])
                distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                    collector=collectors["sparsity"])
                if args.masks_sparsity:
                    msglogger.info(distiller.masks_sparsity_tbl_summary(self.model, compression_scheduler))

            # evaluate on validation set


            if compression_scheduler:
                compression_scheduler.on_epoch_end(epoch, self.optimizer)

            # Update the list of top scores achieved so far, and save the checkpoint

        # Finally run results on the test set

    def train(self,train_loader, model, criterion, optimizer, epoch,
              compression_scheduler, loggers, args):
        """Training loop for one epoch."""
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
        batch_time = tnt.AverageValueMeter()
        data_time = tnt.AverageValueMeter()

        # For Early Exit, we define statistics for each exit
        # So exiterrors is analogous to classerr for the non-Early Exit case
        if args.earlyexit_lossweights:
            args.exiterrors = []
            for exitnum in range(args.num_exits):
                args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

        total_samples = len(train_loader.sampler)
        batch_size = train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)
        msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)
        # Switch to train mode
        model.train()
        acc_stats = []
        end = time.time()
        train_step = 0
        for (inputs, target) in tqdm.tqdm((train_loader)):
            if self.step % self.evaluate_every == 0 and self.step != 0:
                time_stamp = get_time()
                print("Test Model: ", self.conf.test_set)
                accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, time_stamp)
                distiller.log_eval_stasitics(accuracy,self.step,loggers[:1])
                # self.model.train()
                self.save_state(accuracy, time_stamp, extra=self.conf.job_name)
            self.step += 1
            # Measure data loading time
            data_time.add(time.time() - end)
            inputs, target = inputs.to(args.device), target.to(args.device)

            # Execute the forward phase, compute the output and measure loss
            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

            if not hasattr(args, 'kd_policy') or args.kd_policy is None:
                embedding = model(inputs)
                output = self.head(embedding, target)

            else:
                output = args.kd_policy.forward(target, inputs)
            if not args.earlyexit_lossweights:
                classerr.add(output.data, target)
                acc_stats.append([classerr.value(1), classerr.value(5)])
                loss = criterion(output, target)
            else:
                # Measure accuracy and record loss
                loss = self.earlyexit_loss(output, target, criterion, args)
            # Record loss
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            if compression_scheduler:
                # Before running the backward phase, we allow the scheduler to modify the loss
                # (e.g. add regularization loss)
                agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                      optimizer=optimizer, return_loss_components=True)
                loss = agg_loss.overall_loss
                losses[OVERALL_LOSS_KEY].add(loss.item())

                for lc in agg_loss.loss_components:
                    if lc.name not in losses:
                        losses[lc.name] = tnt.AverageValueMeter()
                    losses[lc.name].add(lc.value.item())
            else:
                losses[OVERALL_LOSS_KEY].add(loss.item())

            # Compute the gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

            # measure elapsed time
            batch_time.add(time.time() - end)
            steps_completed = (train_step + 1)
            if steps_completed % args.print_freq == 0:
            #     # Log some statistics

                errs = OrderedDict()
                if not args.earlyexit_lossweights:
                    errs['Top1'] = classerr.value(1)
                    errs['Top5'] = classerr.value(5)
            #     else:
            #         # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
            #         for exitnum in range(args.num_exits):
            #             errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
            #             errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)
            #
                stats_dict = OrderedDict()
                for loss_name, meter in losses.items():
                    stats_dict[loss_name] = meter.mean
                stats_dict.update(errs)
                stats_dict['LR'] = optimizer.param_groups[0]['lr']
                stats_dict['Time'] = batch_time.mean
                stats = ('Peformance/Training/', stats_dict)
            #
                params = model.named_parameters() if args.log_params_histograms else None
                distiller.log_training_progress(stats,
                                                params,
                                                epoch, steps_completed,
                                                steps_per_epoch, args.print_freq,
                                                loggers)
            end = time.time()
            train_step += 1
        return acc_stats

    def update_training_scores_history(self,perf_scores_history, model, top1, top5, epoch, num_best_scores):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""

        model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
        perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                                'sparsity': model_sparsity,
                                                                'top1': top1, 'top5': top5, 'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        # Sort by sparsity as main sort key, then sort by top1, top5 and epoch
        perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'), reverse=True)
        for score in perf_scores_history[:num_best_scores]:
            msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   Params: %d on epoch: %d]',
                           score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)

    def earlyexit_loss(self,output, target, criterion, args):
        loss = 0
        sum_lossweights = 0
        for exitnum in range(args.num_exits - 1):
            loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
            sum_lossweights += args.earlyexit_lossweights[exitnum]
            args.exiterrors[exitnum].add(output[exitnum].data, target)
        # handle final exit
        loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits - 1], target)
        args.exiterrors[args.num_exits - 1].add(output[args.num_exits - 1].data, target)
        return loss

    def earlyexit_validate_loss(self,output, target, criterion, args):
        # We need to go through each sample in the batch itself - in other words, we are
        # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
        # but with a grouping of samples equal to the batch size.
        # Note that final group might not be a full batch - so determine actual size.
        this_batch_size = target.size()[0]
        earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)

        for exitnum in range(args.num_exits):
            # calculate losses at each sample separately in the minibatch.
            args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
            # for batch_size > 1, we need to reduce this down to an average over the batch
            args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

        for batch_index in range(this_batch_size):
            earlyexit_taken = False
            # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
            for exitnum in range(args.num_exits - 1):
                if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                    # take the results from early exit since lower than threshold
                    args.exiterrors[exitnum].add(
                        torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                        torch.full([1], target[batch_index], dtype=torch.long))
                    args.exit_taken[exitnum] += 1
                    earlyexit_taken = True
                    break  # since exit was taken, do not affect the stats of subsequent exits
            # this sample does not exit early and therefore continues until final exit
            if not earlyexit_taken:
                exitnum = args.num_exits - 1
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1

    def earlyexit_validate_stats(self,args):
        # Print some interesting summary stats for number of data points that could exit early
        top1k_stats = [0] * args.num_exits
        top5k_stats = [0] * args.num_exits
        losses_exits_stats = [0] * args.num_exits
        sum_exit_stats = 0
        for exitnum in range(args.num_exits):
            if args.exit_taken[exitnum]:
                sum_exit_stats += args.exit_taken[exitnum]
                msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
                top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
                top5k_stats[exitnum] += args.exiterrors[exitnum].value(5)
                losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
        for exitnum in range(args.num_exits):
            if args.exit_taken[exitnum]:
                msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                               (args.exit_taken[exitnum] * 100.0) / sum_exit_stats)
        total_top1 = 0
        total_top5 = 0
        for exitnum in range(args.num_exits):
            total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
            total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
            msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum, top1k_stats[exitnum],
                           top5k_stats[exitnum])
        msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f", total_top1, total_top5)
        return total_top1, total_top5, losses_exits_stats

    def summarize_model(self,model, dataset, which_summary):
        if which_summary.startswith('png'):
            model_summaries.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
        elif which_summary == 'onnx':
            model_summaries.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
        else:
            distiller.model_summary(model, which_summary, dataset)

    def sensitivity_analysis(self,model, criterion, data_loader, loggers, args, sparsities):
        # This sample application can be invoked to execute Sensitivity Analysis on your
        # model.  The ouptut is saved to CSV and PNG.
        msglogger.info("Running sensitivity tests")
        if not isinstance(loggers, list):
            loggers = [loggers]
        test_fnc = partial(self.test, test_loader=data_loader, criterion=criterion,
                           loggers=loggers, args=args,
                           activations_collectors=self.create_activation_stats_collectors(model))
        which_params = [param_name for param_name, _ in model.named_parameters()]
        sensitivity = distiller.perform_sensitivity_analysis(model,
                                                             net_params=which_params,
                                                             sparsities=sparsities,
                                                             test_func=test_fnc,
                                                             group=args.sensitivity)
        distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
        distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')

    def greedy(self, model, criterion, optimizer, loggers, args):
        train_loader, val_loader, test_loader, _ = apputils.load_data(
            args.dataset, os.path.expanduser(args.data), args.batch_size,
            args.workers, args.validation_split, args.deterministic,
            args.effective_train_size, args.effective_valid_size, args.effective_test_size)

        test_fn = partial(self.test, test_loader=test_loader, criterion=criterion,
                          loggers=loggers, args=args, activations_collectors=None)
        train_fn = partial(self.train, train_loader=train_loader, criterion=criterion, args=args)
        assert args.greedy_target_density is not None
        distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                              args.greedy_target_density,
                                                              args.greedy_pruning_step,
                                                              test_fn, train_fn)

    def create_activation_stats_collectors(self,model, *phases):
        """Create objects that collect activation statistics.

        This is a utility function that creates two collectors:
        1. Fine-grade sparsity levels of the activations
        2. L1-magnitude of each of the activation channels

        Args:
            model - the model on which we want to collect statistics
            phases - the statistics collection phases: train, valid, and/or test

        WARNING! Enabling activation statsitics collection will significantly slow down training!
        """
        distiller.utils.assign_layer_fq_names(model)

        genCollectors = lambda: missingdict({
            "sparsity": SummaryActivationStatsCollector(model, "sparsity",
                                                        lambda t: 100 * distiller.utils.sparsity(t)),
            "l1_channels": SummaryActivationStatsCollector(model, "l1_channels",
                                                           distiller.utils.activation_channels_l1),
            "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                             distiller.utils.activation_channels_apoz),
            "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                             distiller.utils.activation_channels_means),
            "records": RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
        })

        return {k: (genCollectors() if k in phases else missingdict())
                for k in ('train', 'valid', 'test')}

    def create_quantization_stats_collector(self,model):
        distiller.utils.assign_layer_fq_names(model)
        return {'test': missingdict({'quantization_stats': QuantCalibrationStatsCollector(model, classes=None,
                                                                                          inplace_runtime_check=True,
                                                                                          disable_inplace_attrs=True)})}

    def save_collectors_data(self,collectors, directory):
        """Utility function that saves all activation statistics to Excel workbooks
        """
        for name, collector in collectors.items():
            workbook = os.path.join(directory, name)
            msglogger.info("Generating {}".format(workbook))
            collector.save(workbook)

class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior
