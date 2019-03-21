#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import time
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
import myparser as parser
import operator
import sys
import os

from pathlib import Path
from easydict import EasyDict as edict
from torch.nn import CrossEntropyLoss
from datetime import datetime
from distiller.src.utils import get_training_param_from_filename, make_if_not_exist
from distiller.src.train.train_KnowledgeDistillation import TrainKnowledgeDistill
import shutil
import argparse
import torch._utils


# print(torch.__version__)

# Logger handle
msglogger = None


def add_training_conf(device_ids, resume_training):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    conf = edict()
    conf.lr = 1e-1
    conf.batch_size = 256  #
    conf.gamma = 0.1
    conf.milestones = [8, 15, 25]  # down learing rate
    conf.epochs = 30
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 16
    conf.ce_loss = CrossEntropyLoss()

    conf.board_loss_num = 10  # average loss among number of batches
    conf.snapshot = 2500

    conf.test_num = 2500
    conf.test_batch_size = 100
    conf.test_worker_num = 16
    conf.test_set = {"Child-95to5": 1e-4,
                     "XCH-mtcnn-09-01-29": 5e-6}

    conf.device_ids = device_ids
    conf.device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    conf.net_depth = 50
    conf.drop_ratio = 0.6

    # conf.remove_layer_name = ['linear', 'bn']
    conf.remove_layer_name = []

    # conf.fintune_model_path = "/home/zkx-97/Project/pytorch_models/FaceRecognition/finetune_model/2019-02-23-23-00_SpSVGAngFace-b0.4s40t0.3Pm5.3_fc_0.4_112x112_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_MobileFaceNet-d128_model_iter-192500_Bus-0.8514_XCH-0.9638/2019-02-23-23-00_SpSVGAngFace-b0.4s40t0.3Pm5.3_fc_0.4_112x112_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_MobileFaceNet-d128_model_iter-192500_Bus-0.8514_XCH-0.9638.pth"
    conf.fintune_model_path = None

    conf.opt_prefix = None
    conf.resume_training = resume_training

    # get Training DataSet and Training Network
    abs_py_file_name = os.path.abspath(__file__)
    py_file_name = abs_py_file_name.split('/')[-1]
    dataset, net_info, train_subject, loss_type, h_value, w_value, patch_info = \
        get_training_param_from_filename(py_file_name)

    # set ArcFace loss param  bias, scale_value
    sphere_param_part = train_subject.split('-')[-1]
    b_index = sphere_param_part.find('b')
    s_index = sphere_param_part.find('s')
    print(sphere_param_part)
    bias = float(sphere_param_part[b_index + 1:s_index])
    scale_value = float(sphere_param_part[s_index + 1:])

    conf.bias = bias
    conf.scale = scale_value

    job_name = "{}_{}_{}_{}".format(train_subject, patch_info, dataset, net_info)
    snapshot_dir = '../../Project/pytorch_models/FaceRecognition/snapshot/{}/{}'.format(loss_type, job_name)
    test_result_dir = "../../Project/pytorch_models/FaceRecognition/run_time_test/{}/{}".format(loss_type, job_name)
    log_path = './jobs/{}/{}/{}'.format(loss_type, job_name, current_time)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(test_result_dir)
    make_if_not_exist(log_path)

    # copy train_example_file to job file
    base_name = os.path.basename(py_file_name)
    log_parent_path = os.path.abspath(os.path.join(log_path, ".."))
    shutil.copy(abs_py_file_name, "{}/{}-{}".format(log_parent_path, current_time, base_name))

    # save directory path
    conf.model_path = Path(snapshot_dir)
    conf.log_path = log_path
    conf.test_roc_path = test_result_dir
    conf.job_name = job_name

    # net info
    conf.input_size = [int(h_value), int(w_value)]
    conf.net_mode = net_info.split('-')[0]
    conf.loss_type = loss_type
    conf.patch_info = patch_info
    conf.data_mode = dataset

    conf.bn_size = int(net_info.split('-')[1])
    conf.drop_rate = int(net_info.split('-')[2])
    conf.embedding_size = int(net_info.split('-')[3])
    conf.teacher_net_mode = None
    if len(net_info.split('-'))>4 and 'teacher' in net_info.split('-')[4]:
        conf.teacher_net_mode = net_info.split('-')[4].replace('teacher', '')
    print("bn_size", conf.bn_size)
    print("drop_rate", conf.drop_rate)
    print("embedding_size", conf.embedding_size)
    return conf

def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    global msglogger

    # Parse arguments
    args = parser.get_parser().parse_args()

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
    conf = add_training_conf(args.gpus, args.resume)
    learner = TrainKnowledgeDistill(conf)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, model, top1, top5, epoch, args.num_best_scores)
        is_best = epoch == perf_scores_history[0].epoch
        apputils.save_checkpoint(epoch, args.arch, model, optimizer, compression_scheduler,
                                 perf_scores_history[0].top1, is_best, args.name, msglogger.logdir)

    # Finally run results on the test set
    # test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)


OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def train(train_loader, model, criterion, optimizer, epoch,
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
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if not hasattr(args, 'kd_policy') or args.kd_policy is None:
            output = model(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if not args.earlyexit_lossweights:
            loss = criterion(output, target)
            # Measure accuracy
            classerr.add(output.data, target)
            acc_stats.append([classerr.value(1), classerr.value(5)])
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target, criterion, args)
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
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                errs['Top1'] = classerr.value(1)
                errs['Top5'] = classerr.value(5)
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                    errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()
    return acc_stats


def update_training_scores_history(perf_scores_history, model, top1, top5, epoch, num_best_scores):
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


def earlyexit_loss(output, target, criterion, args):
    loss = 0
    sum_lossweights = 0
    for exitnum in range(args.num_exits-1):
        loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, target)
    # handle final exit
    loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].data, target)
    return loss


def earlyexit_validate_loss(output, target, criterion, args):
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
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break                    # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                         torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1


def earlyexit_validate_stats(args):
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
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    total_top1 = 0
    total_top5 = 0
    for exitnum in range(args.num_exits):
        total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum, top1k_stats[exitnum], top5k_stats[exitnum])
    msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f", total_top1, total_top5)
    return total_top1, total_top5, losses_exits_stats


def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        quantizer = quantization.PostTrainLinearQuantizer.from_args(model, args)
        quantizer.prepare_model()
        model.to(args.device)

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, best_top1=top1, scheduler=scheduler,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir)


def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        model_summaries.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        model_summaries.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')


def automated_deep_compression(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)

    args.display_confusion = True
    validate_fn = partial(test, test_loader=test_loader, criterion=criterion,
                          loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion,
                       loggers=loggers, args=args)

    save_checkpoint_fn = partial(apputils.save_checkpoint, arch=args.arch, dir=msglogger.logdir)
    optimizer_data = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    adc.do_adc(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn)


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)

    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
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
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def create_quantization_stats_collector(model):
    distiller.utils.assign_layer_fq_names(model)
    return {'test': missingdict({'quantization_stats': QuantCalibrationStatsCollector(model, classes=None,
                                                                                      inplace_runtime_check=True,
                                                                                      disable_inplace_attrs=True)})}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.save(workbook)


def check_pytorch_version():
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version('1.0.1'):
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 1.0.1 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.0.1 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


if __name__ == '__main__':
    try:
        check_pytorch_version()
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
