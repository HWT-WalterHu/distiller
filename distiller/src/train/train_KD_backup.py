import torch
from tqdm import tqdm
from distiller.src.train.train_base import TrainBase
from distiller.src.roc_evaluate.run_time_evaluate import Run_Time_Evaluate
from tensorboardX import SummaryWriter
from distiller.src.models.model_map import model_map, loss_map
from distiller.src.models.model import sphere_plusLoss
from distiller.src.utils import get_time, separate_bn_paras, find_most_recent_model
from torch import optim
from distiller.src.utils import reinit_certain_layer_param
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
import operator
import sys
import os

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'

class TrainKnowledgeDistill(TrainBase):
    def __init__(self, conf, args):
        super(TrainKnowledgeDistill,self).__init__(conf)
        # init roc running time evalute
        self.roc_evaluate = Run_Time_Evaluate(conf)
        self.args = args

    def define_teacher_network(self):
        print("teacher model:", self.conf.teacher_net_mode)
        net_mode = self.conf.teacher_net_mode
        if net_mode.find("MobileFaceNet") >=0 and "kernel" in self.conf.keys():
                model = model_map[self.conf.net_mode](self.conf.embedding_size, self.conf.kernel).to(self.conf.device)
                #show network in tensorboard
                model_board = model_map[self.conf.net_mode](self.conf.embedding_size, self.conf.kernel)

        elif net_mode.find("FaceNet") >=0 or self.conf.net_mode.find("FaceNetOrg")>=0:
            height = self.conf.input_size[0]
            width = self.conf.input_size[1]
            model = model_map[self.conf.net_mode](self.conf.embedding_size, height, width).to(self.conf.device)
            model_board = model_map[self.conf.net_mode](self.conf.embedding_size, height, width)

        elif net_mode.find("ResNet") >=0:
            assert self.conf.input_size[0] == self.conf.input_size[1]
            model = model_map[self.conf.net_mode](self.conf.input_size[0]).to(self.conf.device)
            model_board = model_map[self.conf.net_mode](self.conf.input_size[0])
        else:
            model = model_map[self.conf.net_mode](self.conf.embedding_size).to(self.conf.device)
            model_board = model_map[self.conf.net_mode](self.conf.embedding_size)
        self.teachermodel = model
        #init weight
        self.teachermodel.apply(self.weight_init)
        self.teachermodel = torch.nn.DataParallel(self.teachermodel, self.conf.device_ids)
        self.teachermodel.to(self.conf.device)

    def load_teacher_params(self, ):
        if self.conf.teacher_model_path != None:
            assert self.conf.opt_prefix == None

            print("Load Teacher Params: \n", self.conf.teacher_model_path)

            if len(self.conf.remove_layer_name) > 0:
                state_dict = torch.load(self.conf.fintune_model_path)
                new_state_dict = reinit_certain_layer_param(self.conf.remove_layer_name, state_dict,
                                                            self.conf.embedding_size)
                self.model.module.load_state_dict(new_state_dict)
            else:
                self.load_teacher_model(self.conf.teacher_model_path)


        #load ArcFace or AMsoftmax's weight
        elif self.conf.opt_prefix != None:
            print("opt_prefix from: {}".format(self.conf.opt_prefix))

            assert self.conf.fintune_model_path == None
            opt_prefix = self.conf.opt_prefix
            model_prefix = opt_prefix.replace('opt', 'model')
            head_prefix = opt_prefix.replace('opt', 'head')
            model_path = '{}/{}.pth'.format(self.conf.model_path, model_prefix)
            head_path = '{}/{}.pth'.format(self.conf.model_path, head_prefix)
            self.load_model(model_path)
            self.head.load_state_dict(torch.load(head_path))

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
            self.teachermodel.module.load_state_dict(new_state_dict)

        elif self.conf.ignore_layer != None:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.split('.')[0] in self.conf.ignore_layer:
                    continue
                new_state_dict[key] = value

        else:
            self.teachermodel.module.load_state_dict(state_dict)

    def init_model_param(self):
        #decide net structure
        self.define_network()
        self.loss_type()
        self.SGD_opt()
        self.fintune_model()

    def init_distiller(self, args):
        script_dir = os.path.dirname(__file__)
        module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
        global msglogger

        # Parse arguments

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

        # Log various details about the execution environment.  It is sometimes useful
        # to refer to past experiment executions and this information may be useful.
        apputils.log_execution_env_state(args.compress, msglogger.logdir, gitroot=module_path)
        msglogger.debug("Distiller: %s", distiller.__version__)

        start_epoch = 0

        cudnn.benchmark = True
        compression_scheduler = None
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
        tflogger = TensorBoardLogger(msglogger.logdir)
        pylogger = PythonLogger(msglogger)


        if args.compress:
            print("using compress")
            # The main use-case for this sample application is CNN compression. Compression
            # requires a compression schedule configuration file in YAML.
            self.compression_scheduler = distiller.file_config(self.model, self.optimizer, args.compress,
                                                          compression_scheduler)
            # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
            # self.model.to(args.device)
        elif compression_scheduler is None:
            self.compression_scheduler = distiller.CompressionScheduler(self.model)


        self.kd_policy = None
        print("teacher net mode :{}".format(self.conf.teacher_net_mode))
        if self.conf.teacher_net_mode is not None:
            self.define_teacher_network()
            self.load_teacher_params()
            dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
            self.kd_policy = distiller.KnowledgeDistillationPolicy(self.model, self.teachermodel, args.kd_temp,
                                                                   dlw)
            self.compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch,
                                             ending_epoch=self.conf.epochs,
                                             frequency=1)

            msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
            msglogger.info('\tTeacher Model: %s', args.kd_teacher)
            msglogger.info('\tTemperature: %s', args.kd_temp)
            msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                           ' | '.join(['{:.2f}'.format(val) for val in dlw]))
            msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)


    def train_KD_batch_data(self, imgs, labels, epoch):
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        imgs = imgs.to(self.conf.device)
        labels = labels.to(self.conf.device)
        if self.kd_policy is not None:
            embeddings = self.kd_policy.forward(imgs)
        else:
            embeddings = self.model.forward(imgs)
        # embeddings = self.model(imgs)

        thetas = self.head(embeddings, labels)  # ArcFace loss add margin
        loss = self.conf.ce_loss(thetas, labels)
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())
        if self.kd_policy is not None:
            agg_loss = self.compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return losses

    def train_stage(self):
        self.model.train()
        running_loss = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                # write graphs in to tensorboard
                net_input = torch.Tensor(1, 3, self.conf.input_size[0], self.conf.input_size[1]).zero_()
                self.writer.add_graph(self.model_board, net_input)
                is_first = False

            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            if self.compression_scheduler:
                self.compression_scheduler.on_epoch_begin(e)
            for imgs, labels in tqdm(iter(self.loader)):
                # continue
                running_loss += self.train_KD_batch_data(imgs, labels,e)
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('Training/Loss', loss_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_rate', lr, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    print("Test Model: ", self.conf.test_set)
                    accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, time_stamp)
                    for (test_set, acc) in accuracy.items():
                        self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
                    self.model.train()
                    self.save_state(accuracy, time_stamp, extra=self.conf.job_name)
                self.step += 1
        #
        # #save final model's result
        # accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, 1e-4)
        # for (test_set, acc) in accuracy.items():
        #     self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
        # self.save_state(accuracy, extra=self.conf.job_name)
        # self.writer.close()

    def train_model(self):
        self.init_model_param()
        self.init_distiller(self.args)
        self.train_stage()


