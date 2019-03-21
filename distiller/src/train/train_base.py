'''
Training frame base class
'''

import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from distiller.src.utils import get_time, separate_bn_paras, find_most_recent_model
import math
from distiller.src.data_io.data_loader import get_train_list_loader
from distiller.src.roc_evaluate.run_time_evaluate import Run_Time_Evaluate
from distiller.src.models.model_map import model_map, loss_map
import torch.nn.init as init
import torch.nn as nn
from distiller.src.utils import reinit_certain_layer_param
import sys

class TrainBase():
    def __init__(self, conf, ):

        self.conf = conf
        self.gamma = conf.gamma

        # init load data
        self.loader, self.class_num = get_train_list_loader(conf)

        self.milestones = conf.milestones
        self.step = 0
        self.start_epoch = 0

        self.board_loss_every = conf.board_loss_num  # self.loader -> number of batch size
        self.evaluate_every = conf.test_num
        self.save_every = conf.snapshot
        print("save every %s steps"%self.save_every)


    def define_network(self):
        print(self.conf.net_mode)
        if self.conf.net_mode.find("MobileFaceNet") >=0 and "kernel" in self.conf.keys():
                self.model = model_map[self.conf.net_mode](self.conf.embedding_size, self.conf.kernel).to(self.conf.device)
                #show network in tensorboard
                self.model_board = model_map[self.conf.net_mode](self.conf.embedding_size, self.conf.kernel)

        elif self.conf.net_mode.find("FaceNet20") >=0 or self.conf.net_mode.find("FaceNetOrg")>=0:
            height = self.conf.input_size[0]
            width = self.conf.input_size[1]
            self.model = model_map[self.conf.net_mode](self.conf.embedding_size, height, width).to(self.conf.device)
            self.model_board = model_map[self.conf.net_mode](self.conf.embedding_size, height, width)

        elif self.conf.net_mode.find("ResNet") >=0:
            assert self.conf.input_size[0] == self.conf.input_size[1]
            self.model = model_map[self.conf.net_mode](self.conf.input_size[0]).to(self.conf.device)
            self.model_board = model_map[self.conf.net_mode](self.conf.input_size[0])
        elif self.conf.net_mode.find("DenseNet") >=0:
            self.model = model_map[self.conf.net_mode](input_size =self.conf.input_size,
                                                       bn_size=self.conf.bn_size, drop_rate=self.conf.drop_rate,
                                                       num_classes=self.conf.embedding_size).to(self.conf.device)
            self.model_board = model_map[self.conf.net_mode](input_size = self.conf.input_size,
                                                             bn_size=self.conf.bn_size, drop_rate=self.conf.drop_rate,
                                                             num_classes=self.conf.embedding_size)
        else:
            self.model = model_map[self.conf.net_mode](self.conf.embedding_size).to(self.conf.device)
            self.model_board = model_map[self.conf.net_mode](self.conf.embedding_size)

        #init weight
        self.model.apply(self.weight_init)
        self.model = torch.nn.DataParallel(self.model, self.conf.device_ids)
        self.model.to(self.conf.device)


    def loss_type(self):
        print(self.conf.loss_type, self.conf.scale, self.conf.bias)
        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias).to(self.conf.device)

    def train_batch_data(self, imgs, labels):
        imgs = imgs.to(self.conf.device)
        labels = labels.to(self.conf.device)
        self.optimizer.zero_grad()
        # embeddings = self.model(imgs)
        embeddings = self.model.forward(imgs)

        thetas = self.head(embeddings, labels)  # ArcFace loss add margin
        loss = self.conf.ce_loss(thetas, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item() / self.conf.batch_size


    def SGD_opt(self):
        # separate param into batch norm and none batchnorm
        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.module)

        if self.conf.net_mode.find("MobileFaceNet") >= 0:
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                {'params': paras_only_bn}
            ], lr=self.conf.lr, momentum=self.conf.momentum)
        else:
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                {'params': paras_only_bn}
            ], lr=self.conf.lr, momentum=self.conf.momentum)


    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] *= self.gamma
        print(self.optimizer)

    def weight_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



    def fintune_model(self, ):
        if self.conf.resume_training:
            opt_prefix = find_most_recent_model(self.conf.model_path)
            print(self.conf.model_path)
            model_prefix = opt_prefix.replace('opt', 'model')
            head_prefix = opt_prefix.replace('opt', 'head')
            model_path = '{}/{}.pth'.format(self.conf.model_path, model_prefix)
            head_path = '{}/{}.pth'.format(self.conf.model_path, head_prefix)
            opt_path = '{}/{}.pth'.format(self.conf.model_path, opt_prefix)
            self.load_model(model_path)
            self.head.load_state_dict(torch.load(head_path))

            print("Resume from: {}".format(model_prefix))
            #load start iteration num
            self.optimizer.load_state_dict(torch.load(opt_path))
            step_num = int(opt_prefix.split('iter-')[-1].split('_')[0])
            self.step = step_num
            num_batch_size = self.loader.__len__()
            self.start_epoch = int(step_num / num_batch_size)
            print("Start from epoch {}, step {}".format(self.start_epoch, self.step))

        elif self.conf.fintune_model_path != None:
            assert self.conf.opt_prefix == None

            print("Fintune from \n", self.conf.fintune_model_path)

            if len(self.conf.remove_layer_name) > 0:
                state_dict = torch.load(self.conf.fintune_model_path)
                new_state_dict = reinit_certain_layer_param(self.conf.remove_layer_name, state_dict,
                                                            self.conf.embedding_size)
                self.model.module.load_state_dict(new_state_dict)
            else:
                self.load_model(self.conf.fintune_model_path)

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


    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key , value in state_dict.items():
                    name_key = key.split('module.')[-1]
                    new_state_dict[name_key] = value
            self.model.module.load_state_dict(new_state_dict)

        elif self.conf.ignore_layer != None:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.split('.')[0] in self.conf.ignore_layer:
                    continue
                new_state_dict[key] = value

        else:
            self.model.module.load_state_dict(state_dict)


    def save_state(self,accuracy, time_stamp, extra=None, model_only=False):

        save_path = self.conf.model_path

        acc_str = ""
        for (test_set,acc) in accuracy.items():
            acc_str += '{}-{:.4f}_'.format(test_set.split('_')[0][0:3], acc)

        acc_str = acc_str.strip('_')
        torch.save(
            self.model.state_dict(), save_path /
            ('{}_{}_model_iter-{}_{}.pth'.format(time_stamp, extra, self.step, acc_str)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('{}_{}_head_iter-{}_{}.pth'.format(time_stamp, extra, self.step, acc_str)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('{}_{}_opt_iter-{}_{}.pth'.format(time_stamp, extra, self.step, acc_str)))




