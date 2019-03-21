from src.models.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from src.utils import get_time, separate_bn_paras, find_most_recent_model
import math
from src.data_io.data_loader import get_train_list_loader
from src.roc_evaluate.run_time_evaluate import Run_Time_Evaluate
from src.models.model_map import model_map, loss_map
import torch.nn.init as init
import torch.nn as nn
from src.utils import reinit_certain_layer_param

class face_learner(object):
    def __init__(self, conf, ):
        print(conf)

        self.gamma = conf.gamma

        #init load data
        self.loader, self.class_num = get_train_list_loader(conf)
        #init tensorboard
        self.writer = SummaryWriter(conf.log_path)
        #init roc running time evalute
        self.roc_evaluate = Run_Time_Evaluate(conf)
        #init model according to model_map

        if conf.net_mode.find("MobileFaceNet") >=0 and "kernel" in conf.keys():
                self.model = model_map[conf.net_mode](conf.embedding_size, conf.kernel).to(conf.device)

        elif conf.net_mode.find("FaceNet20") >=0 or conf.net_mode.find("FaceNetOrg")>=0:
            height = conf.input_size[0]
            width = conf.input_size[1]
            self.model = model_map[conf.net_mode](conf.embedding_size, height, width).to(conf.device)
        else:
            self.model = model_map[conf.net_mode](conf.embedding_size).to(conf.device)
        #init weight
        self.model.apply(self.weight_init)

        #write graps in to tensorboard
        net_input = torch.FloatTensor(1,3, conf.input_size[0], conf.input_size[1]).zero_()
        self.writer.add_graph(self.model, net_input.to(conf.device))

        #using multi-gpus
        self.model = torch.nn.DataParallel(self.model, conf.device_ids)
        self.model.to(conf.device)

        self.milestones = conf.milestones
        self.step = 0
        self.start_epoch = 0

        #using loss type
        self.head = loss_map[conf.loss_type](embedding_size=conf.embedding_size, classnum=self.class_num,
                                             s = conf.scale, m=conf.bias).to(conf.device)
        # self.head.apply(self.weight_init)

        # separate param into batch norm and none batchnorm
        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.module)


        if conf.net_mode.find("MobileFaceNet") >=0:
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                {'params': paras_only_bn}
                            ], lr = conf.lr, momentum = conf.momentum)
        else:
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                {'params': paras_only_bn}
                            ], lr = conf.lr, momentum = conf.momentum)

        print('optimizers generated')
        self.board_loss_every = conf.board_loss_num # self.loader -> number of batch size
        self.evaluate_every = conf.test_num
        self.save_every = conf.snapshot

        if conf.fintune_model_path == None:
            #2018-11-23-11-25_optimizer_XCH_out2_95to5-0.9415_Business_small_95to5-0.8028__step-157500_None
            if conf.opt_prefix  == None:
                opt_prefix = find_most_recent_model(conf.model_path)
            else:
                opt_prefix = conf.opt_prefix

            if opt_prefix != "":
                model_prefix = opt_prefix.replace('opt', 'model')
                head_prefix = opt_prefix.replace('opt', 'head')
                model_path = '{}/{}.pth'.format(conf.model_path, model_prefix)
                head_path = '{}/{}.pth'.format(conf.model_path, head_prefix)
                opt_path = '{}/{}.pth'.format(conf.model_path, opt_prefix)
                # self.model.module.load_state_dict(torch.load(model_path.replace('__step-', '_step-')))
                self.load_model(model_path)
                self.head.load_state_dict(torch.load(head_path))

                if conf.resume_training:
                    self.optimizer.load_state_dict(torch.load(opt_path))
                    step_num = int(opt_prefix.split('iter-')[-1].split('_')[0])
                    self.step = step_num
                    num_batch_size = self.loader.__len__()
                    self.start_epoch = int(step_num / num_batch_size)
                    print("Start from epoch {}, step {}".format(self.start_epoch, self.step))
                    print(opt_prefix)
        else:
                print("Fintine from ", conf.fintune_model_path)

                if len(conf.remove_layer_name) >0:
                    state_dict = torch.load(conf.fintune_model_path)
                    new_state_dict = reinit_certain_layer_param(conf.remove_layer_name,state_dict, conf.embedding_size)
                    self.model.module.load_state_dict(new_state_dict)
                else:
                    self.load_model(conf.fintune_model_path)




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
        else:
            self.model.module.load_state_dict(state_dict)

    def save_state(self, conf, accuracy, extra=None, model_only=False):

        save_path = conf.model_path

        acc_str = ""
        for (test_set,acc) in accuracy.items():
            acc_str += '{}-{:.4f}_'.format(test_set.split('_')[0][0:3], acc)

        acc_str = acc_str.strip('_')
        torch.save(
            self.model.state_dict(), save_path /
            ('{}_{}_model_iter-{}_{}.pth'.format(get_time(), extra, self.step, acc_str)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('{}_{}_head_iter-{}_{}.pth'.format(get_time(), extra, self.step, acc_str)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('{}_{}_opt_iter-{}_{}.pth'.format(get_time(), extra, self.step, acc_str)))


    def train(self, conf,):
        self.model.train()
        running_loss = 0.

        for e in range(self.start_epoch, conf.epochs):
            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()

            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                # embeddings = self.model(imgs)
                embeddings = self.model.forward(imgs)
                thetas = self.head(embeddings, labels)  # ArcFace loss add margin
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item() / conf.batch_size
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('Training/Loss', loss_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_rate', lr, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print("Test Model: ", conf.test_set)

                    accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, 1e-4)
                    for (test_set, acc) in accuracy.items():
                        self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
                    self.model.train()
                    self.save_state(conf, accuracy, extra=conf.job_name)

                self.step += 1

        #save final model's result
        accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, 1e-4)
        for (test_set, acc) in accuracy.items():
            self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
        self.save_state(conf, accuracy, extra=conf.job_name)

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

    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
