'''
sub class from train_base
'''

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

class TrainFace(TrainBase):
    def __init__(self, conf):
        super(TrainFace,self).__init__(conf)
        # init roc running time evalute
        self.roc_evaluate = Run_Time_Evaluate(conf)

    def init_model_param(self):
        #decide net structure
        self.define_network()
        self.loss_type()
        self.SGD_opt()
        self.fintune_model()

    def train_stage(self):
        self.model.train()
        running_loss = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):

            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                # write graps in to tensorboard
                net_input = torch.FloatTensor(1, 3, self.conf.input_size[0], self.conf.input_size[1]).zero_()
                self.writer.add_graph(self.model_board, net_input)

                is_first = False

            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                # continue
                running_loss += self.train_batch_data(imgs, labels)

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
        self.train_stage()




class TrainFaceCombineMargin(TrainFace):
    def __init__(self, conf):
        super(TrainFaceCombineMargin, self).__init__(conf)

    def loss_type(self):
        print(self.conf.m1, self.conf.m2, self.conf.m3, self.conf.scale)
        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             m1 = self.conf.m1, m2 = self.conf.m2, m3 = self.conf.m3,
                                                  s = self.conf.scale).to(self.conf.device)

class TrainFaceSVGSoftmax(TrainFace):
    def __init__(self, conf):
        super(TrainFaceSVGSoftmax, self).__init__(conf)

    def loss_type(self):
        print(self.conf.loss_type, self.conf.scale, self.conf.bias, self.conf.t)
        # conf.t means original  margin
        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias, t = self.conf.t).to(self.conf.device)


class TrainFaceSVGAngSoftmax(TrainFace):
    def __init__(self, conf):
        super(TrainFaceSVGAngSoftmax, self).__init__(conf)

    def loss_type(self):
        print(self.conf.loss_type, self.conf.scale, self.conf.bias, self.conf.t)
        #conf.t means Auglar angle margin
        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias, m_t= self.conf.t).to(self.conf.device)














