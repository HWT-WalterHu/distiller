from distiller.src.train.train_face_recognition import TrainFace
from distiller.src.models.model import sphere_plusLoss
from distiller.src.models.model_map import loss_map
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from distiller.src.utils import get_time

class TrainFaceSpherePlus(TrainFace):
    def __init__(self, conf):
        super(TrainFaceSpherePlus, self).__init__(conf)

    def train_batch_data(self, imgs, labels):
        imgs = imgs.to(self.conf.device)
        labels = labels.to(self.conf.device)
        self.optimizer.zero_grad()
        # embeddings = self.model(imgs)
        embeddings = self.model.forward(imgs)
        thetas = self.head(embeddings, labels)  # ArcFace loss add margin
        loss = self.conf.ce_loss(thetas, labels)
        inter_loss = self.head_sp(embeddings, labels)
        loss.backward()
        inter_loss.backward()
        self.optimizer.step()
        return loss.item() / self.conf.batch_size, inter_loss / self.conf.batch_size


    def train_stage(self):
        self.model.train()
        running_loss = 0.
        inter_loss = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):

            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                # write graps in to tensorboard
                net_input = torch.FloatTensor(1, 3, self.conf.input_size[0], self.conf.input_size[1]).zero_()
                self.writer.add_graph(self.model.modules(), net_input.to(self.conf.device))
                is_first = False

            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                tmp_running_loss , tmp_inter_loss= self.train_batch_data(imgs, labels)
                running_loss += tmp_running_loss
                inter_loss += tmp_inter_loss

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    inter_loss_board = inter_loss / self.board_loss_every
                    self.writer.add_scalar('Training/Loss', loss_board, self.step)
                    self.writer.add_scalar('Training/Inter_loss', inter_loss_board, self.step)

                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_rate', lr, self.step)
                    running_loss = 0.
                    inter_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print("Test Model: ", self.conf.test_set)

                    time_stamp = get_time()
                    accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, time_stamp)

                    for (test_set, acc) in accuracy.items():
                        self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
                    self.model.train()
                    self.save_state(accuracy, time_stamp, extra=self.conf.job_name)

                self.step += 1

        #save final model's result
        accuracy = self.roc_evaluate.evaluate_model(self.model, self.step, 1e-6)
        for (test_set, acc) in accuracy.items():
            self.writer.add_scalar("{}".format("Evaluate/{}".format(test_set)), acc, self.step)
        self.save_state(accuracy, time_stamp, extra=self.conf.job_name)
        self.writer.close()


    def loss_type(self):
        print("Loss", self.conf.loss_type, self.conf.scale, self.conf.bias)
        print("SpherePlus", self.conf.sp_type, self.conf.alpha)
        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias).to(self.conf.device)
        self.head_sp = sphere_plusLoss(embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                       type=self.conf.sp_type, alpha=self.conf.alpha).to(self.conf.device)


class TrainFaceSphereSVGSoftMax(TrainFaceSpherePlus):
    def __init__(self, conf):
        super(TrainFaceSphereSVGSoftMax, self).__init__(conf)

    def loss_type(self):
        print("Loss ", self.conf.loss_type, self.conf.scale, self.conf.bias, self.conf.t)
        print("SpherePlus", self.conf.sp_type, self.conf.alpha)

        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias, t = self.conf.t).to(self.conf.device)
        self.head_sp = sphere_plusLoss(embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                       type=self.conf.sp_type, alpha=self.conf.alpha).to(self.conf.device)

class TrainFaceSphereSVGAngSoftmax(TrainFaceSpherePlus):
    def __init__(self, conf):
        super(TrainFaceSphereSVGAngSoftmax, self).__init__(conf)

    def loss_type(self):
        print("Loss ", self.conf.loss_type, self.conf.scale, self.conf.bias, self.conf.t)
        print("SpherePlus", self.conf.sp_type, self.conf.alpha)

        self.head = loss_map[self.conf.loss_type](embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                             s = self.conf.scale, m=self.conf.bias, m_t= self.conf.t).to(self.conf.device)
        self.head_sp = sphere_plusLoss(embedding_size=self.conf.embedding_size, classnum=self.class_num,
                                       type=self.conf.sp_type, alpha=self.conf.alpha).to(self.conf.device)

