from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
from .common_utility import L2Norm, Flatten
from distiller.src.models.model import l2_norm
##################################  CombineMargin head #############################################################
class CombineMargin(Module):
    # cos_theta = cos(m1*theta + m2) - m3
    def __init__(self, embedding_size, classnum, loss_params_str):
        super(CombineMargin, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # init kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m1, self.m2 , self.m3 ,self.s = self.analysis_params(loss_params_str)

    def analysis_params(self,param_str):
        sphere_param_part = param_str.split('-')[-1]
        s = int(sphere_param_part.split('s')[-1])
        margin_part = sphere_param_part.split('s')[0]

        m1 = float(margin_part.split('m')[1])
        m2 = float(margin_part.split('m')[2])
        m3 = float(margin_part.split('m')[3])
        assert m1 > 0
        assert m2 >= 0
        assert m3 >= 0
        return m1, m2, m3, s
    def forward(self, embeddings, label):
        nB = len(embeddings)  # nB is batch_size
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        theta = torch.acos(cos_theta)

        theta = self.m1 * theta + self.m2
        cos_theta_m = torch.cos(theta) - self.m3

        output = cos_theta_m
        indx = torch.arange(0, nB, dtype=torch.long)
        output[indx, label] = cos_theta_m[indx, label]
        output *= self.s
        return output


##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, loss_params_str = None):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        m,s = self.analysis_params(loss_params_str)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def analysis_params(self,param_str):
        assert param_str is not None
        sphere_param_part = param_str.split('-')[-1]
        b_index = sphere_param_part.find('b')
        s_index = sphere_param_part.find('s')
        print(sphere_param_part)
        b = float(sphere_param_part[b_index + 1:s_index])
        s = float(sphere_param_part[s_index + 1:])
        return b, s

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


#################################SVG-Softmax Based on ArcFace  ###############################################
class SVG_softmax(Arcface):
    def __init__(self, embedding_size=512, classnum=51332, loss_params_str = None):
        super(SVG_softmax, self).__init__(embedding_size, classnum, s, m)
        self.t = t

    def forward(self, embbedings, label):
        #################ArcFace Margin Part #####################
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]

        ###########################ArcFace  Margin  Part ##########################################

        ###########################SVGSoftMax Part ################################################

        cos_svg = self.t * cos_theta + self.t - 1

        # expand label's cos value
        expand_cos_theta_m = cos_theta_m[idx_, label].expand(self.classnum, nB).transpose(0, 1)
        # create mask to find rest cos value bigger than label's cos value(which added a margin)
        svg_mask = (output - expand_cos_theta_m) > 0

        # replace values bigger than cos label to cos_svg
        output[svg_mask] = cos_svg[svg_mask]

        ###########################SVGSoftMax Part ################################################

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


#################################SVG-Softmax Use ArcMargin on rest vecotr space  ######################
############    replace  angular margin with original SVG margin for unlabeled space vector  ##########
##########################################      Based on ArcFace     ##################################
class SVGArc_softmax(Arcface):
    def __init__(self, embedding_size=512, classnum=51332, loss_params_str = None):
        super(SVGArc_softmax, self).__init__(embedding_size, classnum, s, m)
        self.m_t = m_t
        self.sin_m_t = math.sin(m_t)
        self.cos_m_t = math.cos(m_t)
        self.thre_add = math.cos(m_t)
        self.mm_add = self.sin_m_t * m_t

    def forward(self, embbedings, label):
        #################ArcFace Margin Part #####################
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]

        ###########################ArcFace  Margin  Part ##########################################

        ###########################SVGSoftMax Part ################################################

        cos_theta_add_m = (cos_theta * self.cos_m_t + sin_theta * self.sin_m_t)  # cos(theta - m_t)
        # condition controls the theta-m should in range [0, pi]
        #      0<=theta-m_t<=pi
        #      m<=theta<=pi+m       m <= theta <=pi
        mask_add = cos_theta - self.thre_add > 0
        keep_add_val = (cos_theta + self.mm_add)
        cos_theta_add_m[mask_add] = keep_add_val[mask_add]

        # expand label's cos value
        expand_cos_theta_m = cos_theta_m[idx_, label].expand(self.classnum, nB).transpose(0, 1)
        # expand_cos_theta_m = cos_theta_m[idx_, label].view(-1,1) # (batch_size, 1)
        # create mask to find rest cos value bigger than label's cos value(which added a margin)
        svg_mask = (output - expand_cos_theta_m) > 0

        # replace values bigger than cos label to cos_svg
        output[svg_mask] = cos_theta_add_m[svg_mask]

        ###########################SVGSoftMax Part ################################################

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


##################################  Cosface head #############################################################

class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, loss_params_str = None):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # additive margin recommended by the paper
        self.s = s  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class sphere_plusLoss(Module):
    def __init__(self, embedding_size, classnum, loss_params_str = None):
        '''

        :param embedding_size:
        :param classnum:
        :param type: means or among
        :param alpha:
        '''
        super(sphere_plusLoss, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.type = type
        self.alpha = alpha

    def forward(self, embeddings, label):
        batch_size = len(label)
        kernel_norm = l2_norm(self.kernel, axis=0)

        if self.type == "single":
            mean_kernel = torch.sum(kernel_norm, 1) / self.classnum  # (embedding_size, 1)
            norm_mean_kernel = l2_norm(mean_kernel, axis=0)  # (embedding_size ,1)
            norm_mean_kernel_2d = torch.stack([norm_mean_kernel], 1)
            kernel_norm_t = torch.t(kernel_norm)  # (classnum, embedding_size)

            cos_dist = torch.mm(kernel_norm_t,
                                norm_mean_kernel_2d)  # (classnum, embedding_size) * (embedding_size, 1) = (classnum, 1)
            inter_class_loss = self.alpha * torch.sum(cos_dist[label]) / batch_size


        elif self.type == "among":
            sum_sq_diff_weight = 0
            # for indx in label:
            #     label_weight = kernel_norm[:, indx]
            #     label_weight_2d = torch.stack([label_weight], 1)
            #     diff_weight =torch.add(kernel_norm , -label_weight_2d)
            #
            #     # square_diff_weight = torch.mul(diff_weight, diff_weight) #element wise multiplition
            #     square_diff_weight = torch.norm(diff_weight,2, 0, True)
            #     sum_sq_diff_weight += torch.sum

            label_kernel = kernel_norm[:, label]
            for indx in range(self.classnum):
                each_weight = torch.stack([kernel_norm[:, indx]], 1)
                diff_weight = torch.add(label_kernel, -each_weight)
                square_diff_weight = torch.mul(diff_weight, diff_weight)
                sum_sq_diff_weight += torch.sum(square_diff_weight)

            inter_class_loss = sum_sq_diff_weight * self.alpha / self.classnum

        # maxium l2 distance among all vectors in label space
        elif self.type == "ldist":
            sum_sq_diff_weight = 0
            label_kernel = kernel_norm[:, label]
            for indx in range(len(label)):
                each_label = torch.stack([label_kernel[:, indx]], 1)
                diff_weight = torch.add(each_label, -label_kernel)
                square_diff_weight = torch.mul(diff_weight, diff_weight)
                sum_sq_diff_weight += torch.sum(square_diff_weight)

            inter_class_loss = -sum_sq_diff_weight * self.alpha / len(label)

        # maxium cos distance among all vectors in label space
        elif self.type == "lcos":
            sum_cos_dist = 0
            label_kernel = kernel_norm[:, label]
            for indx in range(len(label)):
                each_label = torch.stack([label_kernel[:, indx]], 1)  # (embedding_size, 1)
                each_label_t = torch.t(each_label)  # (1, embedding_size)
                cos_dist = torch.mm(each_label_t,
                                    label_kernel)  # (1, embedding_size) * (embedding_size, label_num) = (1, label_num)

                # exclude label itself
                sel_idx = [i for i in range(len(label)) if i != indx]
                cos_dist_sel = cos_dist[:, sel_idx]
                sum_cos_dist = torch.sum(cos_dist_sel)

            inter_class_loss = sum_cos_dist * self.alpha / len(label)




        else:
            print("single or among or ldist type..")

        return inter_class_loss