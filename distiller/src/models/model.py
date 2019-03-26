from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
from .common_utility import L2Norm, Flatten

##################################  Original Arcface Model #############################################################


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

##################################  MobileFaceNet #############################################################
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Conv_block_no_bn(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block_no_bn, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size, conv6_kernel = (7,7)):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel= conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)

        out = self.l2(out)

        return out
        # return l2_norm(out)

##################################  CombineMargin head #############################################################
class CombineMargin(Module):
    #cos_theta = cos(m1*theta + m2) - m3
    def __init__(self, embedding_size, classnum, m1, m2, m3, s):
        super(CombineMargin, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        #init kernel
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.s  =  s

        assert m1 > 0
        assert m2 >=0
        assert m3 >=0

    def forward(self, embeddings, label):
        nB = len(embeddings) #nB is batch_size
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        theta = torch.acos(cos_theta)

        theta = self.m1*theta+self.m2
        cos_theta_m = torch.cos(theta) - self.m3

        output = cos_theta_m
        indx = torch.arange(0, nB, dtype = torch.long)
        output[indx, label] = cos_theta_m[indx, label]
        output *= self.s
        return output

##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

#################################SVG-Softmax Based on ArcFace  ###############################################
class SVG_softmax(Arcface):
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, t = 1.2):
        super(SVG_softmax, self).__init__(embedding_size, classnum, s,m)
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

        #expand label's cos value
        expand_cos_theta_m = cos_theta_m[idx_, label].expand(self.classnum, nB).transpose(0,1)
        #create mask to find rest cos value bigger than label's cos value(which added a margin)
        svg_mask = (output - expand_cos_theta_m) > 0

        #replace values bigger than cos label to cos_svg
        output[svg_mask] = cos_svg[svg_mask]

        ###########################SVGSoftMax Part ################################################

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

#################################SVG-Softmax Use ArcMargin on rest vecotr space  ######################
############    replace  angular margin with original SVG margin for unlabeled space vector  ##########
##########################################      Based on ArcFace     ##################################
class SVGArc_softmax(Arcface):
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, m_t=0.5):
        super(SVGArc_softmax, self).__init__(embedding_size, classnum, s,m)
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

        cos_theta_add_m = (cos_theta * self.cos_m_t + sin_theta * self.sin_m_t) #cos(theta - m_t)
        #condition controls the theta-m should in range [0, pi]
        #      0<=theta-m_t<=pi
        #      m<=theta<=pi+m       m <= theta <=pi
        mask_add = cos_theta - self.thre_add > 0
        keep_add_val = (cos_theta + self.mm_add)
        cos_theta_add_m[mask_add] = keep_add_val[mask_add]

        #expand label's cos value
        expand_cos_theta_m = cos_theta_m[idx_, label].expand(self.classnum, nB).transpose(0,1)
        #expand_cos_theta_m = cos_theta_m[idx_, label].view(-1,1) # (batch_size, 1)
        #create mask to find rest cos value bigger than label's cos value(which added a margin)
        svg_mask = (output - expand_cos_theta_m) > 0

        #replace values bigger than cos label to cos_svg
        output[svg_mask] = cos_theta_add_m[svg_mask]

        ###########################SVGSoftMax Part ################################################

        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


##################################  Cosface head #############################################################    
    
class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=51332, s=30, m=0.35):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # additive margin recommended by the paper
        self.s = s # see normface https://arxiv.org/abs/1704.06369
    def forward(self,embbedings,label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

class sphere_plusLoss(Module):
    def __init__(self, embedding_size, classnum, type, alpha):
        '''

        :param embedding_size:
        :param classnum:
        :param type: means or among
        :param alpha:
        '''
        super(sphere_plusLoss, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.type = type
        self.alpha = alpha

    def forward(self, embeddings, label):
        batch_size = len(label)
        kernel_norm = l2_norm(self.kernel, axis=0)


        if self.type == "single":
            mean_kernel = torch.sum(kernel_norm, 1)/self.classnum  #(embedding_size, 1)
            norm_mean_kernel = l2_norm(mean_kernel, axis=0) # (embedding_size ,1)
            norm_mean_kernel_2d = torch.stack([norm_mean_kernel], 1)
            kernel_norm_t = torch.t(kernel_norm) #(classnum, embedding_size)

            cos_dist = torch.mm(kernel_norm_t, norm_mean_kernel_2d) #(classnum, embedding_size) * (embedding_size, 1) = (classnum, 1)
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

            label_kernel = kernel_norm[:,label]
            for indx in range(self.classnum):
                each_weight = torch.stack([kernel_norm[:,indx]],1)
                diff_weight = torch.add(label_kernel, -each_weight)
                square_diff_weight = torch.mul(diff_weight, diff_weight)
                sum_sq_diff_weight += torch.sum(square_diff_weight)

            inter_class_loss  = sum_sq_diff_weight * self.alpha / self.classnum

        #maxium l2 distance among all vectors in label space
        elif self.type == "ldist":
            sum_sq_diff_weight = 0
            label_kernel = kernel_norm[:, label]
            for indx in range(len(label)):
                each_label = torch.stack([label_kernel[:,indx]],1)
                diff_weight = torch.add(each_label, -label_kernel)
                square_diff_weight = torch.mul(diff_weight, diff_weight)
                sum_sq_diff_weight += torch.sum(square_diff_weight)

            inter_class_loss  = -sum_sq_diff_weight * self.alpha / len(label)

        #maxium cos distance among all vectors in label space
        elif self.type == "lcos":
            sum_cos_dist = 0
            label_kernel = kernel_norm[:, label]
            for indx in range(len(label)):
                each_label = torch.stack([label_kernel[:,indx]],1) #(embedding_size, 1)
                each_label_t = torch.t(each_label)  #(1, embedding_size)
                cos_dist = torch.mm(each_label_t, label_kernel) #(1, embedding_size) * (embedding_size, label_num) = (1, label_num)

                #exclude label itself
                sel_idx = [i for i in range(len(label)) if i != indx ]
                cos_dist_sel = cos_dist[:, sel_idx]
                sum_cos_dist = torch.sum(cos_dist_sel)

            inter_class_loss  = sum_cos_dist * self.alpha / len(label)




        else:
            print("single or among or ldist type..")

        return inter_class_loss
