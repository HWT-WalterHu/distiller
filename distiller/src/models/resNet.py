import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .common_utility import L2Norm , Flatten, Get_Conv_Size

__all__ = ['ResNet', 'resnet18','resnet18_v1', 'resnet18_v3','resnet18_v3_2', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu_res = nn.ReLU(inplace=True)
        # self.relu_res = nn.ReLU()
        # self.relu_res = nn.PReLU(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu_res(out)

        return out


class BasicBlock_v3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_v3, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        # self.relu_res = nn.ReLU(inplace=True)
        # self.relu_res = nn.ReLU()
        # self.relu_res = nn.PReLU(planes)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#This is for input size 224x224. the first conv 7x7 will inflence a lot for 112x112 input, dont use it
# class ResNet_v0(nn.Module):

#
#     # def __init__(self, block, layers, num_classes=1000):
#     def __init__(self, block, layers, input_size=224):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         # self.relu = nn.ReLU()
#         # self.relu = nn.PReLU(64)
#
#         #In caffe ceil_mode is always True
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)
#
#         self.avgpool = nn.AvgPool2d(fea_size, stride=1)
#         self.flattern = Flatten()
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.l2 = L2Norm()   #add l2 norm layer
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch
#
#         downsample = nn.Sequential(
#             nn.Conv2d(self.inplanes, planes * block.expansion,
#                       kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(planes * block.expansion),
#         )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = self.flattern(x)
#         # x = self.fc(x)
#         x = self.l2(x)  #add l2 norm layer
#
#         return x



# suitable for input size 112x112
class ResNet(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, input_size=224, embedding_size = 512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        #In caffe ceil_mode is always True
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)

        # self.avgpool = nn.AvgPool2d(fea_size, stride=1)
        # self.flattern = Flatten()
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 =  nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Conv2d(512,embedding_size,(7,7),bias=False)
        self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        self.l2 = L2Norm()   #add l2 norm layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn3(x)
        x = self.dp(x)
        x = self.fc1(x)
        # print("fc", x.size())

        x = self.ft1(x)
        # print("ft", x.size())

        x = self.bn4(x)
        x = self.l2(x)
        # x = self.avgpool(x)
        # x = self.flattern(x)
        # # x = self.fc(x)
        # x = self.l2(x)  #add l2 norm layer

        return x


class ResNet_2(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, input_size=224, embedding_size = 512):
        print("embedding size %s"%embedding_size)
        self.inplanes = 64
        super(ResNet_2, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        #In caffe ceil_mode is always True
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)

        # self.avgpool = nn.AvgPool2d(fea_size, stride=1)

        self.bn3 =  nn.BatchNorm2d(512*block.expansion, eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        self.flattern = Flatten()
        self.fc = nn.Linear(512*7*7*block.expansion, embedding_size)
        # self.fc1 = nn.Conv2d(512,512,(7,7),bias=False)
        # self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        self.l2 = L2Norm()   #add l2 norm layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn3(x)
        x = self.dp(x)
        x = self.flattern(x)
        # print("fc", x.size())
        x = self.fc(x)
        # print("fc", x.size())

        # x = self.ft1(x)
        # print("ft", x.size())

        x = self.bn4(x)
        x = self.l2(x)
        # x = self.avgpool(x)
        # x = self.flattern(x)
        # # x = self.fc(x)
        # x = self.l2(x)  #add l2 norm layer

        return x


class ResNet_0(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, input_size=224, embedding_size = 512):
        print("embedding size %s"%embedding_size)
        self.inplanes = 64
        super(ResNet_0, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.flatten = Flatten()
        self.l2 = L2Norm()   #add l2 norm layer
        # self.last_linear = nn.Linear(512 * block.expansion, embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    # def logits(self, features):
    #     x = self.avgpool(features)
    #     x = x.view(x.size(0), -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        # print("avgpool shape:",x.shape)
        x = self.flatten(x)
        x = self.l2(x)
        # print("feature shape:",x.shape)

        # x = self.logits(x)
        return x

# def resnet18(pretrained=False, **kwargs):
def resnet18(input_size):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_0(BasicBlock, [2, 2, 2, 2], input_size)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

# def resnet18_v1(input_size):
#     """Constructs a ResNet-18 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet_0(BasicBlock, [2, 2, 2, 2], input_size)
#
#     return model

#resnet18v3 from insightface unit_v3, using conv replacing the last fullyconnected
def resnet18_v3(input_size, embedding_size = 512):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock_v3, [2, 2, 2, 2], input_size, embedding_size)
    # model = ResNet(BasicBlock, [2, 2, 2, 2], input_size)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

#using flatten and linear to replace the last fullyconnected
def resnet18_v3_2(input_size):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_2(BasicBlock_v3, [2, 2, 2, 2], input_size)
    # model = ResNet(BasicBlock, [2, 2, 2, 2], input_size)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

# trunk version: _0:pool(7) and flatten; _2: flatten and FC, default: conv(7,7)and  flatten
# block version: v3: insighface version v3, default: author version(but modified first block stride from 1 to 2)
def resnet34_0(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_0(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet34_v3(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock_v3, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_0(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50_2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_2(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_0(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

