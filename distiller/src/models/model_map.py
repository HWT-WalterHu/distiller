'''
Map Net work name with function
'''

from .import model as train_net
from .import model_lib as train_lib
from .import resNet as resNet
from .import densenet
from .MnasNet import MnasNet
from easydict import EasyDict as edict

model_map = edict()
model_map.MobileFaceNet = train_net.MobileFaceNet
model_map.FaceNet20 = train_lib.FaceNet_20
model_map.FaceNetOrg20 = train_lib.FaceNet_Origin_20
model_map.MNasNetS1 = MnasNet
model_map.ResNet18 = resNet.resnet18
model_map.FaceNetcf20 = train_lib.FaceNet_cf_20
model_map.ResNet18v3 = resNet.resnet18_v3
model_map.ResNet18v3fc = resNet.resnet18_v3_2

model_map.ResNet34pl = resNet.resnet34_0
model_map.ResNet34cv = resNet.resnet34
model_map.ResNet34v3 = resNet.resnet34_v3
model_map.DenseNet121 = densenet.densenet121


# model_map.ResNet50pl = resNet.resnet50_0
# model_map.ResNet50cv = resNet.resnet50
# model_map.ResNet50fc = resNet.resnet50_2
# model_map.ResNet50v3fc = resNet.resnet50_v3_2
#
# model_map.fResNet50pl = resNet.fresnet50_0
# model_map.fResNet50v3cv = resNet.fresnet50_v3
# model_map.fResNet50fc = resNet.fresnet50_2
# model_map.fResNet50v3fc = resNet.fresnet50_v3_2

model_map.FaceNet36 = train_lib.FaceNet_36
model_map.FaceNet60 = train_lib.FaceNet_60
model_map.FaceNet104 = train_lib.FaceNet_104


loss_map = edict()
loss_map["ArcFace95to5"] = train_net.Arcface
loss_map["SpArcFace95to5"] = train_net.Arcface
loss_map["AMLoss95to5"] = train_net.Am_softmax
loss_map["CombineMargin"] = train_net.CombineMargin
loss_map["SVGArcFace"] = train_net.SVG_softmax    #origin SVGSoftmaxLoss
loss_map["SVGAngArcFace"] = train_net.SVGArc_softmax  #replace the margin with auglar margin
loss_map["SpSVGArcFace"] = train_net.SVG_softmax
loss_map["SpSVGAngFace"] = train_net.SVGArc_softmax

