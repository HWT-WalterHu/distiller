'''
Pytorch training examples
'''
import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from easydict import EasyDict as edict
from torch.nn import CrossEntropyLoss
from datetime import datetime
from distiller.src.utils import get_training_param_from_filename, make_if_not_exist
from distiller.src.train.train_KnowledgeDistillation import TrainKnowledgeDistill as TrainFace
import torch
import os
import shutil
import argparse
import torch._utils
import distiller.src.train.myparser as parser


global msglogger
# msglogger = None

#training params

# training param
def train(device_ids, resume_training, args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    conf = edict()
    conf.lr = 1e-2
    conf.batch_size = 256 #
    # conf.gamma = 0.1
    # conf.milestones = [8, 15, 25]  # down learing rate
    conf.gamma = 0.3162
    conf.milestones = [15, 20, 25]  # down learing rate
    conf.epochs = 30
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 16
    conf.ce_loss = CrossEntropyLoss()

    conf.board_loss_num = 10  # average loss among number of batches
    conf.snapshot = 50

    conf.test_num = 50
    conf.test_batch_size = 100
    conf.test_worker_num = 16
#    conf.test_set = ["Business_small_95to5", "XCH_out2_95to5"]
    conf.test_set = {"Business-mtcnn-95to5":1e-6}

    conf.device_ids = device_ids
    conf.device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    conf.net_depth = 50
    conf.drop_ratio = 0.6


    #conf.remove_layer_name = ['linear', 'bn']
    conf.remove_layer_name = []
    conf.ignore_layer = []
    conf.fintune_model_path =None
    #conf.fintune_model_path = None
    conf.resume_training = resume_training
    conf.opt_prefix = None


    #get Training DataSet and Training Network
    abs_py_file_name = os.path.abspath(__file__)
    py_file_name = abs_py_file_name.split('/')[-1]
    dataset, net_info, train_subject, loss_type, h_value, w_value, patch_info =\
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

    snapshot_root = '/media/hwt/492c0a80-02ce-45ac-9297-63ce30dfdd81/home/minivision/Project'
    job_name = "{}_{}_{}_{}".format(train_subject, patch_info, dataset, net_info)
    snapshot_dir = '{}/pytorch_models/FaceRecognition/snapshot/{}/{}'.format(snapshot_root,loss_type, job_name)
    test_result_dir = "{}/pytorch_models/FaceRecognition/run_time_test/{}/{}".format(snapshot_root,loss_type, job_name)
    log_path = './jobs/{}/{}/{}'.format(loss_type, job_name, current_time)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(test_result_dir)
    make_if_not_exist(log_path)

    #copy train_example_file to job file
    base_name = os.path.basename(py_file_name)
    log_parent_path = os.path.abspath(os.path.join(log_path, ".."))
    shutil.copy(abs_py_file_name, "{}/{}-{}".format(log_parent_path, current_time, base_name))

    # save directory path
    conf.model_path = Path(snapshot_dir)
    conf.log_path = log_path
    conf.test_roc_path = test_result_dir
    conf.job_name = job_name

    #net info
    conf.input_size = [int(h_value), int(w_value)]
    conf.net_mode = net_info.split('-')[0]
    conf.embedding_size = int(net_info.split('-')[1].split('d')[-1])
    conf.teacher_net_mode = None
    if len(net_info.split('-')) > 2 and 'teacher' in net_info.split('-')[-1]:
        conf.teacher_net_mode = net_info.split('-')[-1].replace('teacher', '')
    conf.teacher_model_path = '/media/hwt/492c0a80-02ce-45ac-9297-63ce30dfdd81/home/minivision/Project/pytorch_models/FaceRecognition/snapshot/ArcFace95to5/ArcFace95to5-P9-b0.4s40_fc_0.35_112x96_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_FaceNet36-d512/2019-03-17-16-41_ArcFace95to5-P9-b0.4s40_fc_0.35_112x96_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_FaceNet36-d512_model_iter-140000_Bus-0.6695.pth'


    if conf.teacher_model_path != None:
        teacher_net_mode = conf.teacher_model_path.split('/')[-1].split('_model')[0].split('_')[-1].split('-')[0]
        assert teacher_net_mode == conf.teacher_net_mode
    teacher_file_name = conf.teacher_model_path.split('/')[-1]
    t_loss_info = teacher_file_name.split('_')[1]
    t_loss_type = t_loss_info.split('-')[0]
    # set ArcFace loss param  bias, scale_value
    sphere_param_part = t_loss_info.split('-')[-1]
    b_index = sphere_param_part.find('b')
    s_index = sphere_param_part.find('s')
    bias = float(sphere_param_part[b_index + 1:s_index])
    scale_value = float(sphere_param_part[s_index + 1:])
    conf.teacher_bias = bias
    conf.teacher_scale = scale_value
    conf.teacher_loss_type = t_loss_type
    if net_info.find('-k-') >=0:
        kernel_part = net_info.split('-k-')[-1]
        kernel = (int(kernel_part.split('-')[0]), int(kernel_part.split('-')[1]))
        print("Kernel", kernel)
        conf.kernel = kernel

    conf.loss_type = loss_type
    conf.patch_info = patch_info
    conf.data_mode = dataset



    learner = TrainFace(conf, args)
    learner.train_model()

if __name__ == "__main__":
    argsparser = parser.get_parser()
    group = argsparser.add_argument_group('Training Arguments')
    # group.add_argument("-device", "--device_ids", help="which gpu id, 0123",default="0", type=str)
    group.add_argument("-resume", "--resume_training", help="resume training 0-not or 1-resume", default=0, type=int)
    args = argsparser.parse_args()
    devices = [int(elem) for elem in args.gpus.split(',')]
    if args.resume_training == 0:
        resume_training = False
    elif args.resume_training == 1:
        resume_training = True
    train(devices, args.resume_training, args)




