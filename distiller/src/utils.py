from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import torch
import shutil
import torch.nn.init as init

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')


def transfrom_model_name(pymodel_name):
    prefix = pymodel_name.split('_iter-')[0]
    iter_num = pymodel_name.split('_iter-')[1].split('_')[0]
    trans_model_name = prefix.replace('model', 'pytorch') + '_iter_' + iter_num
    return trans_model_name

def split_date(date_str):
    year = int(date_str.split('-')[0])
    month = int(date_str.split('-')[1])
    day = int(date_str.split('-')[2])
    hour = int(date_str.split('-')[3])
    min = int(date_str.split('-')[4])
    return [year, month, day, hour, min]

def find_most_recent_model(save_model_path):
    # model name's format
    #2018-11-25-14-30_ArcFace95to5-b0.3s30_fc_0.4_112x112_PAKJ_MobileFaceNet_model_iter-40_Bus-0.0014_XCH-0.0000.pth
    most_recent_date = ""
    most_recent_model = ""
    for elem in os.listdir(save_model_path):
        if elem.find("opt") >=0:
            elem = elem.strip('.pth')
            date = elem.split('_')[0]
            date_parts = split_date(date)
            if most_recent_date == "":
                most_recent_date = date_parts
                most_recent_model = elem
            else:
                equal = True  # whether elem in most_recent_date smaller than new coming date_parts
                for indx, each_date in enumerate(date_parts):
                    if equal:
                        if each_date > most_recent_date[indx]:
                            most_recent_date = date_parts
                            most_recent_model = elem
                            break
                        elif each_date == most_recent_date[indx]:
                            equal = True
                        else:
                            equal = False
                    else:
                        break
    return most_recent_model

def whether_beyond_date(standard_date, current_date):
    std_part = split_date(standard_date)
    cur_part = split_date(current_date)
    equal = True

    for indx, std_elem in enumerate(std_part):
        if equal:
            if cur_part[indx] > std_elem:
                return True
            elif cur_part[indx] == std_elem:
                equal = True
            else:
                equal = False
        else:
            return False

    return equal
#2018-12-01-15-49_ArcFace95to5-b0.3s64_fc_0.4_112x112_
# del-LAN-DHUA-Crop-GE15-add-KdGE8_MobileFaceNet-d100_opt_iter-265000_
# Bus-0.7625_XCH-0.9248.pth

def get_test_param_from_modelname(model_name):

    prefix = model_name.split('_iter')[0]
    net_info = prefix.split('_')[-2]

    # get patch info
    prepart_split = prefix.split('x')[0]

    center_id_part = prepart_split.split('_')[2:-2]
    ratio = prepart_split.split('_')[-2]
    h_value = prepart_split.split('_')[-1]
    w_value = prefix.split('x')[1].split('_')[0]

    center_id_str = ""
    for elem in center_id_part:
        center_id_str += '{}_'.format(elem)
    patch_info = '{}{}_{}x{}'.format(center_id_str, ratio, h_value, w_value)


    return   net_info, int(h_value), int(w_value), patch_info

def get_training_param_from_filename(py_file_name):
    dataset = os.path.splitext(py_file_name)[0].split('_')[-2]
    net_info = os.path.splitext(py_file_name)[0].split('_')[-1]

    # get patch info
    file_basename = os.path.splitext(py_file_name)[0]  # FakeFace_fc_0.3_75x60_DeepID
    prepart_split = file_basename.split('x')[0]

    center_id_part = prepart_split.split('_')[1:-2]
    ratio = prepart_split.split('_')[-2]
    h_value = prepart_split.split('_')[-1]
    w_value = file_basename.split('x')[1].split('_')[0]

    center_id_str = ""
    for elem in center_id_part:
        center_id_str += '{}_'.format(elem)
    patch_info = '{}{}_{}x{}'.format(center_id_str, ratio, h_value, w_value)

    # get ArcFace training param
    train_subject = file_basename.split('_')[0]  # FakeFace
    loss_type = train_subject.split('-')[0]

    return  dataset, net_info, train_subject, loss_type, h_value, w_value, patch_info

def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def copy_if_exists(src_path, dst_path):
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)


def reinit_certain_layer_param(remove_param_list, state_dict, embedding_size):

        from collections import OrderedDict
        removed_state_dict = OrderedDict()
        for key, value in state_dict.items():

            if key.find('module.') >= 0:
                key = key.split('module.')[-1]

            prefix = key.split('.')[0]
            if prefix not in remove_param_list: #key conv1.weight conv1.bias
                removed_state_dict[key] = value
            else:
                print("reinit: ", key)
                if len(value.size()) > 1:
                    rows = value.size()[0]
                    num_dim = embedding_size
                    new_value = torch.randn(num_dim, rows)
                    init.xavier_normal_(new_value)
                else:
                    new_value = torch.randn(num_dim)

                removed_state_dict[key] = new_value
        return removed_state_dict