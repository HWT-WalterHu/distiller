'''
Use to select best accuracy model
'''
#2019-03-14 mod by zkx add create_model_info function: create roc statastic log file after select best models

import sys
import os
sys.path.append(os.getcwd())
import glob
import os
import src.utils
import argparse
from src.roc_evaluate.RocCurve import RocCurve
from src.utils import transfrom_model_name, make_if_not_exist, get_time
from src.config_path import ConfigPath




def create_model_info(result_folder_path):
    log_path = "{}/roc_info_{}.log".format(result_folder_path, get_time())
    roc_curve = RocCurve()
    f = open(log_path, 'w')
    for elem in os.listdir(result_folder_path):
        if elem.find(".npy") >=0:
            result_path = "{}/{}".format(result_folder_path, elem)
            roc_info, info_dict = roc_curve.make_roc_curve(result_path)
            f.write("{}\n".format(elem))
            for line in roc_info:
                f.write("{}\n".format(line))
            f.write("\n==============================================================\n\n")
    f.close()

def copy_testset_result(result_path, select_model_path, dst_convert_path, testset_prefix):
    select_model_name = select_model_path.split('/')[-1]
    select_folder_name = select_model_path.split('/')[-2]
    select_prefix = select_model_path.split('/')[-3]

    #model_name
    #2019-02-22-18-35_SpArcFace95to5-b0.4s40Pm3.3_fc_0.35_112x96_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_FaceNetOrg20-d512_model_iter-2500_Bus-0.1139_XCH-0.1448
    trans_name = transfrom_model_name(select_model_name)
    # print(trans_name)
    src_copy_path = "{}/{}/{}".format(result_path, select_prefix, select_folder_name)

    for elem in os.listdir(src_copy_path):
        if elem.find(testset_prefix) >=0:
            each_result_path = "{}/{}".format(src_copy_path, elem)
            dst_copy_path = "{}/{}".format(dst_convert_path, elem)

            for file in glob.glob("{}/result_v2_{}*".format(each_result_path, trans_name)):
                make_if_not_exist(dst_copy_path)
                src.utils.copy_if_exists(file, dst_copy_path)


#2018-11-30-09-15_ArcFace95to5-b0.5s64_fc_0.4_112x112_del-LAN-DHUA-Crop-GE15-add-KdGE8_MobileFaceNet_opt_iter-197500_Bus-0.1528_XCH-0.1504.pth
def select_best_model(root_folder_path, dst_path, result_path, start_date ,select_top=3, beyond = 0.5):

    folder_prefix = root_folder_path.split('/')[-1]
    dst_model_path = "{}/{}".format(dst_path, folder_prefix)
    dst_convert_path = "{}_convert_caffemodel".format(dst_path)

    sum_result = {}
    for elem in os.listdir(root_folder_path):

        current_date = elem.split('_')[0]
        if elem.find('model') >=0 and \
                src.utils.whether_beyond_date(start_date, current_date):

            # print(current_date)
            #197500_Bus-0.1528_XCH-0.1504
            test_result_part = elem.split('iter-')[-1].strip('.pth')
            # iter_num = test_result_part.split('_')[0]

            for index, part in enumerate(test_result_part.split('_')):
                if index == 0:
                    continue
                test_set = part.split('-')[0]
                acc = float(part.split('-')[1])
                if test_set not in sum_result.keys():
                    sum_result[test_set] = []
                sum_result[test_set].append((elem, acc))

    #sort result
    for key, value in sum_result.items():
        value.sort(key=lambda k:k[1], reverse=True)

    #select top n model
    for key, value in sum_result.items():
        print(key)
        select_top = min(len(value), select_top)
        for i in range(select_top):
            model_name = value[i][0]
            model_prefix = model_name.split("_model_")[0]
            acc = value[i][1]
            if acc > beyond:
                print(model_name, acc)
                src.utils.make_if_not_exist(dst_model_path)
                opt_name = model_name.replace('model', 'opt')
                head_name = model_name.replace('model', 'head')
                src.utils.copy_if_exists('{}/{}'.format(root_folder_path, model_name),
                                         '{}/{}'.format(dst_model_path, model_name))
                src.utils.copy_if_exists('{}/{}'.format(root_folder_path, opt_name),
                                         '{}/{}'.format(dst_model_path, opt_name))
                src.utils.copy_if_exists('{}/{}'.format(root_folder_path, head_name),
                                         '{}/{}'.format(dst_model_path, head_name))

                #copy test result
                copy_testset_result(result_path,
                                '{}/{}'.format(root_folder_path, model_name),
                                    dst_convert_path, key)


def travel_folder(root_path, dst_path, result_path, start_date, select_top = 3, beyond = 0.5):
    current_time = src.utils.get_time()
    dst_current_path = "{}/{}".format(dst_path, current_time)
    dst_convert_path = "{}_convert_caffemodel".format(dst_current_path)
    for folder_elem in os.listdir(root_path):
        sub_folder_path = '{}/{}'.format(root_path, folder_elem)
        for sub_elem in os.listdir(sub_folder_path):
            model_path = '{}/{}'.format(sub_folder_path, sub_elem)
            if os.path.isdir(model_path):
                select_best_model(model_path, dst_current_path,result_path,  start_date, select_top, beyond)

    #add result log
    for elem in os.listdir(dst_convert_path):
        folder_path = "{}/{}".format(dst_convert_path, elem)
        if os.path.isdir(folder_path) and elem.find("Result_") >=0:
            create_model_info(folder_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-date", "--date", default='2019-03-13-09-00', help="train model date 2019-01-14-00-00", type=str)
    parser.add_argument("-num", "--num", help="select top n num", default=4, type=int)
    parser.add_argument("-beyond", "--beyond", help="select model beyond threshold", default=0.3, type=float)
    args = parser.parse_args()

    root_folder_path = ConfigPath.root_folder_path
    dst_path = ConfigPath.dst_path
    result_path = ConfigPath.result_path

    start_date = args.date
    select_top = args.num
    beyond = args.beyond
    # start_date = "2018-11-30-00-00"
    # select_top = 3
    travel_folder(root_folder_path, dst_path, result_path, start_date, select_top, beyond)





