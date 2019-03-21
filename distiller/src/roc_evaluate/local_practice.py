'''
Use to test select models
'''
import sys
import os
project_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_path)
import torch
from src.model_config.data_set import TestSet
from src.models.model_map import  model_map
from src.data_io.data_loader import get_batch_test_data
from src.utils import get_test_param_from_modelname,make_if_not_exist, get_time, transfrom_model_name
import numpy as np
from src.roc_evaluate.RocCurve import RocCurve
import os
import time
import argparse
from src.config_path import ConfigPath


class LocalPractice():
    def __init__(self, test_set_list, dst_path, device_num, batch_size=100, worker_num=16):
        self.dst_path = dst_path
        self.device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.embedding_result = {}
        self.test_set_list = test_set_list

    def set_testset(self, test_set):
        self.testset = test_set
        self.image_list_path = TestSet[test_set]["image_list"]
        self.image_root_path = TestSet[test_set]["root_path"]
        image_pair_path = TestSet[test_set]["image_pairs"]
        f = open(image_pair_path, 'r')
        self.image_pairs = f.read().splitlines()
        f.close()

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
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def extract_dataloader_feature(self, model_path):
        # print('Data Loader Extract Feature')
        if len(self.embedding_result) >0:
            self.embedding_result = {}

        self.model_name = model_path.split('/')[-1].strip('.pth')
        net_info, h_value, w_value, patch_info = \
            get_test_param_from_modelname(self.model_name)
        # print(net_info, h_value, w_value, patch_info)

        #"NetName-d200-k-9-8"
        net_name = net_info.split('-d')[0]
        if net_info.find("MobileFaceNet") >=0:
            if net_info.find('-k-') >=0:
                dim = int(net_info.split('-d')[1].split('-')[0])
                kernel_part = net_info.split('-k-')[-1]
                kernel = (int(kernel_part.split('-')[0]), int(kernel_part.split('-')[1]))
                print(kernel)
                self.model = model_map[net_name](dim, kernel).to(self.device)
            else:
                dim = int(net_info.split('-d')[1])
                self.model = model_map[net_name](dim).to(self.device)

        elif net_info.find("FaceNet20") >= 0 or net_info.find("FaceNetOrg20")>=0:
            dim = int(net_info.split('-d')[1])
            self.model = model_map[net_name](dim, h_value, w_value).to(self.device)

        elif net_info.find("ResNet") >=0:
            assert h_value == w_value
            # dim = int(net_info.split('-d')[1])
            self.model = model_map[net_name](h_value).to(self.device)


        data_loader = get_batch_test_data('{}/{}'.format(self.image_root_path,patch_info),
                                          self.image_list_path, self.batch_size, self.worker_num)
        self.load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            for (img, img_prefix) in data_loader:
                emb_vec = self.model.forward(img.to(self.device))
                for idex in range(len(emb_vec)):
                    self.embedding_result[img_prefix[idex]] = emb_vec[idex].cpu().numpy()

    def make_comparation(self):
        # print('Make comparation')
        compare_result = []
        for index, line in enumerate(self.image_pairs):
            image_name1 = line.split(' ')[0]
            image_name2 = line.split(' ')[1]
            label = float(line.split(' ')[2])

            feature1 = self.embedding_result[image_name1]
            feature2 = self.embedding_result[image_name2]
            diff = np.subtract(feature1, feature2)
            dist = np.sum(np.square(diff), 0, dtype=np.float32)
            compare_result.append([dist, label])
        # print('Save distance result')
        save_result_path = '{}/Result_{}'.format(self.dst_path, self.testset)
        make_if_not_exist(save_result_path)

        trans_name = transfrom_model_name(self.model_name)
        #save in npy format
        save_dist_path = "{}/result_v2_{}".format(save_result_path, trans_name)
        np.save(save_dist_path, np.array(compare_result))

        draw_roc = RocCurve()
        roc_info, dict_info = draw_roc.make_roc_curve(save_dist_path + ".npy")
        return roc_info, dict_info

    def test_folder_models(self, model_folder_path):
        for testset in self.test_set_list:
            print(testset)
            save_log_path = "{}/Result_{}/roc_info_{}.log".format(self.dst_path, testset, get_time())
            parent_path = os.path.abspath(os.path.join(save_log_path, ".."))
            make_if_not_exist(parent_path)

            f = open(save_log_path, 'w')

            self.set_testset(testset)
            for root_path, folder_path, file_path in os.walk(model_folder_path):
                for elem in file_path:
                    #2018-12-06-10-31_ArcFace95to5-b0.4s30_fc_0.4_112x112_del-LAN-DHUA-Crop-GE15-add-KdGE8_MobileFaceNet-d128_opt_iter-210000_Bus-0.7764_XCH-0.9136
                    if elem.endswith('.pth') and elem.find("model") >=0:
                        print(elem)
                        model_path = '{}/{}'.format(root_path, elem)

                        start = time.time()
                        self.extract_dataloader_feature(model_path)
                        fea_cost = (time.time() - start)

                        start = time.time()
                        roc_info, dict_info = self.make_comparation()
                        cmp_cost= (time.time() - start)
                        print("Sum cost : {:.2f}s  [feature: {:.2f}s / compare: {:.2f}s]".format(fea_cost+cmp_cost, fea_cost, cmp_cost))

                        f.write("{}\n".format(elem))
                        for line in roc_info:
                            f.write('{}\n'.format(line))
            f.close()


if __name__ == "__main__":

    testset_map = {
        "1": "Child-95to5",
        "2": "XCH-mtcnn-09-01-29",
        "3": "Business-mtcnn-95to5",
        "4": "XCH-mtcnn-outdoor3-95to5-d"
    }
    testset_helper = ""
    for key, value in testset_map.items():
        testset_helper += "{}-{}, ".format(key, value)

    parser = argparse.ArgumentParser(description='for local practice')
    parser.add_argument("-date", "--date", default="", help=" test model's date 2019-01-14-00-00", type=str)
    parser.add_argument("-set", "--set", help=testset_helper, default="3", type=str)
    parser.add_argument("-device", "--device", help="select GPU id", default=7, type=int)
    parser.add_argument("-batch", "--batch", help="test batch size", default=100, type=int)
    parser.add_argument("-worker", "--worker", help="number of workers", default=16, type=int)
    args = parser.parse_args()

    args.date= '2019-03-14-08-56'
    test_set = []
    for elem in args.set:
        test_set.append(testset_map[elem])
    folder_path = "{}/{}".format(ConfigPath.test_folder_path, args.date)
    dst_path = "{}_convert_caffemodel".format(folder_path)

    print("date: ", args.date)
    print("set: ", test_set)
    print("device: ", args.device)
    print("batch_size: ", args.batch)
    print("worker_num: ", args.worker)

    local_test = LocalPractice(test_set, dst_path, args.device)
    local_test.test_folder_models(folder_path)




