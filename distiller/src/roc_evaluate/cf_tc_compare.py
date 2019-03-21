'''
Use to test select models
'''
import sys
import os
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)
import torch
from src.model_config.data_set import TestSet
from src.models.model_map import  model_map
from src.data_io.data_loader import get_batch_test_data
from src.utils import get_test_param_from_modelname,make_if_not_exist, get_time
import numpy as np
from src.roc_evaluate.RocCurve import RocCurve
import os
import time
from tqdm import tqdm
import multiprocessing

class LocalPractice():
    def __init__(self, test_set_list, dst_path):
        self.dst_path = dst_path
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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

    def extract_dataloader_feature(self, model_path, batch_size = 1, worker_num = 1):
        # print('Data Loader Extract Feature')
        if len(self.embedding_result) >0:
            self.embedding_result = {}

        self.model_name = model_path.split('/')[-1].strip('.pth')
        net_info, h_value, w_value, patch_info = \
            get_test_param_from_modelname(self.model_name)
        print(net_info, h_value, w_value, patch_info)

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

        elif net_info.find("FaceNetcf20") >=0:
            dim = int(net_info.split('-d')[1])
            self.model = model_map[net_name](dim, h_value, w_value).to(self.device)

        data_loader = get_batch_test_data('{}/{}'.format(self.image_root_path,patch_info),
                                          self.image_list_path, batch_size, worker_num)


        self.load_model(model_path)

        # part_net = self.model[15]

        self.model.eval()
        with torch.no_grad():
            for (img, img_prefix) in tqdm(data_loader):
            # img = iter(data_loader)
                save_path = '/media/hwt/492c0a80-02ce-45ac-9297-63ce30dfdd81/home/minivision/Project/pytorch_models/FaceRecognition/best_select_model/convert_test_convert_caffemodel/ArcFace95to5-b0.4s32_fc_0.4_112x112_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_ResNet18v3fc-d512/'
                np.save(save_path+'img0',img.numpy())
                emb_vec = self.model.forward(img.to(self.device))
                for idex in range(len(emb_vec)):
                    #modified by hwt@2019-03-04: save to ndarray, then the comparasion speed up from 85s to 20s
                    np.save(save_path+'fea0', emb_vec[idex].cpu().numpy())
                break
            # for (img, img_prefix) in tqdm(data_loader):
            #     emb_vec = self.model.forward(img.to(self.device))
            #     for idex in range(len(emb_vec)):
            #         #modified by hwt@2019-03-04: save to ndarray, then the comparasion speed up from 85s to 20s
            #         self.embedding_result[img_prefix[idex]] = emb_vec[idex].cpu().numpy()

    def make_comparation(self):

        # print('Make comparation')
        compare_result = []
        for line in tqdm(self.image_pairs):
            image_name1 = line.split(' ')[0]
            image_name2 = line.split(' ')[1]
            label = int(line.split(' ')[2])

            feature1 = self.embedding_result[image_name1]
            feature2 = self.embedding_result[image_name2]
            diff = feature1 - feature2
            dist = np.sum(np.square(diff), 0, dtype=np.float32)
            compare_result.append([dist, label])
        # print('Save distance result')
        save_result_path = '{}/Result_{}'.format(self.dst_path, self.testset)
        make_if_not_exist(save_result_path)

        trans_name = self.transfrom_model_name(self.model_name)
        # save_dist_path = "{}/result_v2_{}.txt".format(save_result_path, trans_name)
        # f = open(save_dist_path, 'w')
        # for elem in compare_result:
        #     f.write('{:.5f} {}\n'.format(elem[0], elem[1]))
        # f.close()
        npy_save_dist_path = "{}/result_v2_{}".format(save_result_path, trans_name)
        np_compare_result = np.array(compare_result, dtype=np.float32)
        np.save(npy_save_dist_path,np_compare_result)
        draw_roc = RocCurve()
        roc_info, dict_info = draw_roc.make_roc_curve(npy_save_dist_path+'.npy')# so that compatiable with other functions
        return roc_info, dict_info


    def transfrom_model_name(self, pymodel_name):
        prefix = pymodel_name.split('_iter-')[0]
        iter_num = pymodel_name.split('_iter-')[1].split('_')[0]
        trans_model_name = prefix.replace('model', 'pytorch') + '_iter_'+iter_num
        return trans_model_name

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
                        self.extract_dataloader_feature(model_path)
                        roc_info, dict_info = self.make_comparation()
                        f.write("{}\n".format(elem))
                        for line in roc_info:
                            f.write('{}\n'.format(line))
            f.close()


if __name__ == "__main__":

    test_set = ["Business-mtcnn-95to5"]
    # test_date = ["2019-02-15-15-20"]
    test_date = ["2019-03-05-15-00"]
    save_root = '/media/hwt/492c0a80-02ce-45ac-9297-63ce30dfdd81/home/minivision/Project'

    for elem in test_date:
        folder_path = "{}/pytorch_models/FaceRecognition/best_select_model/{}".format(save_root, elem)
        # folder_path = '/media/hwt/492c0a80-02 ce-45ac-9297-63ce30dfdd81/home/minivision/Project/pytorch_models/FaceRecognition/snapshot/SpSVGAngFace/SpSVGAngFace-b0.4s32t0.2Pm5_fc_0.35_112x96_del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8_FaceNetOrg20-d512'
        print(folder_path)
        dst_path = "{}_convert_caffemodel".format(folder_path)
        # local_test = LocalPractice(test_set, dst_path)
        # local_test.test_folder_models(folder_path)
        local_test = LocalPractice(test_set, dst_path)
        local_test.test_folder_models(folder_path)




