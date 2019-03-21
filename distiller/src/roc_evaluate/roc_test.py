'''
ROC Evaluate Model
Example for single model test
'''
import os
import torch
import cv2
from src.models.model import MobileFaceNet, l2_norm
import numpy as np
from tqdm import tqdm
from src.data_io.data_loader import get_batch_test_data
from src.roc_evaluate.RocCurve import RocCurve


class ROC_Evaluate:

    def __init__(self, mdoel_path, image_root_path,
                 patch_info, image_list, image_pair_list, dst_path, embedding_size = 512):
        '''
        :param patch_info: fc_0.4_112x112
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MobileFaceNet(embedding_size).to(self.device )
        self.load_model(model_path)
        self.embedding_size = embedding_size

        model_name = os.path.basename(mdoel_path).replace('.pth', '.txt')
        self.dst_path = "{}/result_{}".format(dst_path, model_name)
        self.image_list_path = image_list
        self.image_pair_path = image_pair_list

        height = int(patch_info.split('_')[-1].split('x')[0])
        width = int(patch_info.split('_')[-1].split('x')[1])
        self.input_size = (height, width)

        self.image_root_path = image_root_path + '/' + patch_info
        self.embedding_result = {}

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

    def cvimg_to_tensor(self, img):
        assert(img.shape[0] == self.input_size[0])
        assert (img.shape[1] == self.input_size[1])
        # to Tensor range(0-1)
        img = img.astype(np.float32)/255
        tensor_img = torch.from_numpy(img.transpose((2,0,1)))
        tensor_img = torch.stack([tensor_img], 0)
        return tensor_img

    def extract_image_feature(self):

        f = open(self.image_list_path, 'r')
        image_list = f.read().splitlines()
        f.close()

        print('Extract Feature')
        self.model.eval()
        with torch.no_grad():

            for elem_img in tqdm(image_list):
                image_path = '{}/{}'.format(self.image_root_path, elem_img)
                img = cv2.imread(image_path)
                tsr_img = self.cvimg_to_tensor(img)
                # emb_vec = self.model(tsr_img.to(self.device))
                # emb_vec = l2_norm(emb_vec)
                emb_vec = self.model.forward(tsr_img.to(self.device))
                self.embedding_result[elem_img] = emb_vec

    def extract_dataloader_feature(self, batch_size, worker_num):
        print('Data Loader Extract Feature')
        data_loader = get_batch_test_data(self.image_root_path, self.image_list_path, batch_size, worker_num)
        self.model.eval()
        with torch.no_grad():

            for (img, img_prefix) in tqdm(data_loader):
                # emb_vec = self.model(img.to(self.device))
                # emb_vec = l2_norm(emb_vec)
                emb_vec = self.model.forward(img.to(self.device))
                for idex in range(len(emb_vec)):
                    self.embedding_result[img_prefix[idex]] = emb_vec[idex]

    def make_comparation(self):

        f = open(self.image_pair_path, 'r')
        image_pairs = f.read().splitlines()
        f.close()

        print('Make comparation')
        compare_result = []
        for line in tqdm(image_pairs):
            image_name1 = line.split(' ')[0]
            image_name2 = line.split(' ')[1]
            label = int(line.split(' ')[2])

            # feature1 = self.embedding_result[image_name1]
            # feature2 = self.embedding_result[image_name2]
            # diff = np.subtract(feature1, feature2).numpy()

            #To speed up by transform into numpy form
            feature1 = self.embedding_result[image_name1].cpu().numpy()
            feature2 = self.embedding_result[image_name2].cpu().numpy()
            diff = feature1 - feature2

            dist = np.sum(np.square(diff), 0 , dtype=np.float32)
            compare_result.append([dist, label])

        print('Save distance result')
        f = open(self.dst_path, 'w')
        for elem in compare_result:
            f.write('{} {}\n'.format(elem[0], elem[1]))
        f.close()
        draw_roc = RocCurve()
        dict_info = draw_roc.make_roc_curve(self.dst_path)
        return dict_info


    def ModelTest(self, batch_size, worker_num):
        self.extract_dataloader_feature(batch_size, worker_num)
        dict_info = self.make_comparation()
        return dict_info



# /home/zkx-97/PycharmProjects/InsightFace_Pytorch/work_space/model_mobilefacenet.pth
if __name__ == "__main__":
    # model_path = "/home/zkx-97/PycharmProjects/InsightFace_Pytorch/work_space/model_mobilefacenet.pth"
    model_path = "/home/zkx-97/Project/pytorch_models/FaceRecognition/snapshot/ArcFace95to5/ArcFace95to5-b0.3s64_fc_0.4_112x112_del-LAN-DHUA-Crop-GE15-add-KdGE8_MobileFaceNet-d100/2018-12-01-16-00_ArcFace95to5-b0.3s64_fc_0.4_112x112_del-LAN-DHUA-Crop-GE15-add-KdGE8_MobileFaceNet-d100_model_iter-268205_Bus-0.7389_XCH-0.9136.pth"
    image_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Test_Data/O2N/XCH_PAD_08-01_outdoor/mtcnn_patch_95to5"
    patch_info = "fc_0.4_112x112"
    image_list = "/mnt/glusterfs/o2n/FaceRecognition/Test_Data/O2N/XCH_PAD_08-01_outdoor/10-24-landmark-95to5/XCH_PAD_08-01_outdoor_list.txt"
    image_pair_list = "/mnt/glusterfs/o2n/FaceRecognition/Test_Data/O2N/XCH_PAD_08-01_outdoor/10-24-landmark-95to5/XCH_PAD_08-01_outdoor_listpair.txt"
    dst_path = "/home/zkx-97/PycharmProjects/Pytorch_FaceRecognition_Training/predict_result"
    roc = ROC_Evaluate(model_path,image_root_path, patch_info, image_list, image_pair_list, dst_path, 100)
    roc.ModelTest(batch_size=50, worker_num=1)
