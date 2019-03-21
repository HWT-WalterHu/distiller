'''
ROC Evaluate Model
2019-03-01 new features by zkx@__@
1. conf.test_set changes into dictionary with select tolerance
2. save run time test result and share same time stamp with snapshot
3. test device id in conf.device_ids[0]
'''
import os
import torch
import numpy as np
from distiller.src.data_io.data_loader import get_batch_test_data
from distiller.src.roc_evaluate.RocCurve import RocCurve
from distiller.src.models.model import MobileFaceNet
from distiller.src.model_config.data_set import TestSet
from distiller.src.utils import get_time

class Run_Time_Evaluate:

    def __init__(self, conf):

        self.conf = conf
        device_id = conf.device_ids[0]
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.embedding_result = {}
        self.roc_eval = RocCurve()
        self.dst_path = conf.test_roc_path
        self.val_loader = {}
        self.image_pairs = {}
        for elem in self.conf.test_set:
            image_pair_path = TestSet[elem]['image_pairs']
            f = open(image_pair_path, 'r')
            data = f.read().splitlines()
            f.close()
            self.image_pairs[elem] = data
        for elem in self.conf.test_set:
            image_root_path = "{}/{}".format(TestSet[elem]['root_path'], self.conf.patch_info)
            image_list_path = TestSet[elem]['image_list']
            self.val_loader[elem] = get_batch_test_data(image_root_path, image_list_path,
                                          self.conf.test_batch_size, self.conf.test_worker_num)

    def extract_dataloader_feature(self,model):
        # print('Data Loader Extract Feature')

        for elem in self.conf.test_set:
            self.embedding_result[elem] = {}
            data_loader = self.val_loader[elem]
            model.eval()
            with torch.no_grad():
                # for (img, img_prefix) in tqdm(data_loader):
                for (img, img_prefix) in data_loader:
                    emb_vec = model.forward(img.to(self.device))
                    for idex in range(len(emb_vec)):
                        self.embedding_result[elem][img_prefix[idex]] = emb_vec[idex].cpu().numpy()

    def make_comparation(self, iter_num, time_stamp):

        # print('Make comparation')
        compare_result = {}
        accuracy = {}
        for elem, tolerance in self.conf.test_set.items():

            compare_result[elem] = []
            for line in self.image_pairs[elem]:
                image_name1 = line.split(' ')[0]
                image_name2 = line.split(' ')[1]
                label = int(line.split(' ')[2])

                feature1 = self.embedding_result[elem][image_name1]
                feature2 = self.embedding_result[elem][image_name2]
                diff = feature1 - feature2

                dist = np.sum(np.square(diff), 0 , dtype=np.float32)
                compare_result[elem].append([dist, label])

            dist_label = np.array(compare_result[elem])
            save_dst_path = '{}/Result_{}'.format(self.dst_path, elem)
            if not os.path.exists(save_dst_path):
                os.makedirs(save_dst_path)

            save_prefix = "result_v2_{}_{}_pytorch_iter_{}".format(time_stamp, self.conf.job_name, iter_num)
            save_roc_figure_path = "{}/{}_ROC.png".format(save_dst_path, save_prefix)
            save_result_path = "{}/{}".format(save_dst_path, save_prefix)
            # save_result_path = "{}/{}.txt".format(save_dst_path, save_prefix)

            title =  '{}_{}_{}'.format(self.conf.net_mode,iter_num , elem)
            roc_info, info_dict = self.roc_eval.draw_roc(-dist_label[:,0], dist_label[:,1], save_roc_figure_path, title)

            #save result path
            np.save(save_result_path, dist_label)
            accuracy["{}-{}".format(elem, tolerance)] = info_dict[tolerance]

        return accuracy

    def evaluate_model(self, test_model, iter_num, time_stamp):
        self.extract_dataloader_feature(test_model)
        accuracy = self.make_comparation(iter_num, time_stamp)
        return accuracy






