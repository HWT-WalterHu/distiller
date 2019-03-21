#Using multiprocess to make feature's comparation

from src.roc_evaluate.local_practice import LocalTest
from src.roc_evaluate.RocCurve import RocCurve
from src.utils import get_test_param_from_modelname,make_if_not_exist, get_time
import numpy as np
import os
import time
# from multiprocessing import Process
# from collections import deque


# class ProcessCompare(Process):
#     def __init__(self, fea_dict, list_queue, compare_result):
#         self.fea_dict = fea_dict
#         self.list_queue = list_queue
#         self.compare_result = compare_result
#
#     def run(self):
#         while not self.list_queue.empty():
#
#
#
#
#
# class LocalTestProcess(LocalTest):
#
#     def __init__(self, test_set_list, dst_path):
#         super(LocalTestProcess, self).__init__(test_set_list, dst_path)
#
#     def make_comparation(self):



class LocalTestMat(LocalTest):

    def __init__(self, test_set_list, dst_path):
        super(LocalTestMat, self).__init__(test_set_list, dst_path)

    def make_comparation(self):

        # print('Make comparation')
        compare_result = []
        fea1 = []
        fea2 = []
        label_list = []

        start = time.time()
        for line in self.image_pairs:
            image_name1 = line.split(' ')[0]
            image_name2 = line.split(' ')[1]
            label = int(line.split(' ')[2])

            feature1 = self.embedding_result[image_name1].cpu().numpy()
            feature2 = self.embedding_result[image_name2].cpu().numpy()

            fea1.append(feature1)
            fea2.append(feature2)
            label_list.append(label)

        print("Collect feature: ", time.time() - start)

        start2 = time.time()
        fea_dist = np.array(fea1) - np.array(fea2)
        square_fea_dist = fea_dist * fea_dist
        fea_result = np.sum(square_fea_dist, axis=1)
        print("Sum dist: ", time.time() - start2)


        # print('Save distance result')
        save_result_path = '{}/Result_{}'.format(self.dst_path, self.testset)
        make_if_not_exist(save_result_path)

        trans_name = self.transfrom_model_name(self.model_name)
        save_dist_path = "{}/result_v2_{}.txt".format(save_result_path, trans_name)
        f = open(save_dist_path, 'w')
        for i in range(len(label_list)):
            f.write('{:.5f} {}\n'.format(fea_result[i], label_list[i]))
        # for elem in compare_result:
        #     f.write('{:.5f} {}\n'.format(elem[0], elem[1]))
        f.close()
        draw_roc = RocCurve()
        roc_info, dict_info = draw_roc.make_roc_curve(save_dist_path)
        return roc_info, dict_info



if __name__ == "__main__":

    # test_set = ["XCH-mtcnn-outdoor3-95to5-d", "Business-mtcnn-95to5", "Child-95to5"]
    # test_set = ["Business_small_95to5"]
    # test_set = ["XCH-mtcnn-outdoor3-95to5-d"]
    test_set = ["Child-95to5"]

    test_date = ["2019-03-04-10-12"]


    for elem in test_date:
        folder_path = "{}/Project/pytorch_models/FaceRecognition/best_select_model/{}".format(os.path.expandvars('$HOME'), elem)
        print(folder_path)
        dst_path = "{}_convert_caffemodel".format(folder_path)

        for i in range(3):
            local_test = LocalTestMat(test_set, dst_path)
            local_test.test_folder_models(folder_path)







