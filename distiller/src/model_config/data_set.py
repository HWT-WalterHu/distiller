'''
save Training dataset and TestSet

'''



#--------------------Training  Config ---------------------------
list_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn_95to5/MultiPatches_list"
image_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn_95to5/MultiPatches"

TrainSet = {

"PAKJ" : {"root_path": image_root_path,
        "label_list": "{}/single_folder_list/GE_15/PAKJ_09_09_GE15_list_label.txt".format(list_root_path),
            },
"Combine_24_GE15" : {"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/combine_all-24_dataset_GE15_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-GE15-add-KdGE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_GE15_KgartenG8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-GE15-add-Kd-TYLG-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_GE15_Kgarten-TYLG_G8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-GE15-add-Kd1-2-TYLG-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_GE15_Kgarten1-2_TYLG_GE8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-GE15-add-Kd1-2-TYLG-NL-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_GE15_Kgarten1-2_TYLG_NL_GE8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_Kgarten1-2_TYLG_NL_GE8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-Kd1-2-TYLG-NL-AChild-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_Kgarten1-2_TYLG_NL_AChild_GE8_list_label.txt".format(list_root_path),
             },
"del-LAN-DHUA-Crop-GE15-add-Kd-TYLG-AChild-GE8":{"root_path": image_root_path,
             "label_list": "{}/combine_folder_list/Combine_95to5_Age_shot_Orient_PAKJ_Indon_Mig_celeb_CHina_cap_mul_beid_Pad_XCH_Asian_ZheDa_GE15_Kgarten-TYLG-AChild_G8_list_label.txt".format(list_root_path),
                                                }
}



#--------------------Training Test Config ------------------------
test_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Test_Data/O2N"
TestSet = {
# "XCH-mtcnn-outdoor3-95to5" : {
#     "root_path":"{}/XCH_PAD_08-01_outdoor/mtcnn_patch_95to5/".format(test_root_path),
#     "image_list":"{}/XCH_PAD_08-01_outdoor/10-24-landmark-95to5/XCH_PAD_08-01_outdoor_list.txt".format(test_root_path),
#     "image_pairs":"{}/XCH_PAD_08-01_outdoor/10-24-landmark-95to5/XCH_PAD_08-01_outdoor_listpair.txt".format(test_root_path)
#     },

"XCH-mtcnn-outdoor3-95to5-d" : {
    "root_path":"{}/XCH_PAD_08-01_outdoor/mtcnn_patch_95to5/".format(test_root_path),
    "image_list":"{}/XCH_PAD_08-01_outdoor/10-24-landmark-95to5-del/XCH_PAD_08-01_outdoor_list.txt".format(test_root_path),
    "image_pairs":"{}/XCH_PAD_08-01_outdoor/10-24-landmark-95to5-del/XCH_PAD_08-01_outdoor_listpair.txt".format(test_root_path)
    },

"Business-mtcnn-95to5" : {
    "root_path":"{}/Business_09-28_mtcnn/mtcnn_patch_95to5/".format(test_root_path),
    "image_list":"{}/Business_09-28_mtcnn/landmark_95to5-2/image_list.txt".format(test_root_path),
    "image_pairs":"{}/Business_09-28_mtcnn/landmark_95to5-2/image_listpair.txt".format(test_root_path)
    },

"XCH_out2_95to5": {
    "root_path":"{}/XCH-small-outdoor2/mtcnn_patch_95to5/".format(test_root_path),
    "image_list":"{}/XCH-small-outdoor2/mtcnn_95to5-2_landmark/image_list.txt".format(test_root_path),
    "image_pairs":"{}/XCH-small-outdoor2/mtcnn_95to5-2_landmark/image_listpair.txt".format(test_root_path)
    },

"Business_small_95to5":{
    "root_path": "{}/Business_09-28_mtcnn_small/mtcnn_patch_95to5".format(test_root_path),
    "image_list": "{}/Business_09-28_mtcnn_small/image_list.txt".format(test_root_path),
    "image_pairs": "{}/Business_09-28_mtcnn_small/image_listpair.txt".format(test_root_path)
    },

"Child-95to5":{
    "root_path": "{}/Kingdergarten_Child/patch_mtcnn-95to5".format(test_root_path),
    "image_list": "{}/Kingdergarten_Child/id_life_mtcnn_result/id_life_image_list.txt".format(test_root_path),
    "image_pairs": "{}/Kingdergarten_Child/id_life_mtcnn_result/id_life_image_listpair.txt".format(test_root_path)
    },

"XCH-mtcnn-09-01-29":{
    "root_path": "{}/XCH_Pad_19_01_29_hard/mtcnn_patch_95to5".format(test_root_path),
    "image_list": "{}/XCH_Pad_19_01_29_hard/id_life_mtcnn_result/id_life_image_list.txt".format(test_root_path),
    "image_pairs": "{}/XCH_Pad_19_01_29_hard/id_life_mtcnn_result/id_life_image_listpair.txt".format(test_root_path)
    },
}


