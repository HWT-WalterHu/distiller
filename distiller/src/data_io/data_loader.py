'''
 Data load from image list
'''

from .read_image_list_io import DatasetFromList , opencv_loader, read_image_list_test
from . import transforms as trans
from torch.utils.data import DataLoader
from ..model_config.data_set import TrainSet, TestSet

def get_train_dataset(img_root_path, image_list_path):
    train_transform = trans.Compose([
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        # trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = DatasetFromList(img_root_path, image_list_path, opencv_loader, train_transform, None)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_list_loader(conf):
        train_set = conf.data_mode
        img_root_path = TrainSet[train_set]['root_path']
        img_label_list = TrainSet[train_set]['label_list']

        print("img_label_list")
        print(img_label_list)

        patch_info = conf.patch_info
        root_path = '{}/{}'.format(img_root_path, patch_info)
        celeb_ds, celeb_class_num = get_train_dataset(root_path, img_label_list)
        ds = celeb_ds
        class_num = celeb_class_num
        loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
        return loader, class_num


def get_test_dataset(img_root_path, image_list_path):
    train_transform = trans.Compose([
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        # trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = DatasetFromList(img_root_path, image_list_path, opencv_loader, train_transform, None, read_image_list_test)
    return ds

def get_batch_test_data(img_root_path, image_list_path, batch_size,num_workers):
    ds = get_test_dataset(img_root_path, image_list_path)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers)
    return loader

