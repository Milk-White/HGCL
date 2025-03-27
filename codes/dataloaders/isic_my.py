import cv2
import os
import random
import numpy as np
import warnings

from torch.utils.data import Dataset, DataLoader
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import Sampler
import itertools

class PolypDataset(Dataset):
    
    def __init__(self, root_dir=r'data/isic2018', mode='train', num=None, transforms=None):
    # def __init__(self, cfg, roots, mode='train', trainsize=320, scale=(0.99, 1.01)):
        # 初始化函数，用于设置数据加载器的参数
        self.roots = root_dir  # 数据集根目录
        self.mode = mode  # 模式，可以是'train'（训练）或'val'（验证）

        # self.cfg = cfg
        # self.trainsize = trainsize
        # self.scale = scale
        # self.mode = mode
        self.images = []
        # self.num = 0  # 数据集样本数
        self.gts = []
        self.dataset_lens = []

        # 此处操作，未知
        # 如果transforms是一个列表，则将其转换为一个transforms对象
        if isinstance(transforms, list):
            transforms = build_transforms(transforms)
        self.transforms = transforms  # 数据变换操作

        # 遍历根目录列表
        # for root in self.roots: # 导致数据量10倍的原因
        if mode == 'train':
            # 如果是训练模式，设置数据根目录为 TrainDataset，并获取对应的数据增强操作
            self.data_root = os.path.join(root_dir, 'TrainDataset')
            # self.transform = self.get_augmentation()
        elif mode == 'val':
            # 如果是验证模式，设置数据根目录为 ValidationDataset，并进行简单的 Resize 操作
            self.data_root = os.path.join(root_dir, 'ValidationDataset')
            # self.transform = A.Compose([A.Resize(trainsize, trainsize), ])
        else:
            # 如果模式不是 'train' 或 'val'，抛出异常
            raise KeyError('MODE ERROR')

        # 构建图像和标签的根路径
        image_root = os.path.join(self.data_root, 'images')
        gt_root = os.path.join(self.data_root, 'masks')

        # 获取图像和标签的路径列表，并将其按文件名排序
        # _images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if
        #                 f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
        # _gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if
        #             f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
        _images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                        f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        _gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if
                    f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

        # 将当前数据集的图像和标签路径列表添加到总列表中
        # self.images += _images
        # self.gts += _gts
        # self.dataset_lens.append(len(self.images))
        
        # 记录当前数据集的起始索引和结束索引
        # start_index = len(self.images)
        # end_index = start_index + len(_images) - 1

        # 将当前数据集的图像和标签路径列表添加到总列表中
        self.images += _images
        self.gts += _gts
        # self.dataset_lens.append((start_index, end_index))
            

        # 过滤不符合要求的文件
        # self.filter_files()

        # 设置数据集大小和图像变换操作
        # self.size = len(self.images)
        # print("self.size=",self.size)
        # self.to_tensors = A.Compose([A.Normalize(), ToTensorV2()])


            
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        # 获取图像和标签的文件路径
        image = cv2.imread(self.images[index])  # 读取图像
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像颜色通道转换为RGB
        mask_3 = cv2.imread(self.gts[index])  # 读取标签（掩膜）
        mask = mask_3.sum(axis=2) // 384  # 息肉类别只有1类，所以直接二值化        
        mask = mask.astype(np.uint8)  # 将标签转化为8位无符号整数类型
        # 将图像和标签组成一个字典
        sample = {'image': image, 'label': mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def get_isic_loaders(root_dir=r'data/isic2018', labeled_num=7, labeled_bs=12, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None):

    # 构建 PolypDataset 对象，用于加载训练数据
    db_train = PolypDataset(root_dir=root_dir, mode="train", transforms=train_transforms)
    db_val = PolypDataset(root_dir=root_dir, mode="val", transforms=val_transforms)

    total_slices = len(db_train)  # 获取训练数据集总样本数
    
    # args.DATA.LABEL=0.1
    labeled_slice = int(total_slices * 0.1)  # 计算标注样本的切片数量
    # print("labeled_slice=",labeled_slice)

    idxs = list(range(total_slices))  # 构建索引列表
    # fold_len = int(cfg.DATA.LABEL * total_slices)  UGPCL中没有fold参数
    # labeled_idxs = idxs[fold_len * cfg.TRAIN.FOLD: fold_len * (cfg.TRAIN.FOLD + 1)]  # 获取标注样本的索引
    # unlabeled_idxs = list(set(idxs) - set(labeled_idxs))  # 获取非标注样本的索引
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))

    assert len(labeled_idxs) == labeled_slice  # 断言确保标注样本的切片数量正确

    # 使用TwoStreamBatchSampler分配标注和非标注样本到不同的批次
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 
                                           batch_size, batch_size - labeled_bs)
    # 创建数据加载器，使用分配的批次
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, 	  
                               num_workers=num_workers,
                               pin_memory=True, worker_init_fn=worker_init_fn)
                            #    pin_memory=False, worker_init_fn=worker_init_fn)
    
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    print("Data loaders created successfully.")
    print(f"Training loader length: {len(train_loader)}")
    print(f"Validation loader length: {len(val_loader)}")
    

    # 返回训练和验证数据加载器，直接不区分mode
    return train_loader, val_loader
