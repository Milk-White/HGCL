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
    """
    Dataloader for polyp segmentation tasks
    """

    def __init__(self, root_dir=r'/opt/data/private/wd/Dataset/Kvasir', mode='train', num=None, transforms=None):
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
        _images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if
                        f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
        _gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if
                    f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])

        # 将当前数据集的图像和标签路径列表添加到总列表中
        # self.images += _images
        # self.gts += _gts
        # self.dataset_lens.append(len(self.images))
        
        # 记录当前数据集的起始索引和结束索引
        start_index = len(self.images)
        end_index = start_index + len(_images) - 1

        # 将当前数据集的图像和标签路径列表添加到总列表中
        self.images += _images
        self.gts += _gts
        self.dataset_lens.append((start_index, end_index))
            

        # 过滤不符合要求的文件
        # self.filter_files()

        # 设置数据集大小和图像变换操作
        self.size = len(self.images)
        # print("self.size=",self.size)
        self.to_tensors = A.Compose([A.Normalize(), ToTensorV2()])

            
    def __len__(self):
        return self.size


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


def get_colon_loaders(root_dir=r'/opt/data/private/wd/Dataset/Kvasir', channels=3,labeled_num=7,labeled_ratio=0.25, labeled_bs=12, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None):
# def get_colon_loaders(mode, cfg, trainsize=320, scale=(0.75, 1)):
    # 存储数据集根目录的列表
    # data_root = []  直接root_dir路径

    # 根据配置中的数据集名称构建数据集路径
    # if "Kvasir" in cfg.DATA.NAME:
    #     data_root.append(os.path.join(cfg.DIRS.DATA, 'Kvasir-SEG'))
    # if "ISIC" in cfg.DATA.NAME:
    #     data_root.append(os.path.join(cfg.DIRS.DATA, 'ISIC'))

    # if mode == 'train':
    # 训练模式下，构建 ImageDataset 对象，用于加载训练数据
    # dts = ImageDataset(cfg=cfg, roots=data_root, mode='train', trainsize=trainsize, scale=scale)

    # 构建 PolypDataset 对象，用于加载训练数据
    db_train = PolypDataset(root_dir=root_dir, mode="train", transforms=train_transforms)

    print("Creating training and validation loaders...")

    db_val = PolypDataset(root_dir=root_dir, mode="val", transforms=val_transforms)


    # batch_size = cfg.TRAIN.BATCH_SIZE bs参数里包含了

    # total_slices = len(dts)  # 获取数据集总样本数
    total_slices = len(db_train)  # 获取训练数据集总样本数
    # print("total_slices=",total_slices)
    
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
    

    # # 验证模式下，构建 ImageDataset 对象，用于加载验证数据
    # dts = ImageDataset(cfg=cfg, roots=data_root, mode='val', trainsize=trainsize, scale=scale)
    # batch_size = cfg.VAL.BATCH_SIZE
    # # 创建 DataLoader 对象，直接使用指定的批量大小
    # dataloader = DataLoader(dts, batch_size=batch_size,
    #                         shuffle=False, drop_last=False,
    #                         num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    # elif mode == 'test':
    # 测试模式下，构建 ImageDataset 对象，用于加载测试数据
    # dts = ImageDataset(cfg=cfg, roots=data_root, mode='test', trainsize=trainsize, scale=scale)
    # batch_size = cfg.TEST.BATCH_SIZE
    # # 创建 DataLoader 对象，直接使用指定的批量大小
    # dataloader = DataLoader(dts, batch_size=batch_size,
    #                         shuffle=False, drop_last=False,
    #                         num_workers=cfg.SYSTEM.NUM_WORKERS)

    # else:
    #     # 如果 mode 不是 'train', 'valid', 'test' 中的任何一个，抛出 KeyError
    #     raise KeyError(f"mode error: {mode}")

    # return dataloader  # 返回构建的数据加载器

    # 返回训练和验证数据加载器，直接不区分mode
    return train_loader, val_loader
