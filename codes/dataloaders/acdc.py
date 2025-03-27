import h5py

from torch.utils.data import Dataset, DataLoader
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler


class ACDCDataSet(Dataset):

    def __init__(self, root_dir=r'data/ACDC', mode='train', num=None, transforms=None):
        # 初始化函数，用于设置数据加载器的参数
        self.root_dir = root_dir  # 数据集根目录
        self.mode = mode  # 模式，可以是'train'（训练）或'val'（验证）
        
        # 如果transforms是一个列表，则将其转换为一个transforms对象
        if isinstance(transforms, list):
            transforms = build_transforms(transforms)
        self.transforms = transforms  # 数据变换操作
        
        # 根据模式选择相应的文件列表（训练或验证）
        names_file = f'{self.root_dir}/train_slices.list' if self.mode == 'train' else f'{self.root_dir}/val_slices.list'
        
        # 从文件中读取样本列表
        with open(names_file, 'r') as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        # 如果指定了num且模式为训练，则仅保留列表的前num个样本
        if num is not None and self.mode == 'train':
            self.sample_list = self.sample_list[:num]

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 根据索引加载数据集中的一个样本
        case = self.sample_list[idx]
        
        # 使用h5py库打开对应样本的HDF5文件
        h5f = h5py.File(f'{self.root_dir}/data/slices/{case}.h5', 'r')
        
        # 从HDF5文件中读取图像和标签数据
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        # 将图像和标签组成一个字典
        sample = {'image': image, 'label': label}
        
        # 如果定义了数据变换操作，则应用这些变换
        if self.transforms:
            sample = self.transforms(sample)
        
        # 返回最终的样本
        return sample



def get_acdc_loaders(root_dir=r'data/ACDC', labeled_num=7, labeled_bs=12, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None):
    # 定义一个参考字典，映射标注样本数量到对应的标注切片数量
    ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}

    # 创建训练和验证数据集对象
    db_train = ACDCDataSet(root_dir=root_dir, mode="train", transforms=train_transforms)
    db_val = ACDCDataSet(root_dir=root_dir, mode="val", transforms=val_transforms)

    # 根据标注样本数量和标注批量大小确定训练数据集的切片分配
    if labeled_bs < batch_size:
        labeled_slice = ref_dict[str(labeled_num)]
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, len(db_train)))
        # 使用TwoStreamBatchSampler分配标注和非标注样本到不同的批次
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        # 创建数据加载器，使用分配的批次
        train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        # 创建数据加载器，直接使用指定的批量大小
        train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    
    # 创建验证数据加载器
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    # 返回训练和验证数据加载器
    return train_loader, val_loader

