import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler


class ISICDataset(Dataset):
    #初始化
    def __init__(self,
                 root_dir='/opt/data/private/wd/Dataset/Kvasir/TrainDataset',
                 channels=3,
                 mode='train',
                 transforms=None):  #用于数据增强和预处理的操作
        super().__init__()

        self.root_dir = root_dir
        self.channels = channels

        if isinstance(transforms, list):
            transforms = build_transforms(transforms)
        self.transforms = transforms

        with open(f'{self.root_dir}/train.list', 'r') as f:
            name_list = f.readlines()
        name_list = [item.replace('\n', '') for item in name_list]

        # self.name_list = name_list[:0] if mode == 'train' else name_list[0:]
        # self.name_list = name_list[:1150] if mode == 'train' else name_list[1150:]  # 息肉数据集\
        if mode == 'train':
            self.name_list = []  # 训练时取 0 张图
        else:
            self.name_list = name_list[:]  # 验证时取所有图片

        # self.name_list = name_list[:1815] if mode == 'train' else name_list[1815:]  # isic2018

        self.imgs_path = f'{self.root_dir}/images'
        self.labels_path = f'{self.root_dir}/masks'

    def __getitem__(self, index):
        file_name = self.name_list[index]
        
        # isic2018
        # img_path = os.path.join(self.imgs_path, f'{file_name}.jpg')
        # label_path = os.path.join(self.labels_path, f'{file_name}_segmentation.png')

        # 息肉数据集
        img_path = os.path.join(self.imgs_path, f'{file_name}.png')
        label_path = os.path.join(self.labels_path, f'{file_name}.png')

        if self.channels == 1:
            img = Image.open(img_path).convert('L')  # signal channel
        else:
            img = Image.open(img_path).convert('RGB')  # 3 channels
        label = Image.open(label_path).convert('L')
        # label = Image.open(label_path)
        sample = {'image': img, 'label': label}  #将图像和标注封装为字典 sample，然后应用预处理（如果有传入的 transforms），最后返回处理后的样本。
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.name_list)


def get_isic_loaders(root_dir=r'/opt/data/private/wd/Dataset/Kvasir/TrainDataset', channels=3, labeled_ratio=0.25, labeled_bs=12,
                     batch_size=24, batch_size_val=16,
                     num_workers=4, train_transforms=None, val_transforms=None, worker_init_fn=None):
    train_dataset = ISICDataset(root_dir=root_dir, channels=channels, mode='train', transforms=train_transforms)
    val_dataset = ISICDataset(root_dir=root_dir, channels=channels, mode='val', transforms=val_transforms)

    if labeled_bs < batch_size and labeled_ratio < 1.0:  #判断是否需要划分有标注与无标注数据，检查 labeled_bs 是否小于 batch_size，以及 labeled_ratio 是否小于 1.0。如果这两个条件满足，则说明需要区分有标注和无标注的数据
        label_num = int(len(train_dataset) * labeled_ratio)  #有标注数据的样本数量
        labeled_idxs = list(range(label_num))       #标注数据的索引列表
        unlabeled_idxs = list(range(label_num, len(train_dataset))) #无标注数据的索引列表

        #TwoStreamBatchSampler根据有标注和无标注的数据索引列表创建 batch_sampler，batch_size - labeled_bs：控制每个 batch 中无标注数据的数量，批次内会混合有标注和无标注数据
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        #使用 TwoStreamBatchSampler 采样有标注和无标注数据，构造 DataLoader
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    #如果没有划分标注数据：直接使用 DataLoader，按 batch_size 加载训练数据。
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    #
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn)
    return train_loader, val_loader
