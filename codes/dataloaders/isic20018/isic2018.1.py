import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler
import torch
import numpy as np
from  torchvision import transforms

# 频域变换相关函数
def extract_ampl_phase(fft_im):
    # fft_im: size should be b x 3 x h x w
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    w *= 2  # 因为我们只使用了一半空间（rFFT）
    b = (np.floor(0.5 * np.amin((h, w)) * L)).astype(int)
    if b > 0:
        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]
        amp_src[:, :, h-b+1:h, 0:b] = amp_trg[:, :, h-b+1:h, 0:b]
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[imgH, imgW])
    return src_in_trg

# 数据集类
class ISICDataset(Dataset):
    def __init__(self, root_dir='/opt/data/private/wd/Dataset/Kvasir/TrainDataset', channels=3, mode='train', transforms=None, use_fda=False, fda_target_image=None, fda_L=0.1):
        super().__init__()
        self.root_dir = root_dir
        self.channels = channels
        self.use_fda = use_fda  # 新增的标志位，是否使用FDA
        self.fda_target_image = fda_target_image  # FDA目标图像
        self.fda_L = fda_L  # FDA的低频交换比例

        if isinstance(transforms, list):
            transforms = build_transforms(transforms)
        self.transforms = transforms

        with open(f'{self.root_dir}/train.list', 'r') as f:
            name_list = f.readlines()
        name_list = [item.replace('\n', '') for item in name_list]
        self.name_list = name_list[:1150] if mode == 'train' else name_list[1150:]

        self.imgs_path = f'{self.root_dir}/images'
        self.labels_path = f'{self.root_dir}/masks'

    def __getitem__(self, index):
        file_name = self.name_list[index]
        img_path = os.path.join(self.imgs_path, f'{file_name}.png')
        label_path = os.path.join(self.labels_path, f'{file_name}.png')

        if self.channels == 1:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

       # 如果使用 FDA 并且有目标图像
        if self.use_fda and self.fda_target_image is not None:
            img = transforms.ToTensor()(img).unsqueeze(0)  # 将图像转换为 Tensor 并增加 batch 维度
            fda_target_img = transforms.ToTensor()(self.fda_target_image).unsqueeze(0)  # 目标图像
            img = FDA_source_to_target(img, fda_target_img, L=self.fda_L)
            img = img.squeeze(0)  # 去掉 batch 维度

             # 将生成的 Tensor 转换回 PIL Image
            img = transforms.ToPILImage()(img)

            # 确保 label 是 PIL Image，然后再转换为 Tensor
        if isinstance(label, Image.Image):
            label = transforms.ToTensor()(label)

        sample = {'image': img, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)  # 确保 sample['image'] 是 PIL Image

        return sample

        

    def __len__(self):
        return len(self.name_list)

# DataLoader 函数
def get_isic_loaders(root_dir=r'/opt/data/private/wd/Dataset/Kvasir/TrainDataset', channels=3, labeled_ratio=0.25, labeled_bs=12, batch_size=24, batch_size_val=16,
                     num_workers=4, train_transforms=None, val_transforms=None, worker_init_fn=None, use_fda=False, fda_target_image=None, fda_L=0.1):
    train_dataset = ISICDataset(root_dir=root_dir, channels=channels, mode='train', transforms=train_transforms, use_fda=use_fda, fda_target_image=fda_target_image, fda_L=fda_L)
    val_dataset = ISICDataset(root_dir=root_dir, channels=channels, mode='val', transforms=val_transforms)

    if labeled_bs < batch_size and labeled_ratio < 1.0:
        label_num = int(len(train_dataset) * labeled_ratio)
        labeled_idxs = list(range(label_num))
        unlabeled_idxs = list(range(label_num, len(train_dataset)))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    return train_loader, val_loader
