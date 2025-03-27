import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

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
        # 只替换非黑色背景（假设黑色背景的幅度接近于0）
        mask_src = amp_src[:, :, 0:b, 0:b] > 1e-6
        amp_src[:, :, 0:b, 0:b] = torch.where(mask_src, amp_trg[:, :, 0:b, 0:b], amp_src[:, :, 0:b, 0:b])
        
        mask_src = amp_src[:, :, h-b+1:h, 0:b] > 1e-6
        amp_src[:, :, h-b+1:h, 0:b] = torch.where(mask_src, amp_trg[:, :, h-b+1:h, 0:b], amp_src[:, :, h-b+1:h, 0:b])
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

# 使用 FDA_source_to_target 函数生成目标风格图像并保存
def display_and_save_fft_transform(src_img_path, trg_img_path, save_dir='output_images', L=0.1):
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载并预处理源图像和目标图像
    src_img = Image.open(src_img_path).convert('RGB')
    trg_img = Image.open(trg_img_path).convert('RGB')
    transform = transforms.ToTensor()
    
    src_img_tensor = transform(src_img).unsqueeze(0)  # 增加 batch 维度
    trg_img_tensor = transform(trg_img).unsqueeze(0)

    # 生成目标风格图像
    transformed_img_tensor = FDA_source_to_target(src_img_tensor, trg_img_tensor, L=L)
    transformed_img = transforms.ToPILImage()(transformed_img_tensor.squeeze(0))  # 去掉 batch 维度并转换为 PIL 图像

    # 保存原始图像、目标图像和傅里叶变换后的图像
    src_img.save(os.path.join(save_dir, 'source_image.png'))          # 保存源图像
    trg_img.save(os.path.join(save_dir, 'target_image.png'))          # 保存目标图像
    transformed_img.save(os.path.join(save_dir, 'transformed_image.png'))  # 保存傅里叶变换后的图像

    # 可视化原始图像、目标图像和傅里叶变换后的图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(src_img)
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(trg_img)
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    axes[2].imshow(transformed_img)
    axes[2].set_title("Transformed Image (FDA)")
    axes[2].axis('off')

    plt.show()

# 示例使用
src_img_path = '/opt/data/private/wd/Dataset/Kvasir/TrainDataset/images/34.png'  # 源图像路径
trg_img_path = '/opt/data/private/wd/Dataset/Kvasir/TrainDataset/images/365.png'   # 目标图像路径
save_directory = '/opt/data/private/wd/UGPCL-222/codes/dataloaders/isic20018/FULIYE/1'  # 自定义保存目录

# 调用函数并指定保存目录
display_and_save_fft_transform(src_img_path, trg_img_path, save_dir=save_directory, L=0.1)
