import os  
from PIL import Image  
import numpy as np  

def calculate_dice_iou(pred, gt, smooth=1e-6):  
    """  
    计算单个图像的Dice系数和IoU。  
    :param pred: 预测图像的numpy数组，形状为(H, W)，值应为0或1  
    :param gt: GT图像的numpy数组，形状为(H, W)，值应为0或1  
    :param smooth: 平滑项，防止除以零  
    :return: Dice系数和IoU  
    """  
    intersection = np.logical_and(pred, gt)  
    dice = (2. * intersection.sum() + smooth) / (pred.sum() + gt.sum() + smooth)  
    iou = (intersection.sum() + smooth) / (np.logical_or(pred, gt).sum() + smooth)  
    return dice, iou  

def evaluate_folders(pred_folder, gt_folder):  
    mdice_sum = 0  
    miou_sum = 0  
    count = 0  

    # 遍历文件夹中的图片  
    for pred_file, gt_file in zip(sorted(os.listdir(pred_folder)), sorted(os.listdir(gt_folder))):  
        pred_img = Image.open(os.path.join(pred_folder, pred_file))  
        gt_img = Image.open(os.path.join(gt_folder, gt_file))  

        # 将预测图像和GT图像转换为灰度图像
        if pred_img.mode != 'L':  
            pred_img = pred_img.convert('L')  
        if gt_img.mode != 'L':  
            gt_img = gt_img.convert('L')  

        # 加载图片并转换为numpy数组  
        pred_arr = np.array(pred_img)  
        gt_arr = np.array(gt_img)  

        # 将图像转换为二值图像（0 或 1）  
        pred_arr = (pred_arr > 127).astype(np.uint8)  
        gt_arr = (gt_arr > 0).astype(np.uint8)  # 假设gt_arr大于0的像素为前景  

        dice, iou = calculate_dice_iou(pred_arr, gt_arr)  
        mdice_sum += dice  
        miou_sum += iou  
        count += 1  

    mdice = mdice_sum / count  
    miou = miou_sum / count  

    return mdice, miou

# 使用示例  
pred_folder1 = '/opt/data/private/wd/UGPCL-222/shows/Kvasir/0.125/Kvasir/pred'  
gt_folder1 = '/opt/data/private/wd/UGPCL-222/shows/Kvasir/0.125/Kvasir/label'

mdice1, miou1 = evaluate_folders(pred_folder1, gt_folder1)  
print(f"mDice: {mdice1}, mIoU: {miou1}")
