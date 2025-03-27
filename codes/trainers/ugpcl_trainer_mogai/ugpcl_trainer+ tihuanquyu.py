import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable
from colorama import Fore
from torchvision.utils import make_grid
from ._base import BaseTrainer
from ..utils import ramps
from ..losses import PixelContrastLoss
import pdb

class UGPCLTrainer(BaseTrainer):

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=8,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=6000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000,
                 consistency=0.1,
                 consistency_rampup=40.0,
                 tf_decoder_weight=0.4,  # 公式3，监督损失的辅助预测因子的权重
                 cls_weight=0.1,
                 contrast_type='ugpcl',  # ugpcl, pseudo, sup, none
                 contrast_weight=0.1,
                 temperature=0.1,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=1,
                 memory=True,
                 memory_size=100,
                 pixel_update_freq=10,
                 pixel_classes=4,
                 dim=256, 
                 region_size=32 # 新增参数: 区域大小
                 ) -> None:

        super(UGPCLTrainer, self).__init__(model, optimizer, scheduler, criterions, metrics, logger, device,
                                           resume_from, labeled_bs, consistency, consistency_rampup, data_parallel,
                                           ckpt_save_path, max_iter, eval_interval, save_image_interval,
                                           save_ckpt_interval)

        self.tf_decoder_weight = tf_decoder_weight
        self.cls_weight = cls_weight
        self.cls_criterion = torch.nn.CrossEntropyLoss()

        self.contrast_type = contrast_type
        self.contrast_weight = contrast_weight
        self.contrast_criterion = PixelContrastLoss(temperature=temperature,
                                                    base_temperature=base_temperature,
                                                    max_samples=max_samples,
                                                    max_views=max_views,
                                                    device=device)
        # memory param
        self.memory = memory
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq
       
        if self.memory:
            self.segment_queue = torch.randn(pixel_classes, self.memory_size, dim)
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
            self.pixel_queue = torch.zeros(pixel_classes, self.memory_size, dim)
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

           # 修改部分：初始化区域大小
        self.region_size = region_size

    # 新增方法: 划分特征图为多个区域
    def _divide_into_regions(self, features):
        """
        将特征图划分为多个区域，区域的大小由 self.region_size 决定
        """
        batch_size, channels, height, width = features.shape
        regions = []
        for i in range(0, height, self.region_size):
            for j in range(0, width, self.region_size):
                region = features[:, :, i:i + self.region_size, j:j + self.region_size]
                regions.append(region)
        return regions

    # 新增方法: 计算每个区域的对比损失
    def _region_contrast_loss(self, region, labels):
        """
        在每个区域内计算对比损失
        """
        loss = self.contrast_criterion(region, labels)
        return loss

    def train_step(self, batch_data, step, save_image):
        # 初始化日志信息和标量信息
        log_infos, scalars = {}, {}
        images = {}








    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x > 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size

    @staticmethod
    def _random_rotate(image, label):
        angle = float(torch.empty(1).uniform_(-20., 20.).item())
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)
        return image, label

    def train_step(self, batch_data, step, save_image):
        # 初始化日志信息和标量信息
        log_infos, scalars = {}, {}
        # 存储用于可视化的图像
        images = {}

        # 将图像和标签移动到指定的设备上
        data_, label_ = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        
        # 根据分类权重进行数据处理
        if self.cls_weight >= 0.:
            images_, labels_ = [], []
            cls_label = []
            
            # 对每个图像和标签进行数据增强
            for image, label in zip(data_, label_):
                rot_times = random.randrange(0, 4)
                cls_label.append(rot_times)
                image = torch.rot90(image, rot_times, [1, 2])
                label = torch.rot90(label, rot_times, [1, 2])
                image, label = self._random_rotate(image, label)
                images_.append(image)
                labels_.append(label)
            cls_label = torch.tensor(cls_label).to(self.device)
            data = torch.stack(images_, dim=0).to(self.device)
            label = torch.stack(labels_, dim=0).to(self.device)
        else:
            # 没有分类权重时，直接使用原始图像和标签
            data = data_
            label = label_
            cls_label = None

        # 模型的前向传播
        outputs = self.model(data, self.device)
        seg = outputs['seg']  # cnn解码器输出
        seg_tf = outputs['seg_tf']  # tf解码器输出

        # 计算监督损失
        # 先初始化为0，循环计算每个损失函数的损失
        supervised_loss = 0.
        # criterion，包含loss_dice和loss_ce,对多个损失进行逐一计算，并将其累加到监督损失中，并保存到日志中
        for criterion in self.criterions:
            loss_ = criterion(seg[:self.labeled_bs], label[:self.labeled_bs]) + \
                    self.tf_decoder_weight * criterion(seg_tf[:self.labeled_bs], label[:self.labeled_bs])  # 公式3
            supervised_loss += loss_  # 累加到总的监督损失中
            log_infos[criterion.name] = float(format(loss_, '.5f'))  # 转换为浮点数，并保存在日志中
            # print("supervised_loss的",criterion.name,log_infos[criterion.name])
            scalars[f'loss/{criterion.name}'] = loss_  
        # pdb.set_trace()
        # 计算分类损失
        loss_cls = self.cls_criterion(outputs['cls'], cls_label) if self.cls_weight > 0. else 0.
        # print("分类损失loss_cls ",criterion.name,log_infos[criterion.name])

        # 计算分割的软标签
        seg_soft = torch.softmax(seg, dim=1)
        seg_tf_soft = torch.softmax(seg_tf, dim=1)

        # 计算一致性损失
        consistency_weight = self.get_current_consistency_weight(step // 100)  # 根据训练步骤动态调整权重
        # print("consistency_weight=",consistency_weight)
        consistency_loss = torch.mean((seg_soft[self.labeled_bs:] - seg_tf_soft[self.labeled_bs:]) ** 2)  # 计算两个输出的软标签的一致性
        # print("consistency_loss=",consistency_loss)

        # 计算总损失
        loss = supervised_loss + consistency_weight * consistency_loss + self.cls_weight * loss_cls
        # print("loss=",loss)

        # 记录损失和一些信息到日志和标量
        log_infos['loss_cls'] = float(format(loss_cls, '.5f'))
        log_infos['con_weight'] = float(format(consistency_weight, '.5f'))
        log_infos['loss_con'] = float(format(consistency_loss, '.5f'))
        log_infos['loss'] = float(format(loss, '.5f'))

        scalars['loss/loss_cls'] = loss_cls
        scalars['consistency_weight'] = consistency_weight
        scalars['loss/loss_consistency'] = consistency_loss
        scalars['loss/total'] = loss

        # 计算分割预测
        preds = torch.argmax(seg_soft, dim=1, keepdim=True).to(torch.float)

         # 修改部分：划分区域并计算区域内的对比损失
        log_infos['loss_contrast'] = 0.
        scalars['loss/contrast'] = 0.
        if step > 1:
            queue = self.segment_queue if self.memory else None
            if self.contrast_type == 'ugpcl':
                seg_mean = torch.mean(torch.stack([F.softmax(seg, dim=1), F.softmax(seg_tf, dim=1)]), dim=0)
                uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, self.max_iter)) * np.log(2)
                uncertainty_mask = (uncertainty > threshold)
                mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
                certainty_pseudo = mean_preds.clone()
                certainty_pseudo[:self.labeled_bs] = label[:self.labeled_bs]
                certainty_pseudo[uncertainty_mask] = -1


                    # 新增部分：计算基于区域的对比损失
                regions = self._divide_into_regions(outputs['embed'])
                contrast_loss = 0
                for region in regions:
                    contrast_loss += self._region_contrast_loss(region, certainty_pseudo)

                scalars['uncertainty_rate'] = torch.sum(uncertainty_mask == True) / \
                                            (torch.sum(uncertainty_mask == True) + torch.sum(
                                                uncertainty_mask == False))



        # # 计算对比损失（如果符合条件）
        # # 初始化两个字典中对应的键值
        # log_infos['loss_contrast'] = 0.
        
        # scalars['loss/contrast'] = 0.
        # # if step > 1000 and self.contrast_weight > 0.:  # 初始contrast_weight为0.1
        # if step > 1:
        #     # print("对比学习损失开始！")
        #     queue = self.segment_queue if self.memory else None  # 判断是否使用了内存队列，使用了则指向self.segment_queue
            
        #     # 根据对比类型进行处理
        #     if self.contrast_type == 'ugpcl':
        #         # 计算不确定性，并且创建不确定性掩码
        #         # 将不同解码器的输出软标签，进行softmax处理后，叠加取均值
        #         seg_mean = torch.mean(torch.stack([F.softmax(seg, dim=1), F.softmax(seg_tf, dim=1)]), dim=0)
        #         # 使用交叉熵损失，来计算不确定性
        #         uncertainty = -1.0 * torch.sum(seg_mean * torch.log(seg_mean + 1e-6), dim=1, keepdim=True)
        #         # 计算阈值threshold，根据训练进度调整该阈值
        #         threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, self.max_iter)) * np.log(2)
        #         # 根据阈值，设计不确定性掩码
        #         uncertainty_mask = (uncertainty > threshold)
                
        #         # 构建伪标签
        #         # 选择最大概率的类别作为预测标签，并转换为浮点型
        #         mean_preds = torch.argmax(F.softmax(seg_mean, dim=1).detach(), dim=1, keepdim=True).float()
        #         # 创建与mean_preds相同的张量，用于存储伪标签
        #         certainty_pseudo = mean_preds.clone()
                
        #         # 将真实标签赋值给相应位置的伪标签？
        #         # 但是很奇怪，模型图中此处计算的应为未标注数据，也即没有真实标签，只有伪标签？
        #         # 另外，上一条语句划分了不确定区域，这条语句重新覆盖了真实标签，不是会导致划分的不确定区域重置嘛？
        #         certainty_pseudo[:self.labeled_bs] = label[:self.labeled_bs]

        #         # 在uncertainty_mask为true的位置上，伪标签设置为-1
        #         # 即不确定区域，伪标签设置为-1
        #         certainty_pseudo[uncertainty_mask] = -1
                
        #         # 计算不确定性引导的像素级对比损失
        #         contrast_loss = self.contrast_criterion(outputs['embed'], certainty_pseudo, preds, queue=queue)
        #         # 计算不确定掩码中不确定区域占整个区域的比率
        #         scalars['uncertainty_rate'] = torch.sum(uncertainty_mask == True) / \
        #                                     (torch.sum(uncertainty_mask == True) + torch.sum(
        #                                         uncertainty_mask == False))
        #         # 如果使用内存队列，则将嵌入结果和不确定标签一起存储到队列中
                if self.memory:
                    self._dequeue_and_enqueue(outputs['embed'].detach(), certainty_pseudo.detach())
                
                if save_image:
                    grid_image = make_grid(mean_preds * 50., 4, normalize=False)  # 平均预测标签
                    images['train/mean_preds'] = grid_image
                    grid_image = make_grid(certainty_pseudo * 50., 4, normalize=False)  # 伪标签
                    images['train/certainty_pseudo'] = grid_image
                    grid_image = make_grid(uncertainty, 4, normalize=False)  # 不确定性
                    images['train/uncertainty'] = grid_image
                    grid_image = make_grid(uncertainty_mask.float(), 4, normalize=False)  # 不确定性掩码
                    images['train/uncertainty_mask'] = grid_image
            # elif self.contrast_type == 'pseudo':
            #     # 使用伪标签进行对比
            #     contrast_loss = self.contrast_criterion(outputs['embed'], preds.detach(), preds, queue=queue)
            #     if self.memory:
            #         self._dequeue_and_enqueue(outputs['embed'].detach(), preds.detach())
            # elif self.contrast_type == 'sup':
            #     # 使用监督标签进行对比
            #     contrast_loss = self.contrast_criterion(outputs['embed'][:self.labeled_bs], label[:self.labeled_bs],
            #                                             preds[:self.labeled_bs], queue=queue)
            #     if self.memory:
            #         self._dequeue_and_enqueue(outputs['embed'].detach()[:self.labeled_bs],
            #                                 label.detach()[:self.labeled_bs])
            # else:
            #     contrast_loss = 0.
            
            # 计算总损失
            loss += self.contrast_weight * contrast_loss
            log_infos['loss_contrast'] = float(format(contrast_loss, '.5f'))
            scalars['loss/contrast'] = contrast_loss

        # 计算无监督分割预测
        tf_preds = torch.argmax(seg_tf_soft, dim=1, keepdim=True).to(torch.float)  #在 seg_tf_soft 张量的第1维度上进行 argmax 操作找到最大索引，并保持原有维度。且转化为浮点类型
        
        # 计算评估指标
        metric_res = self.metrics[0](preds, label)  # 拿预测值和标签进行评估指标计算，self.metrics[0]从列表中选择第一个度量函数,将预测结果 preds 和真实标签 label 作为输入，传递给度量函数
        # key=bg,c1
        for key in metric_res.keys():
            log_infos[f'{self.metrics[0].name}.{key}'] = float(format(metric_res[key], '.5f'))
            # print(key,"=",log_infos[f'{self.metrics[0].name}.{key}'])
            scalars[f'train/{self.metrics[0].name}.{key}'] = metric_res[key]

        # 如果需要保存图像，将图像添加到images字典中
        if save_image:
            grid_image = make_grid(data, 4, normalize=True)
            images['train/images'] = grid_image
            grid_image = make_grid(preds * 50., 4, normalize=False)
            images['train/preds'] = grid_image
            grid_image = make_grid(tf_preds * 50., 4, normalize=False)
            images['train/tf_preds'] = grid_image
            grid_image = make_grid(label * 50., 4, normalize=False)
            images['train/labels'] = grid_image

        # 返回损失、日志信息、标量信息和图像信息
        return loss, log_infos, scalars, images


    def val_step(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference(data)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    def val_step_tf(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference_tf(data, self.device)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    @torch.no_grad()
    def val_tf(self, val_loader, test=False):
        # 将模型设置为评估模式
        self.model.eval()
        
        # 初始化评估结果和标量值
        val_res = None
        val_scalars = {}
        
        # 打印评估信息（如果有 logger）
        if self.logger is not None:
            self.logger.info('Evaluating...')
        
        # 如果是测试模式，使用 tqdm 包装验证集加载器以显示进度条
        if test:
            val_loader = tqdm(val_loader, desc='Testing', unit='batch',
                            bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))
        
        # 遍历验证集，进行模型验证
        for batch_data in val_loader:
            # 调用 val_step_tf 函数进行验证步骤，得到一个字典形式的批次结果
            batch_res = self.val_step_tf(batch_data)  # {'Dice':{'c1':0.1, 'c2':0.1, ...}, ...}
            
            # 汇总验证结果
            if val_res is None:
                val_res = batch_res
            else:
                for metric_name in val_res.keys():
                    for key in val_res[metric_name].keys():
                        val_res[metric_name][key] += batch_res[metric_name][key]
        
        # 计算每个评估指标的平均值，并记录到标量值中
        for metric_name in val_res.keys():
            for key in val_res[metric_name].keys():
                val_res[metric_name][key] = val_res[metric_name][key] / len(val_loader)
                val_scalars[f'val_tf/{metric_name}.{key}'] = val_res[metric_name][key]

            # 计算每个评估指标的平均值，并记录到标量值中
            val_res_list = [_.cpu() for _ in val_res[metric_name].values()]
            val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
            val_scalars[f'val_tf/{metric_name}.Mean'] = val_res[metric_name]['Mean']

        # 创建一个 PrettyTable 表格对象
        val_table = PrettyTable()

        # 设置表格的列名，包括 'Metric' 和来自 val_res 的指标名称
        val_table.field_names = ['Metric'] + list(list(val_res.values())[0].keys())

        # 遍历每个评估指标的结果，将结果添加到 PrettyTable 表格中
        for metric_name in val_res.keys():
            # 根据指标名称选择性地对结果进行百分比转换
            if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
                temp = [float(format(_ * 100, '.2f')) for _ in val_res[metric_name].values()]
            else:
                temp = [float(format(_, '.2f')) for _ in val_res[metric_name].values()]

            # 将评估指标名称和结果添加到表格中
            val_table.add_row([metric_name] + temp)

        # 返回填充好的表格，以及其他可能的结果（val_res 和 val_scalars）
        return val_res, val_scalars, val_table



    # UGPCL Trainer 的训练函数，用于模型的训练过程
    def train(self, train_loader, val_loader):
    
        # print("start ugpcl_trainer.py train()")

        # iter_train_loader = iter(train_loader)
        max_epoch = self.max_iter // len(train_loader) + 1  # 计算最大的训练轮数

        # print("max_epoch", max_epoch)

        step = self.start_step  # 训练步数，初始为 self.start_step
        self.model.train()  # 将模型设置为训练模式
        with tqdm(total=self.max_iter - self.start_step, bar_format='[{elapsed}<{remaining}] ') as pbar:
            # 使用 tqdm 创建一个进度条，显示训练的进度
            for _ in range(max_epoch):
                for batch_data in train_loader:
                    # save_image = True if (step + 1) % self.save_image_interval == 0 else False
                    save_image = True
                    # 判断是否保存图片，根据 save_image_interval 参数判断

                    # 执行训练步骤，获取损失、日志信息、标量、图像
                    loss, log_infos, scalars, images = self.train_step(batch_data, step, save_image)

                    # 梯度清零，反向传播，优化器更新参数
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()  # 更新学习率

                    # 每 10 步打印一次学习率、日志信息，更新日志记录
                    if (step + 1) % 10 == 0:
                        scalars.update({'lr': self.scheduler.get_lr()[0]})
                        log_infos.update({'lr': self.scheduler.get_lr()[0]})
                        self.logger.update_scalars(scalars, step + 1)
                        self.logger.info(f'[{step + 1}/{self.max_iter}] {log_infos}')

                    # 如果需要保存图片，更新日志记录中的图像信息
                    if save_image:
                        self.logger.update_images(images, step + 1)

                    # 每 eval_interval 步进行一次验证
                    if (step + 1) % self.eval_interval == 0:
                        if val_loader is not None:
                            # 执行验证步骤，记录验证结果
                            val_res, val_scalars, val_table = self.val(val_loader)
                            self.logger.info(f'val result:\n{val_table.get_string()}')
                            self.logger.update_scalars(val_scalars, step + 1)
                            self.model.train()

                            # 执行另一种验证步骤，记录验证结果
                            val_res, val_scalars, val_table = self.val_tf(val_loader)
                            self.logger.info(f'val_tf result:\n{val_table.get_string()}')
                            self.logger.update_scalars(val_scalars, step + 1)
                            self.model.train()

                    # 每 save_ckpt_interval 步保存一次模型参数
                    if (step + 1) % self.save_ckpt_interval == 0:
                        num = 0
                        num = num + 1
                        # print("第", num, "次save_ckpt_interval 步保存一次模型参数")
                        if not os.path.exists(self.ckpt_save_path):
                            os.makedirs(self.ckpt_save_path)
                        self.save_ckpt(step + 1, f'{self.ckpt_save_path}/iter_{step + 1}.pth')

                    # 更新步数，更新进度条
                    step += 1
                    pbar.update(1)

                    # 判断是否达到最大训练步数，如果是，则跳出循环
                    if step >= self.max_iter:
                        break

                # 判断是否达到最大训练步数，如果是，则跳出循环
                if step >= self.max_iter:
                    break

        # 保存最终模型参数
        if not os.path.exists(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)
            torch.save(self.model.state_dict(), f'{self.ckpt_save_path}/ckpt_final.pth')