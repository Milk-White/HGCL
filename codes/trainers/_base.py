import os
from abc import abstractmethod

import numpy as np
import torch

from tqdm import tqdm
from prettytable import PrettyTable
from colorama import Fore
from ..utils import ramps
from ..builder import _build_from_cfg, build_model, build_optimizer, build_scheduler


class BaseTrainer:

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=12,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=10000,
                 eval_interval=1000,
                 save_image_interval=50,
                 save_ckpt_interval=2000) -> None:
        super(BaseTrainer, self).__init__()
        self.model = None
        # build cfg
        if model is not None:
            self.model = build_model(model).to(device)
        if optimizer is not None:
            self.optimizer = build_optimizer(self.model.parameters(), optimizer)
        if scheduler is not None:
            self.scheduler = build_scheduler(self.optimizer, scheduler)
        self.criterions = []
        if criterions is not None:
            for criterion_cfg in criterions:
                self.criterions.append(_build_from_cfg(criterion_cfg))
        self.metrics = []
        if metrics is not None:
            for metric_cfg in metrics:
                self.metrics.append(_build_from_cfg(metric_cfg))

        # semi-supervised params
        self.labeled_bs = labeled_bs
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup

        # train params
        self.logger = logger
        self.device = device
        self.data_parallel = data_parallel
        self.ckpt_save_path = ckpt_save_path

        self.max_iter = max_iter
        self.eval_interval = eval_interval
        self.save_image_interval = save_image_interval
        self.save_ckpt_interval = save_ckpt_interval

        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        if resume_from is not None:
            ckpt = torch.load(resume_from)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            scheduler.load_state_dict(ckpt['scheduler'])
            self.start_step = ckpt['step']

            logger.info(f'Resume from {resume_from}.')
            logger.info(f'Train from step {self.start_step}.')
        else:
            self.start_step = 0
            if self.model is not None:
                logger.info(f'\n{self.model}\n')

        logger.info(f'start training...')

    @abstractmethod
    def train_step(self, batch_data, step, save_image):
        loss = 0.
        log_infos, scalars, images = {}, {}, {}
        return loss, log_infos, scalars, images

    def val_step(self, batch_data):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = self.model.inference(data)
        metric_total_res = {}
        for metric in self.metrics:
            metric_total_res[metric.name] = metric(preds, labels)
        return metric_total_res

    # 训练函数，用于模型的训练过程
    def train(self, train_loader, val_loader):
        print("base_trainer")

        # iter_train_loader = iter(train_loader)
        # 计算最大的训练轮数
        max_epoch = self.max_iter // len(train_loader) + 1
        step = self.start_step  # 训练步数，初始为 self.start_step
        self.model.train()  # 将模型设置为训练模式
        with tqdm(total=self.max_iter - self.start_step, bar_format='[{elapsed}<{remaining}] ') as pbar:
            # 使用 tqdm 创建一个进度条，显示训练的进度
            for _ in range(max_epoch):
                for batch_data in train_loader:
                    save_image = True if (step + 1) % self.save_image_interval == 0 else False
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
                            val_res, val_scalars, val_table = self.val(val_loader)
                            self.logger.info(f'val result:\n{val_table.get_string()}')
                            self.logger.update_scalars(val_scalars, step + 1)
                            self.model.train()

                    # 每 save_ckpt_interval 步保存一次模型参数
                    if (step + 1) % self.save_ckpt_interval == 0:
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

    @torch.no_grad()
    def val(self, val_loader, test=False):
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
            # 调用 val_step 函数进行验证步骤，得到一个字典形式的批次结果
            batch_res = self.val_step(batch_data)  # {'Dice':{'c1':0.1, 'c2':0.1, ...}, ...}
            
            # 汇总验证结果
            if val_res is None:
                val_res = batch_res
                # print("val_res is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                for metric_name in val_res.keys():
                    for key in val_res[metric_name].keys():
                        val_res[metric_name][key] += batch_res[metric_name][key]
        
        # 计算每个评估指标的平均值，并记录到标量值中
        for metric_name in val_res.keys():
            for key in val_res[metric_name].keys():
                val_res[metric_name][key] = val_res[metric_name][key] / len(val_loader)
                val_scalars[f'val/{metric_name}.{key}'] = val_res[metric_name][key]

            # 计算每个评估指标的平均值，并记录到标量值中
            val_res_list = [_.cpu() for _ in val_res[metric_name].values()]
            val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
            val_scalars[f'val/{metric_name}.Mean'] = val_res[metric_name]['Mean']

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

            # 打印 temp 的值
            print(f"metric_name: {metric_name}, temp: {temp}")  

            # 将评估指标名称和结果添加到表格中
            val_table.add_row([metric_name] + temp)

        # 返回填充好的表格，以及其他可能的结果（val_res 和 val_scalars）
        return val_res, val_scalars, val_table


        

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * ramps.sigmoid_rampup(epoch, self.consistency_rampup)

    def save_ckpt(self, step, save_path):
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step}
        torch.save(ckpt, save_path)
        self.logger.info('Checkpoint saved!')
