import os
import yaml
import argparse
import warnings
import random

import torch
import numpy as np

from datetime import datetime
from codes.builder import build_dataloader, build_logger
from codes.utils.utils import Namespace, parse_yaml
from codes.trainers import *
import pdb 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/opt/data/private/wd/UGPCL-222/configs/isic/ugpcl_unet_r50.yaml',
    # parser.add_argument('--config', type=str, default='/opt/data/private/zyb/UGPCL-master-baseline/configs/isic/ugpcl_unet_r50.yaml',
                                            # default='configs/comparison_acdc_224_136/ugpcl_unet_r50.yaml',  
                        help='train config file path: xxx.yaml')
    parser.add_argument('--work_dir', type=str,
                        default=f'results/polyp',
                        help='the dir to save logs and models')
    parser.add_argument('--resume_from', type=str,
                        # default='results/comparison_acdc_224_136/ugcl_mem_unet_r50_0430155558/iter_1000.pth',
                        default=None,
                        help='the checkpoint file to resume from')
    parser.add_argument('--start_step', type=int, default=0)
    # parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--verbose', action='store_true', help='训练时是否打印进度条') 
    args = parser.parse_args()

    args_dict = parse_yaml(args.config)

    for key, value in Namespace(args_dict).__dict__.items():
        if key in ['name', 'dataset', 'train', 'logger']:
            vars(args)[key] = value

    for key, value in Namespace(args_dict).__dict__.items():
        if key not in ['name', 'dataset', 'train', 'logger']:
            vars(args.train.kwargs)[key] = value

    if args.work_dir is None:
        args.work_dir = f'results/{args.dataset.name}'
    if args.resume_from is not None:
        args.logger.log_dir = os.path.split(os.path.abspath(args.resume_from))[0]
        args.logger.file_mode = 'a'
    else:
        args.logger.log_dir = f'{args.work_dir}/{args.name}_{datetime.now().strftime("%m%d%H%M%S")}'
    args.ckpt_save_path = args.logger.log_dir
    # print("1")
    # print("args.ckpt_save_path=", args.ckpt_save_path)

    for key in args.__dict__.keys():
        if key not in args_dict.keys():
            args_dict[key] = args.__dict__[key]

    return args, args_dict


def set_deterministic(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_trainer(name,
                  logger=None,
                  device='cuda',
                  data_parallel=False,
                  ckpt_save_path=None,
                  resume_from=None,
                  **kwargs):
    return eval(f'{name}')(logger=logger, device=device, data_parallel=data_parallel, ckpt_save_path=ckpt_save_path,
                           resume_from=resume_from, **kwargs)


def train():
    # 获取命令行参数和参数字典
    args, args_dict = get_args()

    print(f"Current device: {args.device}")
    
    # pdb.set_trace()

    # 设置随机数种子以保证实验的可重复性
    set_deterministic(args.seed)

    # 定义worker初始化函数，设置随机数种子
    def worker_init_fn(worker_id):
        random.seed(worker_id + args.seed)
    


    # 构建训练和验证数据加载器
    train_loader, val_loader = build_dataloader(args.dataset, worker_init_fn)

    # pdb.set_trace()
    # 输出验证集的长度
    print(f"train.py Validation dataset size: {len(val_loader.dataset)}")

    # # 输出一个 batch 的数据形状
    # for batch in val_loader:
    #     images, labels = batch['image'], batch['label']
    #     print(f"Validation batch shape - Images: {images.shape}, Labels: {labels.shape}")
    #     break  # 只输出一个 batch 即可

    # 构建日志记录器
    logger = build_logger(args.logger)

    # 将参数字典以YAML格式保存到文件中
    args_yaml_info = yaml.dump(args_dict, sort_keys=False, default_flow_style=None)
    yaml_file_name = os.path.split(args.config)[-1]
    with open(os.path.join(args.ckpt_save_path, yaml_file_name), 'w') as f:
        f.write(args_yaml_info)
        f.close()

    # 在日志中记录参数信息
    logger.info(f'\n{args_yaml_info}\n')

    # 构建训练器对象
    trainer = build_trainer(name=args.train.name,
                            logger=logger,
                            device=args.device,
                            data_parallel=args.data_parallel,
                            ckpt_save_path=args.ckpt_save_path,
                            resume_from=args.resume_from,
                            **args.train.kwargs.__dict__)
    
    
    # 开始训练
    trainer.train(train_loader, val_loader)

    # 关闭日志记录器
    logger.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # pdb.set_trace()
    train()
