from .dataloaders import *
from .losses import *
from .metrics import *
from .models import *
from .schedulers import *
from .losses import *
from .logger import Logger

from torch.optim.lr_scheduler import *
from torch.optim import *

import pdb

__all__ = ['build_model', 'build_metric', 'build_dataloader', 'build_scheduler',
           'build_optimizer', 'build_criterion', 'build_logger', '_build_from_cfg']


def _build_from_cfg(cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            return eval(f"{name}")(**cfg['kwargs'])
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            return eval(f"{name}")(**cfg.kwargs.__dict__)
    return eval(f"{name}()")


def build_model(cfg):
    return _build_from_cfg(cfg)


def build_optimizer(model_parameters, cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            kwargs = cfg['kwargs']
            kwargs['params'] = model_parameters
            return eval(f"{name}")(**kwargs)
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            kwargs = cfg.kwargs.__dict__
            kwargs['params'] = model_parameters
            return eval(f"{name}")(**kwargs)
    kwargs = {'params': model_parameters}
    return eval(f"{name}")(**kwargs)


def build_scheduler(optimizer_, cfg):
    if isinstance(cfg, dict):
        name = cfg['name']
        if 'kwargs' in cfg.keys() and cfg['kwargs'] is not None:
            kwargs = cfg['kwargs']
            kwargs['optimizer'] = optimizer_
            return eval(f"{name}")(**kwargs)
    else:
        name = cfg.name
        if hasattr(cfg, 'kwargs') and cfg.kwargs is not None:
            kwargs = cfg.kwargs.__dict__
            kwargs['optimizer'] = optimizer_
            return eval(f"{name}")(**kwargs)
    kwargs = {'optimizer': optimizer_}
    return eval(f"{name}")(**kwargs)


def build_criterion(cfg):
    return _build_from_cfg(cfg)


def build_metric(cfg):
    return _build_from_cfg(cfg)


def build_dataloader(cfg, worker_init_fn):
    # 构建数据加载器的函数
    # pdb.set_trace()  
    # 检查配置是否为字典类型，根据情况提取参数字典
    if not isinstance(cfg, dict):
        kwargs = cfg.kwargs.__dict__
    else:
        kwargs = cfg['kwargs']

    # 将 worker_init_fn 参数添加到参数字典中
    kwargs['worker_init_fn'] = worker_init_fn

    # 动态构建数据加载器函数的名称，例如 get_acdc_loaders
    loader_function_name = f"get_{cfg.name}_loaders"
    print(loader_function_name)

    # 调用数据加载器函数并返回加载器对象
    return eval(loader_function_name)(**kwargs)



def build_logger(cfg):
    return eval(f"Logger")(**cfg.__dict__)
