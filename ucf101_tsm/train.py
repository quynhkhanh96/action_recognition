import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument(
        "--data_root",
        type=str,
        help="Where train rawframes are stored",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        help="working dir path",
    )
    parser.add_argument(
        "--data_root_val",
        type=str,
        help="Where val rawframes are stored",
    )
    parser.add_argument(
        "--ann_file_train",
        type=str,
        help="Path to train annotation file",
    )
    parser.add_argument(
        "--ann_file_val",
        type=str,
        help="Path to val annotation file",
    )
    parser.add_argument(
        "--videos_per_gpu",
        type=int,
        default=10,
        help="Number of videos per gpu",
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    cfg = Config.fromfile('configs/ucf101_rgb_tsm_k400_pretrained_r50_1x1x8.py')
    cfg.omnisource = False 
    # update config's paths according to args
    cfg.data.train.data_prefix = args.data_root 
    cfg.data.val.data_prefix = args.data_root_val
    cfg.work_dir = args.work_dir 
    cfg.data.train.ann_file = args.ann_file_train
    cfg.data.val.ann_file = args.ann_file_val 
    cfg.seed = int(args.seed)
    # cfg.gpu_ids = range(1) #TODO
    num_gpus = torch.cuda.device_count()
    cfg.gpu_ids = range(num_gpus)
    # model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # datasets
    datasets = [build_dataset(cfg.data.train)]

    distributed = False 
    cfg.data.videos_per_gpu = args.videos_per_gpu
    # if num_gpus > 1:
    #     distributed = True 
    #     cfg.data.videos_per_gpu = 10 
    train_model(model, datasets, cfg, distributed=distributed, validate=False)

    

