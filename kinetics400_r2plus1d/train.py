import argparse
import copy
import os
import os.path as osp
import time
import warnings
from attr import validate

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

def main():
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
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    cfg = Config.fromfile('configs/kinetics400_rgb.py')

    # update config's paths according to args
    cfg.data_root = args.data_root 
    cfg.data_root_val = args.data_root_val
    cfg.work_dir = args.work_dir 
    cfg.ann_file_train = args.ann_file_train
    cfg.ann_file_val = args.ann_file_val 
    cfg.seed = int(args.seed)
    cfg.gpu_ids = range(1) #TODO

    # model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # datasets
    datasets = [build_dataset(cfg.data.train)]

    train_model(model, datasets, cfg, distributed=False, validate=False)

    
