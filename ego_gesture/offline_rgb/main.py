import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import random
import torch
import torch.nn as nn
import numpy as np 

from mmaction.datasets import build_dataset
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from mmaction.core.evaluation import top_k_accuracy
from torch.nn.utils import clip_grad_norm_

from core.models.base import build_model
from utils import AverageMeter, seed_everything, Dict2Class
import argparse
import yaml

seed_everything()

def get_model(cfgs):
    model = build_model(cfgs, 'train')

    return model

def get_loaders(args, cfgs):
    if cfgs.arch.startswith('slow'):
        dataset_type = 'RawframeDataset'
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

        train_pipeline = [
            dict(type='SampleFrames', clip_len=cfgs.seq_len, frame_interval=4, num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        val_pipeline = [
            dict(type='SampleFrames', clip_len=cfgs.seq_len, frame_interval=4, num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]
    elif cfgs.arch.startswith('i3d'):
        dataset_type = 'RawframeDataset'
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        train_pipeline = [
            dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.8),
                random_crop=False,
                max_wh_scale_gap=0),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        val_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]

    dst_train_cfg = dict(type=dataset_type, ann_file=args.ann_file_train,
            data_prefix=args.data_root, pipeline=train_pipeline)
    dst_val_cfg = dict(type=dataset_type, ann_file=args.ann_file_val,
            data_prefix=args.data_root, pipeline=val_pipeline)

    train_dataset = build_dataset(dst_train_cfg)
    val_dataset = build_dataset(dst_val_cfg)

    train_loader = DataLoader(train_dataset, batch_size=4, pin_memory=True, 
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True, 
                            shuffle=False, drop_last=False)
    
    return train_loader, val_loader

def criterion(cls_score, labels):
    labels = labels.squeeze(1)
    loss_cls = nn.CrossEntropyLoss()
    loss = loss_cls(cls_score, labels)

    return loss

def get_optimizer(model):
    lr = 0.001 / 4
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return optimizer

def train(model, train_loader, val_loader, optimizer, args, cfgs):
    clip_gradient = 20
    model.to('cuda')

    best = -1
    for epoch in range(cfgs.epochs):
        model.train()
        print(f'*************** Epoch {epoch} ***************')
        losses = AverageMeter("loss")
        for batch_idx, data in enumerate(train_loader):
            imgs, labels = data['imgs'], data['label']
            imgs, labels = Variable(imgs).to('cuda'), Variable(labels).to('cuda')
            outputs = model(imgs) 
            loss = criterion(outputs, labels)       
            optimizer.zero_grad()
            loss.backward()
            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)
            optimizer.step()
            losses.update(loss.item(), labels.size(0))
            if (batch_idx + 1) % cfgs.print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, 
                                    len(train_loader), losses.val, losses.avg))
        
        model.eval()
        results, labels = [], []
        for _, val_data in enumerate(val_loader):
            val_imgs, val_labels = val_data['imgs'], val_data['label']
            val_labels = val_labels.to('cuda')
            
            with torch.no_grad():
                val_imgs = Variable(val_imgs).to('cuda')
                outputs = model(val_imgs)

            results.extend(outputs.cpu().numpy())
            labels.extend(val_labels.cpu().numpy())

        top1_acc, top5_acc = top_k_accuracy(results, labels, topk=(1, 5))
        msg = f'[INFO]Epoch {epoch}: Top1 accuracy = {top1_acc:.3f}, Top5 accuracy = {top5_acc:.3f}\n'
        with open(os.path.join(args.work_dir, 'logs_{}_{}e.txt'.format(cfgs.arch, cfgs.epochs)), 'a') as f:
            f.write(msg)
        print(msg)
        if top1_acc > best:
            best = top1_acc
            torch.save({'state_dict': model.state_dict()}, args.work_dir + '/best.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ExoGesture')
    parser.add_argument(
        "--data_dir",
        type=str
    )
    parser.add_argument(
        "--work_dir",
        type=str
    )
    parser.add_argument(
        "--cfg_path",
        type=str
    )
    args = parser.parse_args()
    args.data_root = os.path.join(args.data_dir, 'rawframes')
    args.ann_file_train = os.path.join(args.data_dir, 'train.txt')
    args.ann_file_val = os.path.join(args.data_dir, 'test.txt')

    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    model = get_model(cfgs)
    train_loader, val_loader = get_loaders(args, cfgs)
    optimizer = get_optimizer(model)
    train(model, train_loader, val_loader, optimizer, args, cfgs)