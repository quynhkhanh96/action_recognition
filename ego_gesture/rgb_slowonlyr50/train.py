# Refer: https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb/20210606_010153.log
import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import random
import torch
import torch.nn as nn
import numpy as np 

from core.models.base import Recognizer3D
from core.models.backbones.resnet3d_slowonly import ResNet3dSlowOnly
from core.models.heads.i3d_head import I3DHead
from mmcv.runner.checkpoint import load_checkpoint

from mmaction.datasets import build_dataset
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from mmaction.core.evaluation import top_k_accuracy
from torch.nn.utils import clip_grad_norm_

import argparse

NUM_CLASSES = 83
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count      

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def build_model():
    model = Recognizer3D(
        backbone=ResNet3dSlowOnly(
            depth=50, pretrained='torchvision://resnet50', lateral=False,
            conv1_kernel=(1, 7, 7), conv1_stride_t=1, pool1_stride_t=1,
            inflate=(0, 0, 1, 1), norm_eval=False
        ),
        cls_head=I3DHead(
            in_channels=2048, num_classes=NUM_CLASSES,
            spatial_type='avg',
            dropout_ratio=0.5
        )
    )

    load_from = 'https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth'
    load_checkpoint(model, load_from)

    return model

def build_loaders(args):
    dataset_type = 'RawframeDataset'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

    train_pipeline = [
        dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1),
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
    dst_train_cfg = dict(type=dataset_type,
            ann_file=args.ann_file_train,
            data_prefix=args.data_root,
            pipeline=train_pipeline)

    val_pipeline = [
        dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1,
            test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    dst_val_cfg = dict(type=dataset_type,
            ann_file=args.ann_file_val,
            data_prefix=args.data_root,
            pipeline=val_pipeline)

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

def build_optimizer(model):
    lr = 0.001 / 4
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return optimizer

def train(model, train_loader, val_loader, optimizer, args):
    clip_gradient = 20
    model.to('cuda')

    for epoch in range(40):
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
            if (batch_idx + 1) % 20 == 0:
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
        with open(os.path.join(args.work_dir, 'logs.txt'), 'a') as f:
            f.write(msg)
        print(msg)

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
    args = parser.parse_args()
    args.data_root = os.path.join(args.data_dir, 'rawframes')
    args.ann_file_train = os.path.join(args.data_dir, 'train.txt')
    args.ann_file_val = os.path.join(args.data_dir, 'test.txt')

    model = build_model()
    train_loader, val_loader = build_loaders(args)
    optimizer = build_optimizer(model)
    train(model, train_loader, val_loader, optimizer, args)