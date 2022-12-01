import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import *
import yaml
import argparse
from mmaction.core.evaluation import top_k_accuracy

def train(train_loader, val_loader, model, criterion, optimizer, cfgs):

    for epoch in range(cfgs.begin_epoch, cfgs.n_epochs + 1):
        print(f'*************** Epoch {epoch} ***************')
        # === train this epoch ===
        ## adjust the learning rate
        adjust_learning_rate(optimizer, i, cfgs)
        model.train()
        losses = AverageMeter()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    
            optimizer.zero_grad()
            loss.backward()
            # if clip_gradient is not None:
            #     _ = clip_grad_norm_(model.parameters(), clip_gradient)
            optimizer.step()
            losses.update(loss.data, inputs.size(0))
            if (i + 1) % 50 == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(i + 1, 
                                    len(train_loader), losses.val, losses.avg))
        # === validate this epoch ===
        model.eval()
        results, labels = [], []
        for _, (val_inputs, val_targets) in enumerate(val_loader):
            val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
            val_inputs, val_targets = Variable(val_inputs), Variable(val_targets)
            
            with torch.no_grad():
                val_outputs = model(val_inputs)

            results.extend(val_outputs.cpu().numpy())
            labels.extend(val_targets.cpu().numpy())

        top1_acc, top5_acc = top_k_accuracy(results, labels, topk=(1, 5))
        msg = f'[INFO]Epoch {epoch}: Top1 accuracy = {top1_acc:.3f}, Top5 accuracy = {top5_acc:.3f}\n'
        with open(os.path.join(cfgs.result_path, 'logs.txt'), 'a') as f:
            f.write(msg)
        print(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ExoGesture')
    parser.add_argument(
        "--root_path",
        type=str
    )
    parser.add_argument(
        "--video_path",
        type=str
    )
    parser.add_argument(
        "--annotation_path",
        type=str
    )
    parser.add_argument(
        "--result_path",
        type=str
    )
    # parser.add_argument(
    #     "--pretrain_path",
    #     type=str
    # )
    parser.add_argument(
        "--cfg_path",
        type=str
    )
    args = parser.parse_args()
    # configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)
    cfgs.root_path = args.root_path
    cfgs.video_path = args.video_path
    cfgs.annotation_path = args.annotation_path
    cfgs.result_path = args.result_path
    
    cfgs.n_classes = int(cfgs.n_classes)
    cfgs.n_finetune_classes = int(cfgs.n_finetune_classes)
    cfgs.sample_size = int(cfgs.sample_size)
    cfgs.sample_duration = int(cfgs.sample_duration)
    cfgs.downsample = int(cfgs.downsample)
    cfgs.initial_scale = float(cfgs.initial_scale)
    cfgs.n_scales = int(cfgs.n_scales)
    cfgs.scale_step = float(cfgs.scale_step)
    cfgs.learning_rate = float(cfgs.learning_rate)
    cfgs.momentum = float(cfgs.momentum)
    cfgs.dampening = float(cfgs.dampening)
    cfgs.weight_decay = float(cfgs.weight_decay)
    cfgs.lr_patience = int(cfgs.lr_patience)
    cfgs.batch_size = int(cfgs.batch_size)
    cfgs.n_epochs = int(cfgs.n_epochs)
    cfgs.begin_epoch = int(cfgs.begin_epoch)
    cfgs.n_val_samples = int(cfgs.n_val_samples)
    cfgs.scale_in_test = float(cfgs.scale_in_test)
    cfgs.n_threads = int(cfgs.n_threads)
    cfgs.norm_value = int(cfgs.norm_value)
    cfgs.version = float(cfgs.version)
    cfgs.model_depth = float(cfgs.model_depth)
    cfgs.wide_resnet_k = int(cfgs.wide_resnet_k)
    cfgs.resnext_cardinality = int(cfgs.resnext_cardinality)
    cfgs.groups = int(cfgs.groups)
    cfgs.width_mult = float(cfgs.width_mult)

    cfgs.scales = [cfgs.initial_scale]
    for i in range(1, cfgs.n_scales):
        cfgs.scales.append(cfgs.scales[-1] * cfgs.scale_step)
    cfgs.arch = '{}'.format(cfgs.model)
    cfgs.mean = get_mean(cfgs.norm_value, dataset=cfgs.mean_dataset)
    cfgs.std = get_std(cfgs.norm_value)
    cfgs.store_name = '_'.join([cfgs.dataset, cfgs.model, str(cfgs.width_mult) + 'x',
                               cfgs.modality, str(cfgs.sample_duration)])

    # seed everything
    torch.manual_seed(cfgs.manual_seed)

    # model
    model, parameters = generate_model(cfgs)

    # loss function
    criterion = nn.CrossEntropyLoss()
    if not cfgs.no_cuda:
        criterion = criterion.cuda()

    # data loaders
    if cfgs.no_mean_norm and not cfgs.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not cfgs.std_norm:
        norm_method = Normalize(cfgs.mean, [1, 1, 1])
    else:
        norm_method = Normalize(cfgs.mean, cfgs.std)

    ## train loader
    assert cfgs.train_crop in ['random', 'corner', 'center']
    if cfgs.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(cfgs.scales, cfgs.sample_size)
    elif cfgs.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(cfgs.scales, cfgs.sample_size)
    elif cfgs.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            cfgs.scales, cfgs.sample_size, crop_positions=['c'])
    spatial_transform = Compose([
        crop_method,
        ToTensor(cfgs.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(cfgs.sample_duration, cfgs.downsample)
    target_transform = ClassLabel()
    training_data = get_training_set(cfgs, spatial_transform,
                                        temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=cfgs.batch_size,
        shuffle=True,
        num_workers=cfgs.n_threads,
        pin_memory=True)

    ## validation loader
    spatial_transform = Compose([
        Scale(cfgs.sample_size),
        CenterCrop(cfgs.sample_size),
        ToTensor(cfgs.norm_value), norm_method
    ])
    temporal_transform = TemporalCenterCrop(cfgs.sample_duration, cfgs.downsample)
    target_transform = ClassLabel()
    validation_data = get_validation_set(
        cfgs, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=8,
        shuffle=False,
        num_workers=cfgs.n_threads,
        pin_memory=True)
    
    # optimizer & scheduler
    if cfgs.nesterov:
        dampening = 0
    else:
        dampening = cfgs.dampening
    optimizer = optim.SGD(
        parameters,
        lr=cfgs.learning_rate,
        momentum=cfgs.momentum,
        dampening=dampening,
        weight_decay=cfgs.weight_decay,
        nesterov=cfgs.nesterov)
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=cfgs.lr_patience)

    train(train_loader, val_loader, model, criterion, optimizer, cfgs)