# Basic libs
from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

# pytorch libs
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms as T
import torch.optim as optim
# custom libs
import models
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint, plot_confusion_matrix, accuracy
from eval_metrics import evaluate

from datasets.utils.video_sampler import *

# visualize libs
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-dir', '--dataset-dir', type=str, default='/home/nhquan/datasets/afosr2022/data')
parser.add_argument('-d', '--dataset', type=str, default='afosr', help="micagestures, ixmas, wvu")
parser.add_argument('--height', type=int, default=224,help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224, help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=16, help="number of images to sample in a tracklet")
parser.add_argument('--train-annotation-file', required=True)
parser.add_argument('--test-annotation-file', required=True)
parser.add_argument('-j', '--workers', default=6, type=int,help="number of data loading workers (default: 4)")

# Optimization options
parser.add_argument('--max-epoch', default=30, type=int, help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=4, type=int, help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
	help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
	help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,	help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,	help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
	help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet503d', 
	help="c3d_bn, resnet503d, movinet_a0, movinet_a2, movinet_a5,mobilenet3d_v2")
parser.add_argument('--width-mult', default=1.0, type=float, help="using with mobilenet3d_v2: 0.2, 0.45, 0.7, 1.0")
# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='./log/best_model.pth.tar',
	help='need to be set for resnet3d models')
parser.add_argument('--eval-step', type=int, default=50,
	help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--save-frequency', type=int, default=5,
	help="save check point for every N epochs (set to -1 if not save)")

parser.add_argument('--gpu-devices', default='cuda:0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--job', type=str, default='train', help="train, evaluate, evaluateCross, extractFeat, extractTA, visMap")
args = parser.parse_args()

device = torch.device(args.gpu_devices if torch.cuda.is_available() else 'cpu')
use_gpu = True

def main():		
	torch.manual_seed(args.seed)
	checkpoint_file = os.path.join(args.save_dir, 'checkpoint.pt')
	best_model_file = os.path.join(args.save_dir, 'best_model.pth.tar')
	
	'''
		Write log
	'''
	if args.job == 'train':
		sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
	elif args.job == 'evaluate':
		sys.stdout = Logger(osp.join(args.save_dir, 'log_test_%s.txt' % (os.path.basename(args.test_annotation_file))))
	
	print("==========\nArgs:{}\n==========".format(args))

	if use_gpu:
		print("Currently using GPU {}".format(args.gpu_devices))
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)
	else:
		print("Currently using CPU (GPU is highly recommended)")

	'''
	Initial dataset
	'''	
	scaler = T.Resize(((args.height, args.width)))
	normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225])    
	transform= T.Compose([
		T.ToPILImage(),
		scaler,
		T.ToTensor(),
		normalize      
		])  	

	if args.dataset == "afosr":
		from datasets.afosr import AFOSRVideoDataset
		train_set = AFOSRVideoDataset(
			video_dir=args.dataset_dir,
			annotation_file_path=args.train_annotation_file,
			sampler=RandomTemporalSegmentSampler(n_frames=args.seq_len),
			# sampler=SystematicSampler(n_frames=16),
			transform=transform,
			use_albumentations=False,
		)
		test_set = AFOSRVideoDataset(
			video_dir=args.dataset_dir,
			annotation_file_path=args.test_annotation_file,
			sampler=SystematicSampler(n_frames=args.seq_len),
			transform=transform,
			use_albumentations=False,
		)

	print(f'[Preparing dataset] n_train_instances = {len(train_set)}, n_test_instances = {len(test_set)}')

	class_names = np.unique(train_set.labels).astype(str).tolist()
	train_loader = DataLoader(train_set, batch_size=args.train_batch, num_workers=args.workers, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=args.test_batch, num_workers=args.workers, shuffle=False,drop_last=True)
	num_classes = len(class_names)
	print('num_classes =', num_classes)

	'''
	Initialize network model
	'''
	print("Initializing model: {}".format(args.arch))
	if args.arch=='resnet503d':	
		from models import resnet3d
		model = resnet3d.resnet50(num_classes=num_classes, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)		
		if args.job=='train':
			if not os.path.exists(args.pretrained_model):
				raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
			print("Loading checkpoint from '{}'".format(args.pretrained_model))
			checkpoint = torch.load(args.pretrained_model)
			state_dict = {}
			for key in checkpoint['state_dict']:
				if 'fc' in key: continue
				state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
			model.load_state_dict(state_dict, strict=False)
	elif args.arch=='resnet183d':
		from models import resnet3d
		model = resnet3d.resnet18(num_classes=num_classes, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)		
		if args.job=='train':
			if not os.path.exists(args.pretrained_model):
				raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
			print("Loading checkpoint from '{}'".format(args.pretrained_model))
			checkpoint = torch.load(args.pretrained_model)
			state_dict = {}
			for key in checkpoint['state_dict']:
				if 'fc' in key: continue
				state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
			model.load_state_dict(state_dict, strict=False)	
	elif args.arch=='c3d_bn':
		from models import c3d
		model = c3d.c3d_bn(num_classes=num_classes, dropout=0.2)
		if args.job=='train':
			if not os.path.exists(args.pretrained_model):
				raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
			print("Loading checkpoint from '{}'".format(args.pretrained_model))
			checkpoint = torch.load(args.pretrained_model)			
			model.load_state_dict(checkpoint)			
	elif args.arch=='movinet_a0':
		from movinets import MoViNet
		from movinets.config import _C
		model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True)
		model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
	elif args.arch=='movinet_a2':
		from movinets import MoViNet
		from movinets.config import _C
		model = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True)
		model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
	elif args.arch=='movinet_a5':
		from movinets import MoViNet
		from movinets.config import _C
		model = MoViNet(_C.MODEL.MoViNetA5, causal = False, pretrained = True)
		model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
	elif args.arch=='efficientnet3d':
		from models.efficientnet_pytorch_3d import EfficientNet3D
		model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes}, in_channels=3)
	elif args.arch[:14]=='mobilenet3d_v2':
		from models import mobilenet3d_v2
		model = mobilenet3d_v2.mobilenet3d_v2(num_classes=num_classes,width_mult=args.width_mult)
		if not os.path.exists(args.pretrained_model):
			raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
		print("Loading checkpoint from '{}'".format(args.pretrained_model))
		checkpoint = torch.load(args.pretrained_model)
		model.load_state_dict(checkpoint,strict=False)
	elif args.arch=='slow_fast_r3d_18':
		from models.slow_fast_r2plus1d import slow_fast_r3d_18
		model = slow_fast_r3d_18(num_classes=num_classes,
                                  pretrained=False,
                                  progress=True,
                                  alpha=4,
                                  beta=8) # 64//beta -->8		
	else:
		model = models.init_model(name=args.arch, num_classes=num_classes, loss={'xent', 'htri'})
	
	# criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	if args.stepsize > 0:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
	start_epoch = args.start_epoch

	if use_gpu:
		# model = nn.DataParallel(model).to(device)	
		model.to(device)

	if args.job in ['evaluate', 'evaluateCross', 'extractFeat', 'extractTA', 'visMap']:
		print("Loading checkpoint from '{}'".format(args.pretrained_model))		
		print("using device:", device)
		checkpoint = torch.load(args.pretrained_model) 
		print(checkpoint['rank1'])     
		print(checkpoint['epoch'])  		
		# model.module.load_state_dict(checkpoint['state_dict'])
		model.load_state_dict(checkpoint['state_dict'])

	if args.job == 'evaluate':
		print("Evaluate top 10 with confusion matrix")			
		top1, top5, top10, infTime = testWithConfusionMatrix(model, test_loader, num_classes, device)
		print('top1: %.3f top5: %.3f top10: %.3f' % (top1, top5, top10))
		print("Average inference time: %6.3f" % (infTime))
		return		

	## Else training
	start_time = time.time()
	best_rank1 = -np.inf
	if args.arch == 'resnet503d':
		torch.backends.cudnn.benchmark = False	

	for epoch in range(start_epoch, args.max_epoch):
		print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))		
		train(model, criterion, optimizer, train_loader, device)	

		if args.stepsize > 0: scheduler.step()	

		if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
			print("==> Test")			
			rank1 = test(model, test_loader, device) # Must do not testing when training
			print('acc:%f' % (rank1))	
			is_best = rank1 > best_rank1
			if is_best: best_rank1 = rank1

			state_dict = model.state_dict()
			if is_best:	
				save_checkpoint({
					'state_dict': state_dict,
					'rank1': rank1,
					'epoch': epoch,
				}, False, osp.join(args.save_dir, 'best_model.pth.tar'))	

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	print("Finished. Total elapsed time (h:m:s): {}, the best rank1 {}".format(elapsed,best_rank1))


def train(model, criterion, optimizer, trainloader, device):
	model.train()
	losses = AverageMeter("loss")	
	for batch_idx, (imgs, labels, _) in enumerate(trainloader):			
		imgs, labels = imgs.to(device), labels.to(device)
		imgs, labels = Variable(imgs), Variable(labels)		
		outputs = model(imgs) 		
		loss = criterion(outputs, labels)	        
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()		 		
		losses.update(loss.item(), labels.size(0))
		if (batch_idx + 1) % args.print_freq == 0:
			print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

def test(model, testLoader, device, ranks=[1, 5, 10, 15, 20]):
	model.eval()
	total = 0
	correct = 0
	for batch_idx, (imgs, labels,video_paths) in enumerate(testLoader):	
		# imgs: (B, C, T, H, W)		
		imgs, labels = imgs.to(device), labels.to(device)
		with torch.no_grad():
			labels = Variable(labels)			
			imgs = Variable(imgs)			
			outputs = model(imgs)			
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)		
		correct += (predicted == labels).sum().item()
		# if wrong match
		if args.job=='evaluate':		
			wrongs = np.where((predicted != labels).cpu().numpy()==True)[0]		
			for it in wrongs:
				print("%s\t%d" % (video_paths[it], predicted[it] + 1))
	return (correct/total)

def testWithConfusionMatrix(model,testLoader,nb_classes,device):	 
	model.eval()
	cm = torch.zeros(nb_classes, nb_classes)	
	
	top1 = AverageMeter('Acc@1', ':6.3f')
	top5 = AverageMeter('Acc@5', ':6.3f')
	top10 = AverageMeter('Acc@10', ':6.3f')
	infTime = AverageMeter('infTimr', ':6.3f')

	with torch.no_grad():	
		for batch_idx, (imgs, labels,_) in enumerate(testLoader):		
			start=time.time()
			imgs = imgs.to(device)
			labels = labels.to(device)	
			outputs = model(imgs)   			
			acc1, acc5,acc10 = accuracy(outputs, labels, topk=(1, 5, 10))			
			top1.update(acc1[0], imgs.size(0))
			top5.update(acc5[0], imgs.size(0))
			top10.update(acc10[0], imgs.size(0))			

			_, preds = torch.max(outputs.data, 1)

			print("inferance time: %6.3f" % (time.time() - start))
			infTime.update(time.time() - start, imgs.size(0))

			for t, p in zip(labels.view(-1), preds.view(-1)):
				cm[t.long(), p.long()] += 1
			
	#### If Display count fmt='d'
	cm = cm.numpy()
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	np.savetxt(os.path.join(args.save_dir,  "confusion.csv"), cm, delimiter=",", fmt='%1.3f') 
	fmt = '.2f'
	class_name = ['1', '2 ', '3', '4','5', '6', '7', '8', '9', '10', '11', '12']		
	plot_confusion_matrix(cm, class_names=class_name, save_file=os.path.join(args.save_dir,"confusion.pdf"), fmt='.2f')
	return top1.avg, top5.avg, top10.avg, infTime.avg

if __name__ == '__main__':
	main()
