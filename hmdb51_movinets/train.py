from models import MoViNet
from config import _C
from data import get_hmdb51_loaders
from utils import * 

import torch 
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse

torch.manual_seed(97)

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data,_ , target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())
 
def evaluate(model, data_load, loss_val):
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HMDB51 MoViNets")
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='', 
        help="image directory",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        help="Where the split is saved",
    )
    args = parser.parse_args()

    # Configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # data loaders
    train_loader, test_loader, num_classes = get_hmdb51_loaders(fold=cfgs.fold,
            video_data_dir=args.data_dir, splits_dir=args.splits_dir,
            num_frames=cfgs.num_frames, frame_rate=cfgs.frame_rate, clip_steps=cfgs.clip_steps,
            train_bz=cfgs.batch_size, test_bz=cfgs.batch_size)

    # model
    model = MoViNet(_C.MODEL.MoViNetA0, causal=True, 
                    pretrained=True , num_classes=num_classes)

    trloss_val, tsloss_val = [], []
    # model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))
    optimz = optim.Adam(model.parameters(), lr=cfgs.lr)
    for epoch in range(1, cfgs.epochs + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, train_loader, trloss_val)
        evaluate(model, test_loader, tsloss_val)
