import os
import shutil
from collections import defaultdict 
import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.optim as optim

class FrameDataset(Dataset):
    def __init__(self, frame_dir,
                    annotation_file_path,
                    n_frames,
                    mode='train',
                    to_rgb=True,
                    transform=None,
                    use_albumentations=False):
        self.frame_dir = frame_dir
        self.annotation_file_path = annotation_file_path
        self.n_frames = n_frames

        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.mode = mode 

        with open(self.annotation_file_path) as f:
            lines = [l.strip() for l in f.readlines()]
            self.clips = [l.strip().split(' ')[0] for l in lines]
            self.labels = [int(l.strip().split(' ')[1]) for l in lines]

    def __len__(self):
        return len(self.clips)

    def sample_frames(self, video_file):
        '''
            Inside each /<video_id> directory, there are images named `frame_<id>.jpg`,
            for eg. `frame_000.jpg`. This function samples a subsequence of length 
            `self.n_frames` of them.
        '''
        all_frames = np.array(sorted(os.listdir(video_file), 
                            key=lambda x: int(x.split('_')[1].split('.')[0])))

        start_frame, end_frame = 0, len(all_frames) - 1
        if self.mode == 'train':
            segments = np.linspace(start_frame, end_frame, self.n_frames + 1)
            segment_length = (end_frame - start_frame) / self.n_frames
            sampled_frame_ids = segments[:-1] + np.random.rand(self.n_frames) * segment_length
        else:
            sampled_frame_ids = np.linspace(start_frame, end_frame, self.n_frames)

        frames = all_frames[sampled_frame_ids.round().astype(np.int64)]
        return frames
    
    def __getitem__(self, idx):
        video_id = self.clips[idx]
        video_file = os.path.join(self.frame_dir, video_id)

        ''' Sample video frames '''
        frame_names = self.sample_frames(video_file)
        frames = []
        for frame_name in frame_names:
            frame = cv2.imread(video_file + '/' + frame_name)
            frames.append(frame)

        ''' Transform and augment RGB images '''
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) where s is seq_len, c is number of channels
        return data, self.labels[idx], video_file 
    
def get_loaders(data_dir, cfgs):

    scaler = T.Resize(((cfgs.height, cfgs.width)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    transform= T.Compose([T.ToPILImage(), scaler, T.ToTensor(), normalize])  
    
    train_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=data_dir + '/train.txt',
        n_frames=cfgs.seq_len,
        mode='train',
        transform=transform,
        use_albumentations=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfgs.train_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=True
    )

    val_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=data_dir + '/val.txt',
        n_frames=cfgs.seq_len,
        mode='test',
        transform=transform,
        use_albumentations=False,
    )
    val_loader = DataLoader(val_set, batch_size=cfgs.test_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=False
    )

    return train_loader, val_loader