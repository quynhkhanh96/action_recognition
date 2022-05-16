import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.video_sampler import *

# __all__ = ['AFOSRVideoDataset']
class AFOSRVideoDataset(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 ):
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.clips = []
        self.labels = []

        with open(self.annotation_file_path) as f:
            subject_dirs = [_.strip() for _ in f.readlines()]
            for subject_dir in subject_dirs:
                subject_dir_path = os.path.join(self.video_dir, subject_dir)
                for timestamp_dir in sorted(os.listdir(subject_dir_path)):
                    timestamp_dir_path = os.path.join(subject_dir_path, timestamp_dir)
                    for video_file in filter(lambda _: _.endswith('.mp4'),
                                             sorted(os.listdir(timestamp_dir_path))):
                        label = int(os.path.splitext(video_file)[0]) - 1
                        video_file = os.path.join(subject_dir, timestamp_dir, video_file)
                        self.clips.append((video_file, subject_dir))
                        self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video_file, subject = self.clips[item]
        video_file = os.path.join(self.video_dir, video_file)
        frames = self.sampler(video_file, sample_id=item)        
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) s for seq_len, c for channel
        return data, self.labels[item], video_file   

class AFOSRFrameDataset(Dataset):
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

        self.clips = []
        self.labels = []

        with open(self.annotation_file_path) as f:
            subject_dirs = [_.strip() for _ in f.readlines()]

            for subject_dir in subject_dirs:
                subject_dir_path = os.path.join(self.frame_dir, subject_dir)

                for timestamp_dir in sorted(os.listdir(subject_dir_path)):
                    timestamp_dir_path = os.path.join(subject_dir_path, timestamp_dir)

                    for video_file in sorted(os.listdir(timestamp_dir_path)):
                        label = int(os.path.splitext(video_file)[0]) - 1
                        video_file = os.path.join(subject_dir, timestamp_dir, video_file)
                        self.clips.append((video_file, subject_dir))
                        self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def sample_frames(self, video_file):
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

    def __getitem__(self, item):
        video_file, _ = self.clips[item]
        video_file = os.path.join(self.frame_dir, video_file)

        """ Sample video frames """
        frame_names = self.sample_frames(video_file)
        frames = []
        for frame_name in frame_names:
            frame = cv2.imread(video_file + '/' + frame_name)
            frames.append(frame)

        """ Transform and augment RGB images"""
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) where s is seq_len, c is number of channels
        return data, self.labels[item], video_file  