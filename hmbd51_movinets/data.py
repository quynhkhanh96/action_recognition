import os 
import random 
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import augments as T

# HMDB dataset
def _data_transforms_hmdb51():
    train_transform = transforms.Compose([  
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.RandomHorizontalFlip(),
        T.RandomCrop((172, 172))
    ])

    valid_transform = transforms.Compose([                           
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.CenterCrop((172, 172))
    ])
    return train_transform, valid_transform

def get_hmdb51_loaders(fold, video_data_dir, splits_dir,
                            num_frames, frame_rate, clip_steps,
                            train_bz, test_bz):

    transform_train, transform_test = _data_transforms_hmdb51()

    hmdb51_train = HMDB51(video_data_dir, splits_dir, 
                            num_frames, frame_rate=frame_rate,
                            step_between_clips=clip_steps, 
                            fold=fold, train=True,
                            transform=transform_train, num_workers=2)

    hmdb51_test = HMDB51(video_data_dir, splits_dir, 
                            num_frames, frame_rate=frame_rate,
                            step_between_clips=clip_steps, 
                            fold=fold, train=False,
                            transform=transform_test, num_workers=2)

    train_loader = DataLoader(hmdb51_train, batch_size=train_bz, shuffle=True)
    test_loader  = DataLoader(hmdb51_test, batch_size=test_bz, shuffle=False)

    hmdb51_classes = hmdb51_train.classes
    num_classes = len(hmdb51_classes)

    return train_loader, test_loader, num_classes