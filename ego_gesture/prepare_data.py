import os 
import shutil
import numpy as np 
import pandas as pd
import argparse

NUM_CLASSES = 83
NUM_SUBJECTS = 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get ExoGesture dataset ready')
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the orginal images directory",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        help="Path to the annotation directory",
    )
    parser.add_argument(
        "--new_data_dir",
        type=str,
        help="Path to preprocessed data, in mmaction format",
    )
    args = parser.parse_args()
    os.makedirs(args.new_data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.new_data_dir, 'rawframes'), exist_ok=True)

    df_train = pd.read_csv(os.path.join(args.annotation_dir, 
                                        'train_no84.csv'))
    df_test = pd.read_csv(os.path.join(args.annotation_dir, 
                                        'test_no84.csv'))
    df_train.columns = ['frame', 'label']
    df_test.columns = ['frame', 'label']

    for mode, df in [('train', df_train), ('test', df_test)]:
        df['subject'] = df.apply(lambda row: row.frame.split('/')[0], 
                                    axis=1)

        for cls in range(1, NUM_CLASSES+1):
            df_cls = df[df['label'] == cls]
            action_dir = os.path.join(args.new_data_dir, 'rawframes', f'action_{cls}')
            os.makedirs(action_dir, exist_ok=True)
            # Need to check: One subject performs a action exactly one time?
            for subject in range(1, NUM_SUBJECTS+1):
                df_frames = df_cls[df_cls['subject'] == f'Subject{subject}']
                if len(df_frames):
                    # frames
                    frames = df_frames['frame'].tolist()
                    frames.sort()

                    # video name
                    org_name = '/'.join(frames[0].split('/')[:-1])
                    video_name = '_'.join(frames[0].split('/')[:-1])
                    video_dir = os.path.join(action_dir, video_name)
                    os.makedirs(video_dir, exist_ok=True)
                    for i, frame in enumerate(frames):
                        frame_name = frame.split('/')[-1]
                        shutil.copy(os.path.join(args.image_dir, frame), 
                                os.path.join(video_dir, f'{i+1}:05.jpg'))
                    
                    num_frames = len(frames)
                    with open(os.path.join(args.new_data_dir, f'{mode}.txt'), 'a') as f:
                        f.write(f'action_{cls}/video_name {num_frames} {cls}\n')
        print(f'Done with action {cls}.')