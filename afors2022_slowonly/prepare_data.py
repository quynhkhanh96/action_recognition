import argparse
import os
import glob
import shutil 
from tqdm import tqdm 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Restructure AFORS dataset')
    parser.add_argument(
        "--data_root",
        type=str,
        help="Where original data are stored",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="restructured data dir",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    NUM_CLASSES = 12 
    all_files = glob.glob(args.data_root + '/data/*/*/*.mp4')
    class2path = {f'{i:02d}': [] for i in range(1, NUM_CLASSES + 1)}
    for fpath in all_files:
        fname = fpath.split('/')[-1].split('.')[0]
        class2path[fname].append(fpath)

    for class_ in class2path:
        os.makedirs(args.output_dir + f'/{class_}', exist_ok=True)
        print(class_)
        class_fpaths = class2path[class_]
        for fpath in tqdm(class_fpaths, total=len(class_fpaths)):
            person = fpath.split('/')[-3]
            record_time = fpath.split('/')[-2]
            new_path = args.output_dir + f'/{class_}/{person}__{record_time}__{class_}.mp4'
            shutil.copy(fpath, new_path)