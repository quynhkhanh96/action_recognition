Centralized action recognition model training, experiments are conducted on standard machines like RTX 2080Ti or embedded devices like Jetson Nano.
Models are only trained on single GPU (hence `videos_per_gpu` is reduced to be able to fit in).
## HMDB51

## UCF101 

## TSM 

Set the paths:
```shell
MMACTION="/ext_data2/comvis/khanhdtq/mmaction2"
DATA_DIR="${MMACTION}/data/ucf101/rawframes"
WORK_DIR="/ext_data2/comvis/khanhdtq/ucf101_tsm"
SPLIT=1
ANN_FILE_TRAIN="${MMACTION}/data/ucf101/ucf101_train_split_${SPLIT}_rawframes.txt"
ANN_FILE_VAL="${MMACTION}/data/ucf101/ucf101_val_split_${SPLIT}_rawframes.txt"
```

Train model:
```shell
CUDA_VISIBLE_DEVICES=3 python -m train --data_root=$DATA_DIR --data_root_val=$DATA_DIR --work_dir=$WORK_DIR --ann_file_train=$ANN_FILE_TRAIN --ann_file_val=$ANN_FILE_VAL
```

Run inference and evaluate recognizer:
```shell
CUDA_VISIBLE_DEVICES=3 python -m inference --epoch=100
```

## AFORS
Restructure the dataset so we can use mmaction2's build rawframes script:
```shell
python prepare_data.py --data_root "/ext_data2/comvis/datasets/afors2022" --output_dir "/ext_data2/comvis/khanhdtq/mmaction2/data/afors/videos"
```
Build rawframes:
```shell 
MMACTION="/ext_data2/comvis/khanhdtq/mmaction2"
cd $MMACTION/tools/data 
python build_rawframes.py ../../data/afors/videos/ ../../data/afors/rawframes/ --task rgb --level 2 --ext mp4 --use-opencv
```
Copy `train.txt` and `val.txt` in `afors2022` directory to `$MMACTION/data/afors/annotations` then run this to build file list:
```shell 
cd ../..
PYTHONPATH=. python tools/data/build_file_list.py afors data/afors/rawframes/ --num-split 1 --level 2 --subset train --format rawframes --shuffle
PYTHONPATH=. python tools/data/build_file_list.py afors data/afors/rawframes/ --num-split 1 --level 2 --subset val --format rawframes --shuffle
```
Now train SlowOnly ResNet50 model by running:
```shell
cd afors2022_slowonly
MMACTION="/ext_data2/comvis/khanhdtq/mmaction2"
DATA_DIR="${MMACTION}/data/afors/rawframes"
WORK_DIR="/ext_data2/comvis/khanhdtq/afors_slowonly"
ANN_FILE_TRAIN="${MMACTION}/data/afors/afors_train_list_rawframes.txt"
ANN_FILE_VAL="${MMACTION}/data/afors/afors_val_list_rawframes.txt"

CUDA_VISIBLE_DEVICES=0 python -m train --data_root=$DATA_DIR --data_root_val=$DATA_DIR --work_dir=$WORK_DIR --ann_file_train=$ANN_FILE_TRAIN --ann_file_val=$ANN_FILE_VAL
```
Evaluate model:
```shell 
CUDA_VISIBLE_DEVICES=3 python -m inference --epoch=30
```

