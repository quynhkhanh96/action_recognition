Centralized action recognition model training, experiments are conducted on standard machines like RTX 2080Ti or embedded devices like Jetson Nano.

## HMDB51

## UCF101 

### TSM 

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


