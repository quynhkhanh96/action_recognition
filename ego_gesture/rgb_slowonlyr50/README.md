# **Introduction**
Data folder structure:
```
images
├── Subject01
│   ├── Scene1
│   │   ├── Color
│   │   │   ├── rgb1
│   │   │   │   ├── 000001.jpg
│   │   │   │   ├── 000002.jpg
│   │   │   │   ├── ...
│   │   │   │   └── 001316.jpg
│   │   │   ├── rgb2
│   │   │   ├── ...
│   │   │   └── rgb8
│   │   └── Depth
│   ├── Scene2
│   ├── ...
│   └── Scene6
├── Subject02
├── ...
└── Subject50
```
Annotations files: `train_no84.csv`, `test_no84.csv`.
```
Subject03/Scene1/Color/rgb1/000067.jpg,7
Subject03/Scene1/Color/rgb1/000068.jpg,7
Subject03/Scene1/Color/rgb1/000069.jpg,7
Subject03/Scene1/Color/rgb1/000070.jpg,7
Subject03/Scene1/Color/rgb1/000071.jpg,7
```
# **Data preparation**
Convert the original data to mmaction format:
- Data folder:
    ```
    rawframes
    ├── action_1
    │   ├── Subject03_Scene1_rgb1_0
    │   │   ├── img_00001.jpg
    │   │   ├── img_00002.jpg
    │   │   └── ...
    │   ├── Subject04_Scene1_rgb1_0
    │   │   ├── img_00001.jpg
    │   │   ├── img_00002.jpg
    │   │   └── ...
    │   └── Subject04_Scene1_rgb1_0
    ├── action_2
    ├── ...
    └── action_83
    ```
- Annotation file: Each line consists of `video_id`, `#frames` and `label` (0-indexed).
    ```
    action_1/Subject03_Scene1_rgb1_0 52 0
    action_1/Subject04_Scene1_rgb1_0 33 0
    ```
Run the following:
```shell
python prepare_data.py --image_dir=$IMAGE_DIR --annotation_dir=$ANN_DIR --new_data_dir=$DATA_DIR
```
# **Experiments**
Run this to train the model:
```shell
python train.py --data_dir=$DATA_DIR --work_dir=$DATA_DIR/cent_exps
```
# **Results**
Top1 accuracy = 81.4%, Top5 accuracy = 96.7%