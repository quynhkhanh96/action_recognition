# **Introduction**
Train EgoGesture dataset as in this [paper](arxiv.org/abs/1901.10323), the code is mainly borrowed from [here](https://github.com/ahmetgunduz/Real-time-GesRec). However in our experiments, we just train the classifier.
# **Data preparation**
We need video data (which is RGB and depth frame images of each video) and annotation. Our data directory will be in the following structure:
```shell
    realtime_EgoGesture
    ├── annotation_EgoGesture
    │   ├── egogestureall.json
    │   ├── egogestureall_but_None.json
    │   └── egogesturebinary.json
    └── images
        ├── Subject01
        │   ├── Scene1
        │   │   ├── Color
        │   │   └── Depth
        │   ├── Scene2
        │   └── Scene6
        ├── Subject02
        └── Subject50
```
Start by looking at `opts.py`, our `root_path` will be `path/to/realtime_EgoGesture`, `video_path` will be `path/to/realtime_EgoGesture/images` and `annotation_path` will be `path/to/realtime_EgoGesture/annotation_EgoGesture/egogestureall_but_None.json` (we don't include No Gesture (`None`) class in our training).

Also download the pretrained models from [here](https://drive.google.com/file/d/11MJWXmFnx9shbVtsaP1V8ak_kADg0r7D/view)
# **Experiments**
```shell
bash scripts/run_train.sh $ROOT_PATH $PRETRAINED_PATH/jester_resnext_101_RGB_32.pth resnext 1.0
```
# **Results**