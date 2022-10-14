'''
    Dataloader:
        Train Dataset: RawframeDataset(
            ann_file,
            data_prefix,
            pipeline=[
                SampleFrames(clip_len=8, frame_interval=4,
                num_clips=1),
                RawFrameDecode(),
                Resize(-1, 256),
                RandomResizedCrop()
                Resize(scale=(224, 224)),
                Flip(0.5),
                Normalize(),
                FormatShape(input_format='NCTHW'),
                ToTensor()
            ]
        )
'''