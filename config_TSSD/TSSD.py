# dataset
dataset_type = 'CocoDataset'
classes = ('ship',)
data_root = './data/TSSD/'
# data_root = '/home/wxn078/dataset/HRSID/HRSID_jpg/'
img_norm_cfg = dict(
    mean=[46.173172, 46.173172, 46.173172], std=[40.773808, 40.773808, 40.773808], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=[(1333, 640), (1333, 960)], multiscale_mode='range', keep_ratio=True),
    dict(type='Resize', img_scale=(512,512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512,512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # TSSD SSDD
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        # HRSID
        # ann_file=data_root + 'annotations/train2017.json',
        # img_prefix=data_root + 'JPEGImages',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        # ann_file=data_root + 'annotations/test2017.json',
        # img_prefix=data_root + 'JPEGImages/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        # ann_file=data_root + 'annotations/test2017.json',
        # img_prefix=data_root + 'JPEGImages/',
        classes=classes,
        pipeline=test_pipeline))


