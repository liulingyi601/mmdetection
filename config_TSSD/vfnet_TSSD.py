_base_ = [
    './TSSD.py',
    './_base_/default_runtime.py'
]
# model settings
INF = 1e8

model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        # regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
        regress_ranges=((-1, 8), (8, 16), (16, 32), (32, 64), (64, INF)),

        # strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16, 32, 64],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=2,
            scales_per_octave=1,
            center_offset=0.0,
            strides=[4, 8, 16, 32, 64,]),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
find_unused_parameters=True
img_norm_cfg = dict(
    mean=[46.173172, 46.173172, 46.173172], std=[40.773808, 40.773808, 40.773808], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(576, 576), (448,448)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5,direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=[4, 4]),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    train=dict(pipeline=train_pipeline))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
#
runner=dict(type='IterBasedRunner', max_iters=72000)
# checkpoint_config = dict(interval=12)
checkpoint_config = dict(interval=3000)
auto_resume=True
fp16 = dict(loss_scale=512.)
evaluation=dict(interval=3000)