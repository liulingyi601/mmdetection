_base_ = [
    './TSSD.py',
    './_base_/default_runtime.py'
]
INF = 1e8
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

# model settings
model = dict(
    type='VFNet',
    backbone=dict(  
        # _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6,2],
        num_heads=[3, 6, 12,24],
        strides=(4, 2, 2,2),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=3),
    bbox_head=dict(
        type='BGMSCRefineHead',
        cdf_conv=dict(num_heads=1, num_samples=5, use_pos=True, kernel_size=1),
        auto_weighted_loss=True,
        sample_weight=True,
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        post_stacked_convs=1,
        feat_channels=256,
        # regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
        regress_ranges=((-1, 32),(32, 64), (64, INF)),
        # strides=[8, 16, 32, 64, 128],
        bbox_norm_type='reg_denom',
        reg_denoms=[8,16,32],

        strides=[4, 8, 16],
        center_sampling=False,
        dcn_on_last_conv=False,
        # use_atss=True,
        use_vfl=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=4,
            scales_per_octave=1,
            center_offset=0.0,
            # strides=[8, 16, 32, 64, 128]),
            strides=[4, 8, 16]),
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
# resume_from = '/data/data1/lxp/open-mmlab/mmdetection/work_dirs/crfsdet_r50_c1/latest.pth'
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
data_root = './data/HRSID/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train2017.json',
               img_prefix=data_root + 'JPEGImages/',
               pipeline=train_pipeline),
    val=dict(ann_file=data_root + 'annotations/test2017.json',
               img_prefix=data_root + 'JPEGImages/'),
    test=dict(ann_file=data_root + 'annotations/test2017.json',
               img_prefix=data_root + 'JPEGImages/'))
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
runner=dict(type='IterBasedRunner', max_iters=30000)
# checkpoint_config = dict(interval=12)
checkpoint_config = dict(interval=3000)
auto_resume=True
fp16 = dict(loss_scale=512.)
evaluation=dict(interval=3000)