_base_ = [
    './TSSD.py',
    '../_base_/default_runtime.py'
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
        type='BGMSRefineHead',
        auto_weighted_loss=True,
        sample_weight=True,
        use_pos=True,
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        bbox_norm_type='stride',
        # regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
        regress_ranges=((-1, 32),(32, 64), (64, INF)),
        # strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16],
        center_sampling=False,
        dcn_on_last_conv=False,
        # use_atss=True,
        use_vfl=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=2,
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
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
find_unused_parameters=True
# resume_from = '/data/data1/lxp/open-mmlab/mmdetection/work_dirs/crfsdet_r50_c1/latest.pth'
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     gamma=0.3,
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[5, 9, 12, 14])
# runner = dict(type='EpochBasedRunner', max_epochs=15)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
#
runner=dict(type='IterBasedRunner', max_iters=54000)
# checkpoint_config = dict(interval=12)
checkpoint_config = dict(interval=2400)

evaluation=dict(interval=2400)