# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[50, 90, 120, 140])
runner = dict(type='EpochBasedRunner', max_epochs=150)
