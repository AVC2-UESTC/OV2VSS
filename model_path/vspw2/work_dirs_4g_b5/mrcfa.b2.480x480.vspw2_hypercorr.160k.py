norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/home/lixinhao/vss/mit_b2.pth',
    backbone=dict(type='mit_b2', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead_CAT',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_seen_classes=81,
        num_clips=5,
        hypercorre=True,
        backbone='b2'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CityscapesDataset_clips'
data_root = '/datadisk2/lixinhao/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(1024, 512),
        ratio_range=(0.5, 2.0),
        process_clips=True),
    dict(type='RandomCrop_clips', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(
        type='Normalize_clips',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad_clips', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, process_clips=True),
            dict(type='RandomFlip_clips'),
            dict(
                type='Normalize_clips',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type='CityscapesDataset_clips',
            data_root='/datadisk2/lixinhao/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 512),
                    ratio_range=(0.5, 2.0),
                    process_clips=True),
                dict(
                    type='RandomCrop_clips',
                    crop_size=(512, 1024),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip_clips', prob=0.5),
                dict(type='PhotoMetricDistortion_clips'),
                dict(
                    type='Normalize_clips',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad_clips',
                    size=(512, 1024),
                    pad_val=0,
                    seg_pad_val=255),
                dict(type='DefaultFormatBundle_clips'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            dilation=[-9, -6, -3],
            istraining=True)),
    val=dict(
        type='CityscapesDataset_clips',
        data_root='/datadisk2/lixinhao/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, process_clips=True),
                    dict(type='RandomFlip_clips'),
                    dict(
                        type='Normalize_clips',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor_clips', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        dilation=[-9, -6, -3],
        istraining=False),
    test=dict(
        type='CityscapesDataset_clips',
        data_root='/datadisk2/lixinhao/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, process_clips=True),
                    dict(type='RandomFlip_clips'),
                    dict(
                        type='Normalize_clips',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor_clips', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        dilation=[-9, -6, -3],
        istraining=False))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=3e-06,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=100000, metric='mIoU')
work_dir = 'model_path/vspw2/work_dirs_4g_b5/'
gpu_ids = range(0, 1)
