_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    # pretrained = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    pretrained='/home/lixinhao/vss/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),
        # type='ResNet',
        # depth=101),
    decode_head=dict(
        type='SegFormerHead_CAT',
        in_channels=[64, 128, 320, 512],
        # in_channels=[256,512,1024,2048],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        num_seen_classes=81,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # num_clips=4,
        num_clips=5,
        hypercorre=True,
        backbone='b2'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(_delete_=True, type='AdamW', lr=0.000003 , betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                #  warmup_ratio=1e-8,
                 power=0.9, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=1,workers_per_gpu=4)
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)
# evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=100000, metric='mIoU')