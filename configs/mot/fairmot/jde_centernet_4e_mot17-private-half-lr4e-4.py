model = dict(
    type='FairMOT',
    detector=dict(
        type='CenterNet',
        backbone=dict(
            type='DLASeg',
            base_name='dla34',
            # heads=dict(hm=1, wh=4, id=128, reg=2),
            heads=None,
            pretrained=True,
            down_ratio=4,
            final_kernel=1,
            last_level=5,
            head_conv=256,
            out_channel=0),
        neck=None,
        # neck=dict(
        #     type='CTResNetNeck',
        #     in_channel=512,
        #     num_deconv_filters=(256, 128, 64),
        #     num_deconv_kernels=(4, 4, 4),
        #     use_dcn=True),
        bbox_head=dict(
            type='CenterNetHead',
            num_classes=1,
            in_channel=64,
            feat_channel=256,
            loss_center_heatmap=dict(
                type='GaussianFocalLoss', loss_weight=1.0),
            loss_wh=dict(type='L1Loss', loss_weight=0.1),
            loss_offset=dict(type='L1Loss', loss_weight=1.0)),
        train_cfg=None,
        test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100)),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='ConvReIDHead',
        in_channel=64,
        feat_channel=256,
        out_channel=128,
        num_classes=359,
        loss=dict(type='CrossEntropyLoss', loss_weight=0.1)),
    tracker=dict(
        type='JDETracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadFairMOTAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Resize', img_scale=(1088, 608), keep_ratio=True),
    dict(type='Pad', size=(608, 1088)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FairMOTFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

data_root = 'data/MOT17/'
dataset_type = 'MOTChallengeDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/half-train_cocoformat.json',
        ann_file=
        '/mnt/lustre/shensanjing/data/MOT17/annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        classes=('pedestrian', ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        classes=('pedestrian', ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        classes=('pedestrian', ),
        pipeline=test_pipeline))

optimizer = dict(type='Adam', lr=0.0004, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[20])
# runtime settings
total_epochs = 30
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

checkpoint_config = dict(interval=1)
# load_from = 'ckpts/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'  # noqa: E501
load_from = None

optimizer_config = dict(grad_clip=None)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
