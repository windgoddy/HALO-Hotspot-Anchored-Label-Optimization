# BoxInst (CondInst + box supervision) adapted for IRSTD (single class, tiny grayscale targets).
# Self-contained config (no _base_), so it can be referenced by absolute path from anywhere:
#   cd BoxInstSeg && python tools/train.py /path/HALO/boxseg/boxinst_r50_irstd.py
#
# Set HALO_DATA_ROOT to the directory containing the benchmark dataset folders.
import os
dataset_name = os.environ.get('HALO_DATASET', 'NUAA-SIRST')
data_root = os.path.join(os.environ.get('HALO_DATA_ROOT', 'data'), dataset_name) + '/'
img_prefix = data_root + ('IRSTD1k_Img/' if dataset_name == 'IRSTD-1k' else 'images/')
ann_train  = data_root + 'coco/instances_train.json'
ann_test   = data_root + 'coco/instances_test.json'
classes = ('target',)

# ---------------- model ----------------
model = dict(
    type='CondInst',
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256,
        start_level=1, add_extra_convs='on_output', num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CondInstBoxHead',
        num_classes=1,                    # <-- IRSTD: single class
        in_channels=256, center_sampling=True, center_sample_radius=3.0,  # ↑ for tiny targets
        norm_on_bbox=True, stacked_convs=4, feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    mask_branch=dict(
        type='CondInstMaskBranch', in_channels=256, in_indices=[0, 1, 2],
        strides=[8, 16, 32], branch_convs=4, branch_channels=128, branch_out_channels=16),
    mask_head=dict(
        type='CondInstMaskHead', in_channels=16, in_stride=8, out_stride=4,
        dynamic_convs=3, dynamic_channels=8, disable_rel_coors=False,
        bbox_head_channels=256,
        # ↓↓↓ TINY-TARGET TUNABLE: lowered from COCO [64,128,256,512,1024] so tiny IR
        #     targets are assigned to the finest FPN levels (must keep length 5).
        sizes_of_interest=[16, 32, 64, 128, 256],
        max_proposals=-1, topk_per_img=64,
        boxinst_enabled=True,
        bottom_pixels_removed=0,          # was 10; small IR images -> don't crop borders
        pairwise_size=3, pairwise_dilation=2,
        pairwise_color_thresh=0.3,        # BoxInst color-affinity; weak on grayscale IR (note in paper)
        pairwise_warmup=10000),
    train_cfg=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4,
                      min_pos_iou=0, ignore_iof_thr=-1),
        allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05,
                  nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=2000, output_segm=True))

# ---------------- data ----------------
# Register the custom dataset that evaluates HALO-style mIoU/Pd/Fa every epoch.
# Launch with:  PYTHONPATH=<HALO>/boxseg python tools/train.py ...
custom_imports = dict(imports=['irstd_coco_dataset'], allow_failed_imports=False)
dataset_type = 'CocoDataset'
# Grayscale IR PNGs are loaded as 3-channel (replicated) and normalized with ImageNet
# stats to match the torchvision ResNet-50 pretraining.
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # BOX supervision only
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),   # upscale tiny targets so they hit feature grid
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False,
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
    samples_per_gpu=4, workers_per_gpu=2,
    train=dict(type=dataset_type, classes=classes, ann_file=ann_train,
               img_prefix=img_prefix, pipeline=train_pipeline),
    val=dict(type='IRSTDCocoDataset', classes=classes, ann_file=ann_test,
             img_prefix=img_prefix, pipeline=test_pipeline),
    test=dict(type='IRSTDCocoDataset', classes=classes, ann_file=ann_test,
              img_prefix=img_prefix, pipeline=test_pipeline))
# HALO-style mIoU/Pd/Fa every epoch (computed inside IRSTDCocoDataset.evaluate)
evaluation = dict(interval=1, metric=['segm'], save_best='mIoU', rule='greater')

# ---------------- schedule (small dataset) ----------------
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)  # 1 GPU, small data
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))           # prevent NaN divergence
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000,
                 warmup_ratio=0.001, step=[280, 360])
runner = dict(type='EpochBasedRunner', max_epochs=400)

# ---------------- runtime ----------------
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/boxinst_r50_irstd_nuaa'
