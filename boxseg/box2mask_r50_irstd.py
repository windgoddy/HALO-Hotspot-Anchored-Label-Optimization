# Box2Mask (level-set, Mask2Former-style) adapted for IRSTD (single class, tiny targets).
# Self-contained (no _base_) so it can be referenced by absolute path.
# Evaluates HALO-style mIoU/Pd/Fa every epoch via IRSTDCocoDataset.
# Launch:  cd BoxInstSeg && PYTHONPATH=<HALO>/boxseg python tools/train.py <this> --work-dir ...
#
# Set HALO_DATA_ROOT to the directory containing the benchmark dataset folders.
import os
dataset_name = os.environ.get('HALO_DATASET', 'NUAA-SIRST')
data_root = os.path.join(os.environ.get('HALO_DATA_ROOT', 'data'), dataset_name) + '/'
img_prefix = data_root + ('IRSTD1k_Img/' if dataset_name == 'IRSTD-1k' else 'images/')
ann_train  = data_root + 'coco/instances_train.json'
ann_test   = data_root + 'coco/instances_test.json'
classes = ('target',)

custom_imports = dict(imports=['irstd_coco_dataset'], allow_failed_imports=False)

num_thing_classes = 1          # IRSTD: single class
num_stuff_classes = 0
num_classes = num_thing_classes + num_stuff_classes

model = dict(
    type='Box2Mask',
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='https://download.pytorch.org/models/resnet50-11ad3fa6.pth')),
    panoptic_head=dict(
        type='Box2MaskHead',
        in_channels=[256, 512, 1024, 2048], strides=[4, 8, 16, 32],
        feat_channels=256, out_channels=256,
        num_things_classes=num_thing_classes, num_stuff_classes=num_stuff_classes,
        num_queries=100, num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder', num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32), act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder', num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256,
                        num_heads=8, num_levels=3, num_points=4, im2col_step=64,
                        dropout=0.0, batch_first=False, norm_cfg=None, init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=256, feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder', return_intermediate=True, num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(type='MultiheadAttention', embed_dims=256, num_heads=8,
                               attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=False),
                ffn_cfgs=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2,
                              act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0,
                              dropout_layer=None, add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
                      reduction='mean', class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(type='LevelsetLoss', loss_weight=1.0),
        loss_box=dict(type='BoxProjectionLoss', loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead', num_things_classes=num_thing_classes,
        num_stuff_classes=num_stuff_classes, loss_panoptic=None, init_cfg=None),
    train_cfg=dict(
        assigner=dict(type='MaskHungarianAssigner',
                      cls_cost=dict(type='ClassificationCost', weight=2.0),
                      dice_cost=dict(type='BoxMatchingCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(panoptic_on=False, semantic_on=False, instance_on=True,
                  max_per_image=100, iou_thr=0.8, filter_low_score=True),
    init_cfg=None)

# ---------------- data ----------------
image_size = (1024, 1024)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='GenerateBoxMask'),                       # box supervision: mask from box
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=image_size, ratio_range=(0.5, 2.0),  # softened LSJ (was 0.1-2.0; protect tiny targets)
         multiscale_mode='range', keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), keep_empty=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=image_size, flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,  # Box2Mask (transformer + 1024) is memory-heavy
    train=dict(type='CocoDataset', classes=classes, ann_file=ann_train,
               img_prefix=img_prefix, pipeline=train_pipeline),
    val=dict(type='IRSTDCocoDataset', classes=classes, ann_file=ann_test,
             img_prefix=img_prefix, pipeline=test_pipeline),
    test=dict(type='IRSTDCocoDataset', classes=classes, ann_file=ann_test,
              img_prefix=img_prefix, pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['segm'], save_best='mIoU', rule='greater')

# ---------------- optimizer & schedule ----------------
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0),
                     'query_embed': embed_multi, 'query_feat': embed_multi,
                     'level_embed': embed_multi},
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(policy='step', gamma=0.1, by_epoch=True, step=[280, 360],
                 warmup='linear', warmup_iters=1000, warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=400)

# ---------------- runtime ----------------
checkpoint_config = dict(interval=50, save_last=True, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
work_dir = './work_dirs/box2mask_r50_irstd_nuaa'
