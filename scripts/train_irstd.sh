#!/usr/bin/env bash

ROOT="${1:-./data}"

python train.py \
  --model DNANet \
  --dataset IRSTD-1k \
  --root "$ROOT" \
  --image_folder IRSTD1k_Img \
  --mask_folder masks_sweep_sigma1.500_expand1.5 \
  --epochs 1500 \
  --lr 0.05 \
  --optimizer Adagrad \
  --scheduler CosineAnnealingLR \
  --min_lr 1e-5 \
  --deep_supervision True \
  --in_channels 1 \
  --train_batch_size 4 \
  --test_batch_size 4 \
  --thres 0.3 \
  --seed 42 \
  --experiment_name P1_D_pag_irstd_reproduce
