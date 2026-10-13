#!/usr/bin/env bash
# Generate YOLO-format box annotations from GT segmentation masks for all three datasets.
# Usage:  bash scripts/gen_box_labels.sh [DATA_ROOT]
# DATA_ROOT defaults to ./data

set -e
ROOT="${1:-./data}"

python gen_box_labels.py \
  --dataset NUAA-SIRST NUDT-SIRST IRSTD-1k \
  --root "$ROOT" \
  --mask_folder masks \
  --output_folder labels_box
