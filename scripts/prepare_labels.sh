#!/usr/bin/env bash

set -e
ROOT="${1:-./data}"

python label_gen.py --dataset NUAA-SIRST --root "$ROOT" --save_dir "$ROOT/NUAA-SIRST/masks_sweep_sigma1.500_expand1.5" --label_folder labels_box --expand_ratio 1.5 --gauss_sigma_ratio 1.5 --fg_threshold 0.0 --spatial_gaussian
python label_gen.py --dataset NUDT-SIRST --root "$ROOT" --save_dir "$ROOT/NUDT-SIRST/masks_sweep_sigma1.500_expand1.5" --label_folder labels_box --expand_ratio 1.5 --gauss_sigma_ratio 1.5 --fg_threshold 0.0 --spatial_gaussian
python label_gen.py --dataset IRSTD-1k --root "$ROOT" --save_dir "$ROOT/IRSTD-1k/masks_sweep_sigma1.500_expand1.5" --label_folder labels_box --expand_ratio 1.5 --gauss_sigma_ratio 1.5 --fg_threshold 0.0 --spatial_gaussian
