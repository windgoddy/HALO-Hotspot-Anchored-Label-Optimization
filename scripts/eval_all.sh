#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-./data}"

python test.py --dataset NUAA-SIRST --root "$ROOT" --model DNANet --image_folder images --test_batch_size 4 --checkpoint ./checkpoints/best_model_epoch0417_mIoU0.7353_BoxIoU0.7089.pth.tar
python test.py --dataset NUDT-SIRST --root "$ROOT" --model DNANet --image_folder images --test_batch_size 4 --checkpoint ./checkpoints/best_model_epoch0440_mIoU0.6749_BoxIoU0.8307.pth.tar
python test.py --dataset IRSTD-1k --root "$ROOT" --model DNANet --image_folder IRSTD1k_Img --test_batch_size 4 --checkpoint ./checkpoints/best_model_epoch0273_mIoU0.6433_BoxIoU0.6277.pth.tar
