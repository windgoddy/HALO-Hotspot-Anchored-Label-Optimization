# Pretrained Checkpoints

Place the downloaded checkpoint files in this `checkpoints/` folder.

## Download

The three final main-result checkpoints are provided via Baidu Netdisk:

- Share name: `HALO_weights_P1_D`
- Link: `https://pan.baidu.com/s/12MSYkBqD6t9E5UxvycSAkQ?pwd=HALO`
- Extraction code: `HALO`

After downloading, verify that the files match the locally evaluated weights by running `shasum -a 256 -c checkpoints/SHA256SUMS` from the repository root. The manifest contains hashes only; the weights themselves are ignored by Git.

## Included Main-Result Weights

| Dataset | Backbone | Image Folder | Mask Folder | Best Epoch | mIoU | BoxIoU | Expected File Name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NUAA-SIRST | DNANet | `images` | `masks_sweep_sigma1.500_expand1.5` | 417 | 0.7353 | 0.7089 | `best_model_epoch0417_mIoU0.7353_BoxIoU0.7089.pth.tar` |
| NUDT-SIRST | DNANet | `images` | `masks_sweep_sigma1.500_expand1.5` | 440 | 0.6749 | 0.8307 | `best_model_epoch0440_mIoU0.6749_BoxIoU0.8307.pth.tar` |
| IRSTD-1k | DNANet | `IRSTD1k_Img` | `masks_sweep_sigma1.500_expand1.5` | 273 | 0.6433 | 0.6243 | `best_model_epoch0273_mIoU0.6433_BoxIoU0.6277.pth.tar` |

The IRSTD-1k checkpoint filename contains `BoxIoU0.6277`, but the metric stored inside that checkpoint and reproduced by `test.py` is approximately 0.6243. The filename is retained so the released file can be found. BoxIoU is an auxiliary checkpoint metric, separate from the manuscript's main mIoU, Pd, and Fa results.

## Training Arguments

The argument records of the training runs that produced the three released checkpoints are listed below. In these records, `split_method: 50_50` is the name of the directory that held the split files during training; its contents correspond to the files in `splits/` (NUAA-SIRST 224/96, NUDT-SIRST 663/664, IRSTD-1k 800/201). The `root` and `use_lidar_dataloader` entries come from the original development code base and are not used by this repository's scripts.

- [halo_dnanet_nuaa.args.txt](halo_dnanet_nuaa.args.txt)
- [halo_dnanet_nudt.args.txt](halo_dnanet_nudt.args.txt)
- [halo_dnanet_irstd.args.txt](halo_dnanet_irstd.args.txt)

## Pseudo Label Settings

These uploaded checkpoints correspond to the final pseudo-label setting:

- label folder: `labels_box`
- mask folder: `masks_sweep_sigma1.500_expand1.5`
- `expand_ratio = 1.5`
- `gauss_sigma_ratio = 1.5`
- `temperature = 3.0`
- `fg_threshold = 0.0`
- `spatial_gaussian = True`

Generate the pseudo labels with:

```bash
bash scripts/prepare_labels.sh ./data
```

## Test Commands

After downloading the checkpoints into `HALO/checkpoints/`, run:

```bash
python test.py \
  --dataset NUAA-SIRST \
  --root ./data \
  --model DNANet \
  --image_folder images \
  --test_batch_size 4 \
  --checkpoint ./checkpoints/best_model_epoch0417_mIoU0.7353_BoxIoU0.7089.pth.tar
```

```bash
python test.py \
  --dataset NUDT-SIRST \
  --root ./data \
  --model DNANet \
  --image_folder images \
  --test_batch_size 4 \
  --checkpoint ./checkpoints/best_model_epoch0440_mIoU0.6749_BoxIoU0.8307.pth.tar
```

```bash
python test.py \
  --dataset IRSTD-1k \
  --root ./data \
  --model DNANet \
  --image_folder IRSTD1k_Img \
  --test_batch_size 4 \
  --checkpoint ./checkpoints/best_model_epoch0273_mIoU0.6433_BoxIoU0.6277.pth.tar
```

Or evaluate all three:

```bash
bash scripts/eval_all.sh ./data
```
