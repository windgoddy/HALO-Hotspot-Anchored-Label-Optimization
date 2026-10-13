# Box-supervised instance-segmentation comparisons

This directory contains IRSTD-specific configurations and conversion tools for the [BoxInstSeg](https://github.com/LiWentomng/BoxInstSeg) implementation of BoxInst and Box2Mask. These are **external ResNet-50 system references**, not backbone-matched DNANet ablations. The source framework and its own license must be obtained separately.

## Prepare COCO files

Run from the HALO repository root after downloading the benchmark datasets. Change `HALO_DATASET` to `NUDT-SIRST` or `IRSTD-1k` when preparing those datasets.

```bash
export HALO_REPO="$PWD"
export HALO_DATA_ROOT="${HALO_DATA_ROOT:-$HALO_REPO/data}"
export HALO_DATASET=NUAA-SIRST
if [ "$HALO_DATASET" = "IRSTD-1k" ]; then HALO_IMAGE_FOLDER=IRSTD1k_Img; else HALO_IMAGE_FOLDER=images; fi
python boxseg/mask_to_coco.py \
  --images "$HALO_DATA_ROOT/$HALO_DATASET/$HALO_IMAGE_FOLDER" \
  --masks "$HALO_DATA_ROOT/$HALO_DATASET/masks" \
  --train-list "$HALO_REPO/splits/$HALO_DATASET/train.txt" \
  --test-list "$HALO_REPO/splits/$HALO_DATASET/test.txt" \
  --out "$HALO_DATA_ROOT/$HALO_DATASET/coco"
```

The training configuration requests boxes, not ground-truth masks. Ground-truth masks in the generated COCO records are used for evaluation. The exact split lists are distributed in `../splits/`.

## Run the external framework

Install BoxInstSeg and its documented MMDetection dependencies in a separate environment. Its custom CUDA operators must be built for the installed PyTorch/CUDA versions. Set `HALO_DATA_ROOT` to the absolute path containing the three dataset folders and `HALO_DATASET` to the dataset being evaluated. From the BoxInstSeg root, make this directory importable so `irstd_coco_dataset.py` is found:

```bash
export PYTHONPATH="$HALO_REPO/boxseg:${PYTHONPATH:-}"
python tools/train.py "$HALO_REPO/boxseg/boxinst_r50_irstd.py" --work-dir work_dirs/boxinst_nuaa
python tools/train.py "$HALO_REPO/boxseg/box2mask_r50_irstd.py" --work-dir work_dirs/box2mask_nuaa
```

The configurations refer to `HALO_DATA_ROOT/HALO_DATASET/coco/instances_train.json` and `instances_test.json`. Their evaluation hook uses `IRSTDCocoDataset`. For semantic mIoU comparisons, `export_semantic_pred.py` unions retained instance masks into one binary mask per image. Its default instance-score threshold is 0.05. Run `python export_semantic_pred.py --help` from this directory for the input arguments.

The framework configuration and postprocessing differ from the HALO/DNANet route. The manuscript therefore labels these rows as external system references. The repository does not bundle the BoxInstSeg source or its trained weights.
