# HALO: Hotspot-Anchored Label Optimization

Code for **Hotspot-Anchored Label Optimization for Box-Supervised Infrared Small Target Detection**. HALO converts bounding boxes into soft pixel labels offline and trains an infrared small target segmentation network on those labels. This repository contains the label generator, the DNANet training and evaluation route for the three main benchmark results, exact split files, and the small set of configurations needed for comparisons reported in the manuscript. Unreported model attempts are omitted.

The manuscript has also appeared as [arXiv:2607.17148](https://arxiv.org/abs/2607.17148) under an earlier title. The current submission title is given above.

## What this repository reproduces

| Dataset | Reported HALO mIoU (%) | Main checkpoint | Split |
| --- | ---: | --- | --- |
| NUAA-SIRST | 73.53 | `best_model_epoch0417_mIoU0.7353_BoxIoU0.7089.pth.tar` | `splits/NUAA-SIRST/` (224 train / 96 test) |
| NUDT-SIRST | 67.49 | `best_model_epoch0440_mIoU0.6749_BoxIoU0.8307.pth.tar` | `splits/NUDT-SIRST/` (663 train / 664 test) |
| IRSTD-1k | 64.33 | `best_model_epoch0273_mIoU0.6433_BoxIoU0.6277.pth.tar` | `splits/IRSTD-1k/` (800 train / 201 test) |

The code in the repository prioritizes these split files over split files that may be bundled with a dataset download. `--split_dir` can override that choice. In a local evaluation with the supplied NUAA checkpoint, the 96-image test list reproduced the reported 73.53% mIoU, 98.04% Pd, and 7.95e-07 Fa. The dataset's separate 214-image test list produced 71.09% mIoU with the same weights, so these splits are not interchangeable. The main checkpoints are **not committed to Git**. A download link and the recorded training arguments are documented in [checkpoints/README.md](checkpoints/README.md). The `boxseg/` directory contains auxiliary BoxInst/Box2Mask comparison configurations; those require their upstream framework and are not part of the DNANet reproduction route.

## Requirements

- Python 3.9.20 and PyTorch 2.1.2 were used for the manuscript experiments.
- Create a Python 3.9 environment and run `python -m pip install -r requirements.txt`. The requirements pin PyTorch 2.1.2 and include the image-processing packages imported by the main route.
- Run the commands below from the repository root. GPU training is recommended; the label generator and evaluation can run on CPU.

## Data access and layout

Download the three third-party datasets from their original sources. Their licenses and redistribution terms remain with the dataset owners; this repository stores only text split files and generation scripts.

- [NUAA-SIRST](https://github.com/YimianDai/sirst)
- [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
- [IRSTD-1k](https://github.com/RuiZhang97/ISNet)

Place the downloaded images and ground-truth masks under a common data root:

```text
data/
  NUAA-SIRST/{images,masks}/
  NUDT-SIRST/{images,masks}/
  IRSTD-1k/{IRSTD1k_Img,masks}/
```

Image and mask stems must match the IDs in `splits/<dataset>/{train,test}.txt`. The masks are used to derive tight boxes and to evaluate predictions. Test masks are not used to generate training labels. See the cited benchmark repositories for acquisition and dataset-specific file formats.

Check the split lists and downloaded image/mask files before running an experiment:

```bash
python scripts/check_data.py --root ./data
```

## Evaluate the released checkpoints

Download the three weights using [checkpoints/README.md](checkpoints/README.md), place them under `checkpoints/`, verify them against the included SHA-256 manifest, then run:

```bash
bash scripts/eval_all.sh ./data
```

The expected mIoU values are in the table above. This evaluation route does not require generating pseudo labels or retraining.

## Regenerate labels and train

```bash
# 1. Create tight YOLO-format boxes from benchmark masks.
bash scripts/gen_box_labels.sh ./data

# 2. Generate soft labels for training images only.
bash scripts/prepare_labels.sh ./data

# 3. Train the three DNANet configurations from the generated labels.
bash scripts/train_nuaa.sh ./data
bash scripts/train_nudt.sh ./data
bash scripts/train_irstd.sh ./data
```

The box generator processes all available masks for convenience, but `label_gen.py` selects only IDs in the repository's training split when creating training labels. The training scripts use a fixed seed of 42, a 256-pixel input, Adagrad with an initial learning rate of 0.05, cosine annealing, batch size 4, and 1500 epochs. Training output is written to ignored `result/` directories. The reported mIoU is calculated after thresholding predictions at 0.3. The auxiliary detection metrics use a separate 0.5 threshold, as described in the manuscript.

## Other backbones

`train.py` and `test.py` also accept `ACM`, `ALCNet`, `SCTransNet`, and `MSHNet` for the manuscript's backbone comparison. ACM and ALCNet are included under `model/`. SCTransNet and MSHNet require their upstream source trees; `bash scripts/setup_external_backbones.sh` downloads them into the ignored `external/` directory and checks out the revisions used for the manuscript (SCTransNet `e027628`, MSHNet `46cdfd4`). For single-output models (`ACM`, `ALCNet`, and `MSHNet`), train with `--deep_supervision False`. The backbone-comparison models can be retrained with the same HALO labels and training scripts.

## Repository map

- `gen_box_labels.py`: derive tight boxes from ground-truth masks for the controlled benchmark protocol.
- `label_gen.py`: produce HALO soft masks from images and boxes.
- `train.py`, `test.py`, `metrics.py`, `dataset.py`: DNANet training and evaluation.
- `model/`: DNANet, ACM, ALCNet, and adapters for the two external backbones.
- `scripts/`: commands for the three main datasets.
- `scripts/check_data.py`: verify the split IDs and required image/mask files.
- `splits/`: exact train/test ID lists used by the public reproduction route.
- `checkpoints/`: download instructions and training argument records; model weights are ignored by Git.
- `boxseg/`: auxiliary BoxInst/Box2Mask comparison configuration and conversion tools.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software and cite the benchmark owners for the datasets. This repository is distributed under the [MIT License](LICENSE); external dependencies and third-party datasets retain their own licenses.
