"""Convert binary segmentation masks to YOLO-format box annotations.

Each foreground connected component in a mask is converted to one bounding
box line in YOLO format:

    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

All values are normalised to [0, 1] by the image width / height.
The class id is always 0 (infrared small target).

Usage
-----
Single dataset (results written to labels_box/ inside the dataset root):

    python gen_box_labels.py --dataset NUAA-SIRST --root ./data

All three standard datasets at once:

    bash scripts/gen_box_labels.sh ./data
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_boxes(mask: np.ndarray):
    """Return YOLO-format box list for every foreground component in *mask*.

    Parameters
    ----------
    mask:
        Binary uint8 image (0 = background, >0 = foreground).

    Returns
    -------
    list of (cx_norm, cy_norm, w_norm, h_norm) tuples, one per component.
    """
    H, W = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels_cc = cv2.connectedComponents(binary, connectivity=8)
    boxes = []
    for label_id in range(1, num_labels):  # skip background (0)
        ys, xs = np.where(labels_cc == label_id)
        if len(xs) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        boxes.append((cx / W, cy / H, w / W, h / H))
    return boxes


def process_dataset(dataset_root: str, mask_folder: str, output_folder: str, img_ids):
    mask_dir = Path(dataset_root) / mask_folder
    out_dir = Path(dataset_root) / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    empty = 0
    for img_id in img_ids:
        # Try common mask extensions
        mask_path = None
        for ext in (".png", ".bmp", ".jpg"):
            candidate = mask_dir / (img_id + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            print(f"[WARN] mask not found for {img_id} in {mask_dir}")
            empty += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] failed to read {mask_path}")
            empty += 1
            continue

        boxes = mask_to_yolo_boxes(mask)
        label_path = out_dir / (img_id + ".txt")
        with open(label_path, "w") as f:
            for cx, cy, w, h in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        written += 1

    print(f"  Done: {written} label files written, {empty} skipped -> {out_dir}")


def get_all_ids(dataset_root: str, mask_folder: str):
    mask_dir = Path(dataset_root) / mask_folder
    ids = []
    for p in sorted(mask_dir.iterdir()):
        if p.suffix.lower() in {".png", ".bmp", ".jpg"}:
            ids.append(p.stem)
    return ids


_DATASET_MASK_FOLDER = {
    "NUAA-SIRST": "masks",
    "NUDT-SIRST": "masks",
    "IRSTD-1k": "masks",
}

_DATASET_SUBFOLDER = {
    "NUAA-SIRST": "NUAA-SIRST",
    "NUDT-SIRST": "NUDT-SIRST",
    "IRSTD-1k": "IRSTD-1k",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO-format box labels from binary segmentation masks."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k"],
        choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k"],
        help="Dataset(s) to process (default: all three).",
    )
    parser.add_argument(
        "--root",
        default="./data",
        help="Root directory containing all dataset subfolders.",
    )
    parser.add_argument(
        "--mask_folder",
        default="masks",
        help="Name of the folder inside each dataset directory that contains GT masks.",
    )
    parser.add_argument(
        "--output_folder",
        default="labels_box",
        help="Name of the output folder for YOLO .txt files (default: labels_box).",
    )
    args = parser.parse_args()

    for ds in args.dataset:
        ds_root = Path(args.root) / _DATASET_SUBFOLDER[ds]
        if not ds_root.exists():
            print(f"[SKIP] {ds_root} does not exist")
            continue
        print(f"Processing {ds} ...")
        ids = get_all_ids(str(ds_root), args.mask_folder)
        process_dataset(str(ds_root), args.mask_folder, args.output_folder, ids)


if __name__ == "__main__":
    main()
