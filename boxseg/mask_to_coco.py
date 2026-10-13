"""Convert an IRSTD dataset (images + binary GT masks + split lists) into COCO
instance-segmentation JSON, single class ('target'), for the reported
BoxInst and Box2Mask comparisons in the BoxInstSeg toolbox.

Each foreground connected component in a GT mask becomes one instance:
  * bbox        : tight box [x, y, w, h]  (box supervision used for training)
  * segmentation: RLE of the component   (GT mask, used only for mask-AP evaluation)
  * category_id : 1  ('target')

Run inside the BoxInstSeg conda env (needs pycocotools, which it already has).

Example (NUAA-SIRST, using the split files shipped in HALO/splits/):
    python mask_to_coco.py \
        --images   /path/HALO/data/NUAA-SIRST/images \
        --masks    /path/HALO/data/NUAA-SIRST/masks \
        --train-list /path/HALO/splits/NUAA-SIRST/train.txt \
        --test-list  /path/HALO/splits/NUAA-SIRST/test.txt \
        --out      /path/HALO/data/NUAA-SIRST/coco \
        --suffix .png
Produces:  <out>/instances_train.json  and  <out>/instances_test.json
(img_prefix for the mmdet config is the --images dir; file_name is the bare filename.)
"""

import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def _find(dir_path, stem, suffixes):
    for s in suffixes:
        p = os.path.join(dir_path, stem + s)
        if os.path.exists(p):
            return p
    return None


def build_split(ids, images_dir, masks_dir, img_suffix, mask_suffixes):
    images, annotations = [], []
    ann_id = 1
    for img_id, stem in enumerate(ids, start=1):
        img_path = _find(images_dir, stem, [img_suffix, ".png", ".bmp", ".jpg"])
        msk_path = _find(masks_dir, stem, mask_suffixes)
        if img_path is None or msk_path is None:
            print(f"[WARN] missing image/mask for '{stem}', skipped")
            continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] cannot read {img_path}, skipped")
            continue
        h, w = img.shape[:2]
        images.append(dict(id=img_id, file_name=os.path.basename(img_path),
                           width=int(w), height=int(h)))

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        binary = (mask > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(binary, connectivity=8)
        for comp in range(1, num):  # 0 is background
            comp_mask = np.asfortranarray((labels == comp).astype(np.uint8))
            ys, xs = np.where(comp_mask)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            rle = mask_utils.encode(comp_mask)
            rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
            annotations.append(dict(
                id=ann_id,
                image_id=img_id,
                category_id=1,
                bbox=[x1, y1, bw, bh],
                area=int(comp_mask.sum()),
                segmentation=rle,
                iscrowd=0,
            ))
            ann_id += 1
    return images, annotations


def make_coco(images, annotations):
    return dict(
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name="target", supercategory="target")],
    )


def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of input images")
    ap.add_argument("--masks", required=True, help="directory of binary GT masks")
    ap.add_argument("--train-list", required=True, help="train.txt (one id per line, no suffix)")
    ap.add_argument("--test-list", required=True, help="test.txt")
    ap.add_argument("--out", required=True, help="output dir for COCO json files")
    ap.add_argument("--suffix", default=".png", help="image file suffix (default .png)")
    ap.add_argument("--mask-suffix", default=".png", help="mask file suffix (default .png)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mask_suffixes = [args.mask_suffix, ".png", ".bmp", ".jpg"]

    for split, list_path in [("train", args.train_list), ("test", args.test_list)]:
        ids = read_ids(list_path)
        images, anns = build_split(ids, args.images, args.masks, args.suffix, mask_suffixes)
        out_path = os.path.join(args.out, f"instances_{split}.json")
        with open(out_path, "w") as f:
            json.dump(make_coco(images, anns), f)
        print(f"[{split}] {len(images)} images, {len(anns)} instances -> {out_path}")


if __name__ == "__main__":
    main()
