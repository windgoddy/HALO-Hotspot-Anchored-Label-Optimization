"""Like mask_to_coco.py, but the TRAIN bounding boxes are PERTURBED (expand + jitter)
to simulate noisy box annotations, for the noise-robustness experiment.

Key point: only the bbox (box supervision used for training) is perturbed; the
segmentation (true GT mask, used only for mIoU evaluation) is kept exact, and the
TEST split is left tight (the model uses no boxes at test time).

Example (moderate noise):
    python mask_to_coco_noisy.py \
        --images /path/HALO/data/NUAA-SIRST/images --masks /path/HALO/data/NUAA-SIRST/masks \
        --train-list .../50_50/train.txt --test-list .../50_50/test.txt \
        --out .../coco_e2.0_j0.30 --expand 2.0 --jitter 0.30 --seed 0
"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def _find(d, stem, sfx):
    for s in sfx:
        p = os.path.join(d, stem + s)
        if os.path.exists(p):
            return p
    return None


def build_split(ids, images_dir, masks_dir, img_sfx, mask_sfx, expand, jitter, rng):
    images, anns = [], []
    ann_id = 1
    for img_id, stem in enumerate(ids, start=1):
        ip = _find(images_dir, stem, [img_sfx, ".png", ".bmp", ".jpg"])
        mp = _find(masks_dir, stem, mask_sfx)
        if ip is None or mp is None:
            print(f"[WARN] missing image/mask for '{stem}', skipped")
            continue
        im = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        H, W = im.shape[:2]
        images.append(dict(id=img_id, file_name=os.path.basename(ip), width=int(W), height=int(H)))
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        binary = (mask > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(binary, connectivity=8)
        for comp in range(1, num):
            comp_mask = np.asfortranarray((labels == comp).astype(np.uint8))
            ys, xs = np.where(comp_mask)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w, h = x2 - x1 + 1, y2 - y1 + 1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            # ---- perturb the bbox (training supervision) ----
            ncx = cx + rng.uniform(-jitter, jitter) * w
            ncy = cy + rng.uniform(-jitter, jitter) * h
            nw, nh = w * expand, h * expand
            bx1 = max(0.0, ncx - nw / 2.0)
            by1 = max(0.0, ncy - nh / 2.0)
            bx2 = min(W, ncx + nw / 2.0)
            by2 = min(H, ncy + nh / 2.0)
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 1 or bh < 1:
                continue
            rle = mask_utils.encode(comp_mask)         # segmentation = TRUE mask (for eval)
            rle["counts"] = rle["counts"].decode("ascii")
            anns.append(dict(id=ann_id, image_id=img_id, category_id=1,
                             bbox=[bx1, by1, bw, bh], area=int(comp_mask.sum()),
                             segmentation=rle, iscrowd=0))
            ann_id += 1
    return images, anns


def coco(images, anns):
    return dict(images=images, annotations=anns,
               categories=[dict(id=1, name="target", supercategory="target")])


def read_ids(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default=".png")
    ap.add_argument("--mask-suffix", default=".png")
    ap.add_argument("--expand", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    msfx = [a.mask_suffix, ".png", ".bmp", ".jpg"]
    rng = np.random.default_rng(a.seed)

    # TRAIN: perturbed boxes
    tr_imgs, tr_anns = build_split(read_ids(a.train_list), a.images, a.masks, a.suffix, msfx,
                                   a.expand, a.jitter, rng)
    with open(os.path.join(a.out, "instances_train.json"), "w") as f:
        json.dump(coco(tr_imgs, tr_anns), f)
    print(f"[train exp{a.expand} jit{a.jitter}] {len(tr_imgs)} imgs, {len(tr_anns)} inst")

    # TEST: tight boxes (eval uses segmentation; model uses no box at test)
    te_imgs, te_anns = build_split(read_ids(a.test_list), a.images, a.masks, a.suffix, msfx,
                                   1.0, 0.0, np.random.default_rng(0))
    with open(os.path.join(a.out, "instances_test.json"), "w") as f:
        json.dump(coco(te_imgs, te_anns), f)
    print(f"[test tight] {len(te_imgs)} imgs, {len(te_anns)} inst -> {a.out}")


if __name__ == "__main__":
    main()
