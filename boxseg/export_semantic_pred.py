#!/usr/bin/env python3
# coding=utf-8
"""Export BoxInst or Box2Mask instance predictions as semantic binary masks.

Instances above the score threshold are combined by union. This provides
one output PNG per ground-truth mask stem for downstream mIoU evaluation.
Run this script from boxseg/ so custom dataset imports are available."""
import argparse
import os

import cv2
import numpy as np

import mmcv  # noqa: F401  (ensures mmdet deps present)
from mmdet.apis import inference_detector, init_detector

try:
    import pycocotools.mask as mask_util
except Exception:  # pragma: no cover
    mask_util = None

_IMG_SUFFIXES = ['.png', '.bmp', '.jpg', '.jpeg']


def _find_image(images_dir, stem):
    """Find an input image with the requested stem and a supported suffix."""
    for suf in _IMG_SUFFIXES:
        p = os.path.join(images_dir, stem + suf)
        if os.path.exists(p):
            return p
    for f in os.listdir(images_dir):
        if os.path.splitext(f)[0] == stem:
            return os.path.join(images_dir, f)
    return None


def _decode_segms(segm_result):
    """Decode single-class instance segmentation results into boolean masks."""
    if segm_result is None:
        return []
    segms = segm_result[0] if len(segm_result) > 0 else []
    out = []
    for m in segms:
        if isinstance(m, dict):  # RLE
            if mask_util is None:
                raise RuntimeError('pycocotools is required to decode RLE masks')
            out.append(mask_util.decode(m).astype(bool))
        else:  # ndarray
            out.append(np.asarray(m).astype(bool))
    return out


def build_binary_union(result, h, w, score_thresh):
    """Merge qualifying instance masks into one uint8 semantic prediction."""
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result) if len(bbox_result) > 0 else np.zeros((0, 5))
    scores = bboxes[:, 4] if bboxes.shape[0] > 0 else np.zeros((0,))
    segms = _decode_segms(segm_result)

    union = np.zeros((h, w), dtype=bool)
    for i, mask in enumerate(segms):
        if i >= len(scores) or scores[i] < score_thresh:
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        union |= mask
    return (union.astype(np.uint8) * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--images', required=True, help='Directory containing source images')
    ap.add_argument('--masks', required=True,
                    help='Ground-truth mask directory, used to identify file stems and output sizes')
    ap.add_argument('--out', required=True, help='Output directory for binary semantic masks')
    ap.add_argument('--score-thresh', type=float, default=0.05,
                    help='Minimum instance score for union (default 0.05)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)

    mask_files = sorted(f for f in os.listdir(args.masks)
                        if os.path.splitext(f)[1].lower() in _IMG_SUFFIXES)
    print(f'[export] {len(mask_files)} images, out -> {args.out}')

    missing = 0
    for k, mf in enumerate(mask_files):
        stem = os.path.splitext(mf)[0]
        img_path = _find_image(args.images, stem)
        gt = cv2.imread(os.path.join(args.masks, mf), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f'[WARN] Could not read ground-truth mask: {mf}; skipping')
            continue
        h, w = gt.shape[:2]
        if img_path is None:
            print(f'[WARN] No source image for stem={stem}; writing an empty prediction')
            cv2.imwrite(os.path.join(args.out, stem + '.png'),
                        np.zeros((h, w), np.uint8))
            missing += 1
            continue
        result = inference_detector(model, img_path)
        out_png = build_binary_union(result, h, w, args.score_thresh)
        cv2.imwrite(os.path.join(args.out, stem + '.png'), out_png)
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(mask_files)}')

    print(f'[export] done. missing images: {missing}')


if __name__ == '__main__':
    main()
