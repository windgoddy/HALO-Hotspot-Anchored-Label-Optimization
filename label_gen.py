"""HALO pseudo-label generation."""

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

from dataset import (
    get_default_image_folder,
    get_default_split_method,
    read_id_file,
    resolve_dataset_root,
    resolve_split_files,
)


def compute_bsnr_weight(
    image_gray: np.ndarray,
    box: Tuple[int, int, int, int],
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    epsilon: float = 1e-4,
    spatial_gaussian: bool = False,
    gauss_sigma_ratio: float = 1.5,
) -> np.ndarray:
    """Compute a context-normalized radiometric confidence map inside one box.

The optional peak-centered Gaussian weights the confidence map using
a spread estimated from the half-maximum radiometric response.
Returns a float32 map with the spatial size of the input box."""
    H, W = image_gray.shape
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return np.zeros((max(1, box_h), max(1, box_w)), dtype=np.float32)

    pad_x = int((expand_ratio - 1.0) / 2.0 * box_w)
    pad_y = int((expand_ratio - 1.0) / 2.0 * box_h)
    pad_x = max(pad_x, 3)
    pad_y = max(pad_y, 3)

    ctx_x1 = max(0, x1 - pad_x)
    ctx_y1 = max(0, y1 - pad_y)
    ctx_x2 = min(W, x2 + pad_x)
    ctx_y2 = min(H, y2 + pad_y)

    ctx_region = image_gray[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    mu = ctx_region.mean()
    sigma = ctx_region.std()

    target_region = image_gray[y1:y2, x1:x2]  # (box_h, box_w)

    snr = (target_region - mu) / (sigma + epsilon)
    weight = 1.0 / (1.0 + np.exp(-temperature * snr))  # sigmoid

    w_min = weight.min()
    w_max = weight.max()
    if w_max - w_min > 1e-8:
        weight = (weight - w_min) / (w_max - w_min)
    else:
        weight = np.full_like(weight, 0.5)

    if spatial_gaussian:
        peak_flat = snr.argmax()
        peak_y = peak_flat // box_w
        peak_x = peak_flat % box_w

        yy, xx = np.mgrid[0:box_h, 0:box_w]
        dist_sq = ((yy - peak_y) ** 2 + (xx - peak_x) ** 2).astype(np.float32)

        peak_snr_val = float(snr[peak_y, peak_x])
        half_max = peak_snr_val * 0.5
        snr_fwhm = np.where(snr >= half_max, np.maximum(snr, 0.0), 0.0).astype(np.float32)
        snr_sum = snr_fwhm.sum()
        if snr_sum > 1e-6:
            snr_weighted_var = (snr_fwhm * dist_sq).sum() / snr_sum
            sigma_px = max(gauss_sigma_ratio * np.sqrt(snr_weighted_var), 1.0)
        else:
            sigma_px = 1.0

        gauss = np.exp(-dist_sq / (2 * sigma_px ** 2)).astype(np.float32)
        weight = weight * gauss

    return weight.astype(np.float32)


def generate_bsnr_mask(
    labels: List[List[float]],
    img_size: Tuple[int, int],
    image: Optional[np.ndarray] = None,
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.0,
    soft_label: bool = True,
    max_value: float = 1.0,
    spatial_gaussian: bool = False,
    gauss_sigma_ratio: float = 1.5,
) -> np.ndarray:
    """Convert normalized YOLO boxes and an infrared image into a soft mask.

For multiple boxes, per-box confidence maps are merged by pixelwise maximum.
An empty image or label list yields an all-zero mask."""
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)

    if image is None or len(labels) == 0:
        return mask

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0
    elif gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = gray.astype(np.float32)

    for label in labels:
        if len(label) < 5:
            continue

        class_id, cx_norm, cy_norm, w_norm, h_norm = label[:5]

        cx = cx_norm * W
        cy = cy_norm * H
        bw = w_norm * W
        bh = h_norm * H

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(W, cx + bw / 2))
        y2 = int(min(H, cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            continue

        weight = compute_bsnr_weight(
            gray, (x1, y1, x2, y2),
            expand_ratio=expand_ratio,
            temperature=temperature,
            spatial_gaussian=spatial_gaussian,
            gauss_sigma_ratio=gauss_sigma_ratio,
        )

        if fg_threshold > 0:
            weight[weight < fg_threshold] = 0.0

        if soft_label:
            refined = weight * max_value
        else:
            refined = (weight >= fg_threshold).astype(np.float32) * max_value

        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], refined)

    return mask


def generate_bsnr_mask_batch_offline(
    dataset_dir: str,
    img_ids: List[str],
    output_dir: str,
    image_folder: str = 'images',
    label_folder: str = 'labels_box',
    suffix: str = '.png',
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.0,
    soft_label: bool = True,
    max_value: float = 1.0,
    visualize: bool = False,
    vis_output_dir: Optional[str] = None,
    spatial_gaussian: bool = False,
    gauss_sigma_ratio: float = 1.5,
):
    """Generate PNG pseudo masks for the supplied image IDs before training.

Images and YOLO boxes are read from the dataset directory. Optional
visualizations are saved separately; no detector forward pass is used."""
    os.makedirs(output_dir, exist_ok=True)
    if visualize and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)

    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)

    total = len(img_ids)
    success = 0
    empty = 0

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(images_dir, img_id + suffix)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        H, W = image.shape[:2]

        label_path = os.path.join(labels_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)

        mask = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=expand_ratio,
            temperature=temperature,
            fg_threshold=fg_threshold,
            soft_label=soft_label,
            max_value=max_value,
            spatial_gaussian=spatial_gaussian,
            gauss_sigma_ratio=gauss_sigma_ratio,
        )

        mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, img_id + '.png')
        cv2.imwrite(out_path, mask_uint8)

        if mask.max() > 0:
            success += 1
        else:
            empty += 1

        if visualize and vis_output_dir and len(labels) > 0 and success <= 50:
            _save_visualization(
                image, mask, labels, (H, W),
                os.path.join(vis_output_dir, img_id + '_vis.png'),
                expand_ratio=expand_ratio,
            )

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] success={success}, empty={empty}")

    print(f"Done: {success} masks generated, {empty} empty, saved to {output_dir}")


def _load_yolo_labels(label_path: str) -> List[List[float]]:
    """Read one YOLO-format box-label file into a list of numeric rows."""
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels.append([class_id, cx, cy, w, h])
    except FileNotFoundError:
        pass
    return labels


def _save_visualization(
    image: np.ndarray,
    bsnr_mask: np.ndarray,
    labels: List[List[float]],
    img_size: Tuple[int, int],
    save_path: str,
    expand_ratio: float = 1.5,
):
    """Save an image, box-fill map, centered Gaussian, and HALO map side by side."""
    H, W = img_size

    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image.astype(np.uint8) if image.dtype != np.uint8 else image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)

    for label in labels:
        if len(label) < 5:
            continue
        _, cx_n, cy_n, w_n, h_n = label[:5]
        x1 = int(max(0, cx_n * W - w_n * W / 2))
        y1 = int(max(0, cy_n * H - h_n * H / 2))
        x2 = int(min(W, cx_n * W + w_n * W / 2))
        y2 = int(min(H, cy_n * H + h_n * H / 2))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    box_mask = np.zeros((H, W), dtype=np.float32)
    for label in labels:
        if len(label) < 5:
            continue
        _, cx_n, cy_n, w_n, h_n = label[:5]
        x1 = int(max(0, cx_n * W - w_n * W / 2))
        y1 = int(max(0, cy_n * H - h_n * H / 2))
        x2 = int(min(W, cx_n * W + w_n * W / 2))
        y2 = int(min(H, cy_n * H + h_n * H / 2))
        box_mask[y1:y2, x1:x2] = 1.0

    gauss_mask = np.zeros((H, W), dtype=np.float32)
    for lbl in labels:
        if len(lbl) < 5:
            continue
        _, cx_n, cy_n, w_n, h_n = lbl[:5]
        cx_px, cy_px = cx_n * W, cy_n * H
        r = max(w_n * W, h_n * H) / 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        gauss_mask = np.maximum(gauss_mask,
            np.exp(-((xx - cx_px)**2 + (yy - cy_px)**2) / (2 * (r + 1e-4)**2)).astype(np.float32)
        )
    gauss_mask = np.clip(gauss_mask, 0, 1)

    def to_heatmap(m):
        m_u8 = (m * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(m_u8, cv2.COLORMAP_JET)

    box_heat = to_heatmap(box_mask)
    gauss_heat = to_heatmap(gauss_mask)
    bsnr_heat = to_heatmap(bsnr_mask)

    row = np.hstack([vis_img, box_heat, gauss_heat, bsnr_heat])
    cv2.imwrite(save_path, row)


# ==========================================
# CLI entry point
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HALO pseudo masks from box annotations")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k"])
    parser.add_argument("--root", type=str, default=None,
                        help="Dataset root, e.g. ./data/NUAA-SIRST")
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Backward-compatible alias of --root')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to train.txt split file. If omitted, use repo splits or all labels.')
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--output_folder', type=str, default='masks_sweep_sigma1.500_expand1.5')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional absolute or relative output directory')
    parser.add_argument('--label_folder', type=str, default='labels_box')
    parser.add_argument('--split_method', type=str, default=None)
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--fg_threshold', type=float, default=0.0)
    parser.add_argument('--spatial_gaussian', action='store_true',
                        help='Enable PAG: B-SNR x physical-anchor Gaussian (Stage II)')
    parser.add_argument('--gauss_sigma_ratio', type=float, default=1.5)
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    dataset_dir = args.root or args.dataset_dir
    if dataset_dir is None:
        raise ValueError("Please provide --root (or legacy --dataset_dir).")
    if args.dataset is not None:
        dataset_dir = str(resolve_dataset_root(dataset_dir, args.dataset))

    image_folder = args.image_folder or get_default_image_folder(args.dataset)
    if args.split_file:
        img_ids = read_id_file(args.split_file)
        print(f"Loaded {len(img_ids)} IDs from {args.split_file}")
    elif args.dataset is not None:
        split_method = args.split_method or get_default_split_method(args.dataset)
        split_file, _ = resolve_split_files(args.root or args.dataset_dir, args.dataset, split_method, None)
        img_ids = read_id_file(split_file)
        print(f"Loaded {len(img_ids)} training IDs from {split_file}")
    else:
        labels_dir = os.path.join(dataset_dir, args.label_folder)
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"Found {len(img_ids)} label files in {labels_dir}")

    output_dir = args.save_dir or os.path.join(dataset_dir, args.output_folder)
    vis_dir = os.path.join(dataset_dir, args.output_folder + '_vis') if args.visualize else None

    stage = 'II (B-SNR + PAG)' if args.spatial_gaussian else 'I (B-SNR only)'
    print(f"\nHALO Pseudo-Label Generation — Stage {stage}")
    print(f"  dataset_dir:      {dataset_dir}")
    print(f"  image_folder:     {image_folder}")
    print(f"  output_folder:    {output_dir}")
    print(f"  expand_ratio:     {args.expand_ratio}")
    print(f"  temperature:      {args.temperature}")
    print(f"  spatial_gaussian: {args.spatial_gaussian}")
    print()

    generate_bsnr_mask_batch_offline(
        dataset_dir    = dataset_dir,
        img_ids        = img_ids,
        output_dir     = output_dir,
        image_folder   = image_folder,
        label_folder   = args.label_folder,
        suffix         = args.suffix,
        expand_ratio   = args.expand_ratio,
        temperature    = args.temperature,
        fg_threshold   = args.fg_threshold,
        soft_label     = True,
        max_value      = args.max_value,
        visualize      = args.visualize,
        vis_output_dir = vis_dir,
        spatial_gaussian  = args.spatial_gaussian,
        gauss_sigma_ratio = args.gauss_sigma_ratio,
    )
