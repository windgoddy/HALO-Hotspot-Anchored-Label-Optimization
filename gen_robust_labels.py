"""Generate soft labels from PERTURBED boxes for the noise-robustness experiment.
One run = one (dataset, noise level, mode). Modes: halo (radiometric PAG),
center (geometric-center PAG), boxfill. Feeds the HALO training pipeline (train.py).

Example (moderate noise, HALO label):
    python gen_robust_labels.py --root ./data --dataset NUAA-SIRST --mode halo \
        --expand 2.0 --jitter 0.30 --seed 0 \
        --out ./data/NUAA-SIRST/masks_rob_halo_e2.0_j0.30
"""
import argparse
import glob
import os

import cv2
import numpy as np

EXPAND_CTX, TEMP, EPS, GAMMA = 1.5, 3.0, 1e-4, 1.5
IMG_FOLDER = {"NUAA-SIRST": "images", "NUDT-SIRST": "images", "IRSTD-1k": "IRSTD1k_Img"}
EXTS = (".png", ".bmp", ".jpg")


def _find(p):
    for e in EXTS:
        if os.path.exists(p + e):
            return p + e
    return None


def _to_gray(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0
    return im.astype(np.float32)


def make_label(g, x1, y1, x2, y2, mode):
    H, W = g.shape
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    if mode == "boxfill":
        return np.ones((bh, bw), np.float32)
    px = max(int((EXPAND_CTX - 1) / 2 * bw), 3)
    py = max(int((EXPAND_CTX - 1) / 2 * bh), 3)
    ctx = g[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
    mu, sig = float(ctx.mean()), float(ctx.std())
    snr = (g[y1:y2, x1:x2] - mu) / (sig + EPS)
    w = 1.0 / (1.0 + np.exp(-TEMP * snr))
    lo, hi = w.min(), w.max()
    w = (w - lo) / (hi - lo) if hi - lo > 1e-8 else np.full_like(w, 0.5)
    pf = int(snr.argmax())
    rpy, rpx = pf // bw, pf % bw
    yy, xx = np.mgrid[0:bh, 0:bw]
    peak = float(snr[rpy, rpx])
    hm = 0.5 * peak
    sf = np.where(snr >= hm, np.maximum(snr, 0.0), 0.0).astype(np.float32)
    s = sf.sum()
    sigma = max(GAMMA * np.sqrt((sf * ((yy - rpy) ** 2 + (xx - rpx) ** 2)).sum() / s), 1.0) if s > 1e-6 else 1.0
    cy, cx = (rpy, rpx) if mode == "halo" else ((bh - 1) / 2.0, (bw - 1) / 2.0)
    lab = (w * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32)
    return lab / lab.max() if lab.max() > 0 else lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True, choices=["halo", "center", "boxfill"])
    ap.add_argument("--expand", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-folder", default="labels_box")
    a = ap.parse_args()

    dsdir = os.path.join(a.root, a.dataset)
    imgf = IMG_FOLDER.get(a.dataset, "images")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n = 0
    for lp in sorted(glob.glob(os.path.join(dsdir, a.label_folder, "*.txt"))):
        sid = os.path.splitext(os.path.basename(lp))[0]
        ip = _find(os.path.join(dsdir, imgf, sid))
        if ip is None:
            cv2.imwrite(os.path.join(a.out, sid + ".png"), np.zeros((1, 1), np.uint8))
            continue
        g = _to_gray(cv2.imread(ip, cv2.IMREAD_UNCHANGED))
        H, W = g.shape
        m = np.zeros((H, W), np.float32)
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            ncx = cx + rng.uniform(-a.jitter, a.jitter) * w
            ncy = cy + rng.uniform(-a.jitter, a.jitter) * h
            x1 = int(max(0, ncx - w * a.expand / 2))
            y1 = int(max(0, ncy - h * a.expand / 2))
            x2 = int(min(W, ncx + w * a.expand / 2))
            y2 = int(min(H, ncy + h * a.expand / 2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lab = make_label(g, x1, y1, x2, y2, a.mode)
            if lab is None:
                continue
            m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], lab)
        cv2.imwrite(os.path.join(a.out, sid + ".png"), (m * 255).clip(0, 255).astype(np.uint8))
        n += 1
    print(f"[{a.dataset} {a.mode} exp{a.expand} jit{a.jitter} seed{a.seed}] {n} masks -> {a.out}")


if __name__ == "__main__":
    main()
