"""Export binarized HALO predictions as PNG images for qualitative comparison.

The model configuration follows test.py; predictions are resized back
to the source image size and saved as <image_id>.png."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import _pil_to_tensor, get_default_image_folder, resolve_dataset_root
from model import build_model


def str2bool(v):
    return v.lower() in {"1", "true", "yes", "y"} if isinstance(v, str) else bool(v)


def detect_from_checkpoint(path):
    """Infer input channels and deep supervision from a saved checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    in_ch = next((v.shape[1] for v in state.values() if v.ndim == 4), None)
    if any("final1" in k for k in state):
        ds = True
    elif [k for k in state if k.endswith("final.weight") or ".final" in k]:
        ds = False
    else:
        ds = None
    return ckpt, in_ch, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DNANet", choices=["DNANet", "ACM", "ALCNet"])
    ap.add_argument("--dataset", required=True,
                    choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "sirst_aug", "MDFA"])
    ap.add_argument("--root", default="./data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--channel_size", default="three")
    ap.add_argument("--backbone", default="resnet_18")
    ap.add_argument("--deep_supervision", default=None)
    ap.add_argument("--in_channels", type=int, default=None)
    ap.add_argument("--base_size", type=int, default=256)
    ap.add_argument("--thres", type=float, default=0.3)
    ap.add_argument("--suffix", default=".png")
    args = ap.parse_args()

    ckpt, ck_in, ck_ds = detect_from_checkpoint(args.checkpoint)
    in_ch = args.in_channels if args.in_channels is not None else (ck_in or 1)
    ds = str2bool(args.deep_supervision) if args.deep_supervision is not None else bool(ck_ds)

    root = resolve_dataset_root(args.root, args.dataset)
    img_folder = args.image_folder or get_default_image_folder(args.dataset)
    img_dir = Path(root) / img_folder
    stems = sorted(p.stem for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg"))
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, in_channels=in_ch, deep_supervision=ds,
                        channel_size=args.channel_size, backbone=args.backbone).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[{args.dataset}] {len(stems)} imgs | in_ch={in_ch} ds={ds} thres={args.thres} → {args.out_dir}")
    with torch.no_grad():
        for i, stem in enumerate(stems):
            ip = img_dir / f"{stem}{args.suffix}"
            pil = Image.open(ip).convert("L" if in_ch == 1 else "RGB")
            W, H = pil.size
            resized = pil.resize((args.base_size, args.base_size), Image.BILINEAR)
            t = _pil_to_tensor(resized, in_ch).unsqueeze(0).to(device)
            out = model(t)
            out = out[-1] if ds else out
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()     # base_size×base_size
            binm = (prob > args.thres).astype(np.uint8) * 255
            pred = Image.fromarray(binm).resize((W, H), Image.NEAREST)
            pred.save(os.path.join(args.out_dir, f"{stem}.png"))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(stems)}")
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
