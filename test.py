"""Evaluation script aligned with the final HALO best checkpoints."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TestSetLoader, get_default_image_folder, get_default_split_method, load_split_ids, resolve_dataset_root
from metrics import PD_FA, calculate_mask_to_box_iou, mIoU
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="HALO evaluation")
    parser.add_argument("--model", default="DNANet", choices=["DNANet", "ACM", "ALCNet", "SCTransNet", "MSHNet"])
    parser.add_argument("--dataset", required=True, choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "sirst_aug", "MDFA"])
    parser.add_argument("--root", default="./data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_method", default=None)
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--mask_folder", default="masks")
    parser.add_argument("--channel_size", default="three")
    parser.add_argument("--backbone", default="resnet_18")
    parser.add_argument("--deep_supervision", default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--thres", type=float, default=0.3)
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def detect_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]
    in_channels = None
    deep_supervision = None
    for key, value in state.items():
        if value.ndim == 4:
            in_channels = value.shape[1]
            break
    final_keys = [key for key in state.keys() if key.endswith("final.weight") or ".final" in key]
    if any("final1" in key for key in state.keys()):
        deep_supervision = True
    elif final_keys:
        deep_supervision = False
    return checkpoint, in_channels, deep_supervision


def main():
    args = parse_args()
    checkpoint, ckpt_in_channels, ckpt_deep_supervision = detect_from_checkpoint(args.checkpoint)
    saved_args = checkpoint.get("args")
    saved_args = saved_args if isinstance(saved_args, dict) else {}
    in_channels = args.in_channels if args.in_channels is not None else saved_args.get("in_channels", ckpt_in_channels or 1)
    if args.deep_supervision is not None:
        deep_supervision = str2bool(args.deep_supervision)
    else:
        deep_supervision = str2bool(saved_args["deep_supervision"]) if "deep_supervision" in saved_args else bool(ckpt_deep_supervision)

    dataset_root = resolve_dataset_root(args.root, args.dataset)
    split_method = args.split_method or get_default_split_method(args.dataset)
    image_folder = args.image_folder or get_default_image_folder(args.dataset)
    _, test_ids = load_split_ids(args.root, args.dataset, split_method, args.split_dir)
    testset = TestSetLoader(
        str(dataset_root),
        img_id=test_ids,
        base_size=args.base_size,
        in_channels=in_channels,
        image_folder=image_folder,
        mask_folder=args.mask_folder,
    )
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        in_channels=in_channels,
        deep_supervision=deep_supervision,
        channel_size=args.channel_size,
        backbone=args.backbone,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    eval_miou = mIoU(1, threshold=args.thres)
    eval_pdfa = PD_FA(1, bins=10, img_size=args.base_size)
    box_iou_sum = 0.0
    box_iou_count = 0

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            preds = model(data)
            pred = preds[-1] if deep_supervision else preds
            eval_miou.update(pred, labels)
            eval_pdfa.update(torch.sigmoid(pred) * 255, labels)
            box_iou_sum += calculate_mask_to_box_iou(pred, labels, threshold=args.thres)
            box_iou_count += 1

    pix_acc, mean_iou = eval_miou.get()
    fa, pd = eval_pdfa.get(len(test_ids))
    pd_report = float(pd[5]) if not torch.isnan(torch.tensor(pd[5])) else 0.0
    fa_report = float(fa[5])
    mean_box_iou = box_iou_sum / max(box_iou_count, 1)

    print(f"Dataset   : {args.dataset}")
    print(f"Model     : {args.model}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print(f"deep_supervision: {deep_supervision}")
    print(f"PixAcc    : {pix_acc:.4f}")
    print(f"mIoU      : {mean_iou:.4f}")
    print(f"BoxIoU    : {mean_box_iou:.4f}")
    print(f"Pd        : {pd_report:.4f}")
    print(f"Fa        : {fa_report:.2e}")


if __name__ == "__main__":
    main()
