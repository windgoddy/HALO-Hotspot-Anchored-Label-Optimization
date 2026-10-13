"""Training script aligned with the final HALO best checkpoints."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adagrad, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    TestSetLoader,
    TrainSetLoader,
    get_default_image_folder,
    get_default_split_method,
    load_split_ids,
    resolve_dataset_root,
)
from losses import soft_iou_loss
from metrics import PD_FA, ROCMetric, calculate_mask_to_box_iou, mIoU
from model import build_model
from utils import AverageMeter, ensure_dir, set_seed, weights_init_xavier


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="HALO training")
    parser.add_argument("--model", default="DNANet", choices=["DNANet", "ACM", "ALCNet", "SCTransNet", "MSHNet"])
    parser.add_argument("--dataset", required=True, choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "sirst_aug", "MDFA"])
    parser.add_argument("--root", default="./data")
    parser.add_argument("--split_method", default=None)
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--mask_folder", default="masks_sweep_sigma1.500_expand1.5")
    parser.add_argument("--channel_size", default="three")
    parser.add_argument("--backbone", default="resnet_18")
    parser.add_argument("--deep_supervision", default="True")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", default="Adagrad", choices=["Adagrad", "Adam"])
    parser.add_argument("--scheduler", default="CosineAnnealingLR")
    parser.add_argument("--thres", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--aug_rotate", action="store_true")
    return parser.parse_args()


def save_best_checkpoint(save_dir, epoch, model, optimizer, args, mean_iou, box_iou, test_loss, recall, precision):
    save_dir = ensure_dir(save_dir)
    filename = f"best_model_epoch{epoch:04d}_mIoU{mean_iou:.4f}_BoxIoU{box_iou:.4f}.pth.tar"
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": test_loss,
        "mean_IOU": mean_iou,
        "box_IOU": box_iou,
        "recall": recall,
        "precision": precision,
        "args": vars(args),
    }
    for old_file in Path(save_dir).glob("best_model_epoch*_mIoU*_BoxIoU*.pth.tar"):
        old_file.unlink(missing_ok=True)
    torch.save(state, Path(save_dir) / filename)


def save_last_checkpoint(save_dir, epoch, model, optimizer, args, mean_iou, box_iou):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "mean_IOU": mean_iou,
        "box_IOU": box_iou,
        "args": vars(args),
    }
    torch.save(state, Path(save_dir) / "last.pth.tar")


def main():
    args = parse_args()
    args.deep_supervision = str2bool(args.deep_supervision)
    set_seed(args.seed)

    dataset_root = resolve_dataset_root(args.root, args.dataset)
    split_method = args.split_method or get_default_split_method(args.dataset)
    image_folder = args.image_folder or get_default_image_folder(args.dataset)
    train_ids, test_ids = load_split_ids(args.root, args.dataset, split_method, args.split_dir)

    trainset = TrainSetLoader(
        str(dataset_root),
        img_id=train_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        in_channels=args.in_channels,
        image_folder=image_folder,
        mask_folder=args.mask_folder,
        aug_rotate=args.aug_rotate,
    )
    testset = TestSetLoader(
        str(dataset_root),
        img_id=test_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        in_channels=args.in_channels,
        image_folder=image_folder,
        mask_folder="masks",
    )
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        in_channels=args.in_channels,
        deep_supervision=args.deep_supervision,
        channel_size=args.channel_size,
        backbone=args.backbone,
    ).to(device)
    model.apply(weights_init_xavier)

    optimizer = Adagrad(model.parameters(), lr=args.lr) if args.optimizer == "Adagrad" else Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    if args.save_dir:
        save_dir = ensure_dir(args.save_dir)
    else:
        exp_name = args.experiment_name or f"{args.dataset}_{args.model}"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = ensure_dir(Path("result") / f"{exp_name}_{args.dataset}_{args.model}_{stamp}")

    with open(Path(save_dir) / "args.txt", "w", encoding="utf-8") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")

    best_iou = 0.0
    roc = ROCMetric(1, 10)
    miou_metric = mIoU(1, threshold=args.thres)
    pd_fa_metric = PD_FA(1, bins=10, img_size=args.base_size)

    for epoch in range(args.epochs):
        model.train()
        train_loss = AverageMeter()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for data, labels in progress:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(data)
            if args.deep_supervision:
                loss = 0.0
                for pred in preds:
                    loss += soft_iou_loss(pred, labels)
                loss = loss / len(preds)
                pred_for_log = preds[-1]
            else:
                pred_for_log = preds
                loss = soft_iou_loss(pred_for_log, labels)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), pred_for_log.size(0))
            progress.set_postfix(loss=f"{train_loss.avg:.4f}")

        scheduler.step()

        model.eval()
        roc.reset()
        miou_metric.reset()
        pd_fa_metric.reset()
        test_loss = AverageMeter()
        box_iou_sum = 0.0
        box_iou_count = 0
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                preds = model(data)
                if args.deep_supervision:
                    loss = 0.0
                    for pred in preds:
                        loss += soft_iou_loss(pred, labels)
                    loss = loss / len(preds)
                    pred = preds[-1]
                else:
                    pred = preds
                    loss = soft_iou_loss(pred, labels)

                test_loss.update(loss.item(), pred.size(0))
                roc.update(pred, labels)
                miou_metric.update(pred, labels)
                pd_fa_metric.update(torch.sigmoid(pred) * 255, labels)
                box_iou_sum += calculate_mask_to_box_iou(pred, labels, threshold=args.thres)
                box_iou_count += 1

        _, _, recall, precision = roc.get()
        _, mean_iou = miou_metric.get()
        mean_box_iou = box_iou_sum / max(box_iou_count, 1)
        fa, pd = pd_fa_metric.get(len(test_ids))
        pd_report = float(pd[5]) if not np.isnan(pd[5]) else 0.0
        fa_report = float(fa[5])

        log_line = (
            f"epoch={epoch:04d} train_loss={train_loss.avg:.4f} "
            f"test_loss={test_loss.avg:.4f} mIoU={mean_iou:.4f} "
            f"BoxIoU={mean_box_iou:.4f} Pd={pd_report:.4f} Fa={fa_report:.2e}"
        )
        print(log_line)
        with open(Path(save_dir) / "train.log", "a", encoding="utf-8") as handle:
            handle.write(log_line + "\n")

        save_last_checkpoint(save_dir, epoch, model, optimizer, args, mean_iou, mean_box_iou)
        if mean_iou > best_iou:
            best_iou = mean_iou
            save_best_checkpoint(save_dir, epoch, model, optimizer, args, mean_iou, mean_box_iou, test_loss.avg, recall, precision)


if __name__ == "__main__":
    main()
