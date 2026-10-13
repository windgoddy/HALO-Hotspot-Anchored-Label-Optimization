"""Small utilities for training and evaluation."""

from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import init


def weights_init_xavier(module):
    if module.__class__.__name__.find("Conv2d") != -1 and hasattr(module, "weight"):
        init.xavier_normal_(module.weight.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_save_dir(dataset, backbone):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path("result") / f"{dataset}_{backbone}_{stamp}")


def save_json_like_log(path, items):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(items + "\n")


def save_checkpoint(path, model, optimizer, epoch, best_miou, args):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "best_miou": best_miou,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
