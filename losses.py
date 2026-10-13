"""Loss functions for HALO."""

import torch


def soft_iou_loss(logits, targets, smooth=0.8):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou


def SoftIoULoss(logits, targets):
    return soft_iou_loss(logits, targets)
