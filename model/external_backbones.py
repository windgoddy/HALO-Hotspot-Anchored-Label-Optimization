"""Adapters for external IRSTD backbones (SCTransNet, MSHNet).

The official model code is cloned under ``HALO/external/`` (see scripts/setup_external_backbones.sh).
These thin wrappers expose the same interface HALO expects:

* input  : (B, in_channels, H, W) image tensor
* output : raw **logits** — a single (B,1,H,W) tensor when deep supervision is off,
           or a list of (B,1,H,W) tensors when on. HALO's ``soft_iou_loss`` applies
           sigmoid internally and the mIoU metric thresholds the raw logit, so every
           backbone must emit logits (not probabilities) to stay comparable to DNANet.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = os.path.normpath(os.path.join(_HERE, "..", "external"))
# Each official repo keeps its network under <repo>/model/. Add those dirs to the path
# so we can ``import SCTransNet`` / ``import MSHNet`` by their module file names.
for _sub in ("SCTransNet/model", "MSHNet/model"):
    _p = os.path.join(_EXT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class SCTransNetWrapper(nn.Module):
    """Official SCTransNet (IEEE TGRS 2024). Its forward returns sigmoid probabilities;
    we convert them back to logits so HALO's loss/metric convention is preserved."""

    def __init__(self, in_channels=1, deepsuper=True):
        super().__init__()
        from SCTransNet import SCTransNet as _SCTransNet, get_CTranS_config

        cfg = get_CTranS_config()
        self.net = _SCTransNet(
            cfg,
            n_channels=in_channels,
            n_classes=1,
            img_size=256,
            mode="train",          # 'train' makes it emit the deep-supervision tuple
            deepsuper=deepsuper,
        )
        self.deepsuper = deepsuper

    @staticmethod
    def _to_logit(p):
        return torch.logit(p.clamp(1e-6, 1.0 - 1e-6))

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            return [self._to_logit(o) for o in out]
        return self._to_logit(out)


class MSHNetWrapper(nn.Module):
    """Official MSHNet (CVPR 2024). Output conv has no activation -> already logits.
    We use the single-scale head (warm_flag=False) for a clean single-output backbone."""

    def __init__(self, in_channels=1):
        super().__init__()
        from MSHNet import MSHNet as _MSHNet

        self.net = _MSHNet(in_channels)

    def forward(self, x):
        _, output = self.net(x, False)  # warm_flag=False -> single full-resolution logit map
        return output
