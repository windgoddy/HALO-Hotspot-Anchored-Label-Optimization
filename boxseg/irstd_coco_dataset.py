"""Custom mmdet dataset for the reported BoxInst and Box2Mask comparisons.

Evaluate with the same mIoU / Pd / Fa metrics as HALO every epoch.

Per test image: union predicted instance masks (score > SCORE_THR) and GT masks
into binary maps, resize to 256 (HALO base_size), feed HALO's mIoU / PD_FA.

Accepts both common instance-result formats:
  CondInst: res = (bbox_result, segm_result), segm_result = list[list[RLE/array]] per class
  SOLOv2  : res = (bbox_result, (mask_results, score_results)) per-class lists of (H,W) tensors
"""

import importlib.util
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from mmdet.datasets import DATASETS, CocoDataset

_HALO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("halo_metrics", os.path.join(_HALO_DIR, "metrics.py"))
_halo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_halo)
HaloMIoU, HaloPDFA = _halo.mIoU, _halo.PD_FA

EVAL_SIZE = 256
SCORE_THR = 0.05
MIOU_THR = 0.3


def _to_mask(seg, h, w):
    """Decode one instance (RLE dict / np array / torch tensor) to (h, w) bool, or None."""
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        m = mask_utils.decode(rle)
    elif isinstance(seg, torch.Tensor):
        m = seg.detach().cpu().numpy()
    else:
        m = np.asarray(seg)
    m = np.squeeze(m)
    if m.ndim != 2 or m.size == 0:
        return None
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0


def _iter_pred(res):
    """Yield (mask, score) for each predicted instance."""
    if isinstance(res, tuple) and len(res) == 2:
        bbox_result, segm_result = res
    else:
        bbox_result, segm_result = None, res
    # Some instance segmenters return (mask_results, score_results).
    if (isinstance(segm_result, tuple) and len(segm_result) == 2
            and isinstance(segm_result[0], (list, tuple))):
        mask_results, score_results = segm_result
        for c in range(len(mask_results)):
            masks_c = mask_results[c]
            scores_c = score_results[c] if c < len(score_results) else None
            if masks_c is None:
                continue
            for i in range(len(masks_c)):
                s = float(scores_c[i]) if (scores_c is not None and i < len(scores_c)) else 1.0
                yield masks_c[i], s
        return
    # CondInst standard: segm_result = list[list[mask]] per class; score from bbox_result
    for c in range(len(segm_result)):
        masks_c = segm_result[c]
        bbox_c = bbox_result[c] if (isinstance(bbox_result, list) and c < len(bbox_result)) else None
        for i in range(len(masks_c)):
            s = float(bbox_c[i, 4]) if (bbox_c is not None and i < len(bbox_c)) else 1.0
            yield masks_c[i], s


@DATASETS.register_module()
class IRSTDCocoDataset(CocoDataset):
    CLASSES = ("target",)

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        miou = HaloMIoU(1, threshold=MIOU_THR)
        pdfa = HaloPDFA(1, bins=10, img_size=EVAL_SIZE)
        d_inst = d_kept = d_ppx = d_gpx = 0
        for idx in range(len(self)):
            info = self.data_infos[idx]
            H, W = info["height"], info["width"]

            pred = np.zeros((H, W), dtype=np.uint8)
            for seg, score in _iter_pred(results[idx]):
                d_inst += 1
                if score < SCORE_THR:
                    continue
                d_kept += 1
                mm = _to_mask(seg, H, W)
                if mm is not None:
                    pred |= mm

            ann_ids = self.coco.get_ann_ids(img_ids=[info["id"]])
            gt = np.zeros((H, W), dtype=np.uint8)
            for a in self.coco.load_anns(ann_ids):
                mm = _to_mask(a["segmentation"], H, W)
                if mm is not None:
                    gt |= mm

            d_ppx += int(pred.sum())
            d_gpx += int(gt.sum())
            pr = cv2.resize(pred.astype(np.uint8), (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST)
            gr = cv2.resize(gt.astype(np.uint8), (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST)

            gt_t = torch.from_numpy(gr.astype(np.float32)).view(1, 1, EVAL_SIZE, EVAL_SIZE)
            logit = torch.from_numpy(((pr.astype(np.float32) * 2) - 1) * 20).view(1, 1, EVAL_SIZE, EVAL_SIZE)
            miou.update(logit, gt_t)
            pdfa.update(torch.from_numpy(pr.astype(np.float32) * 255).view(1, 1, EVAL_SIZE, EVAL_SIZE), gt_t)

        _, iou = miou.get()
        fa, pd = pdfa.get(len(self))
        out = {"mIoU": float(iou), "Pd": float(pd[5]), "Fa": float(fa[5])}
        if logger is not None:
            from mmcv.utils import print_log
            print_log(f"IRSTD eval (256, score>{SCORE_THR}): "
                      f"mIoU={out['mIoU']:.4f}  Pd={out['Pd']:.4f}  Fa={out['Fa']:.3e}", logger=logger)
            print_log(f"[dbg] pred_inst={d_inst} kept={d_kept} pred_px={d_ppx} gt_px={d_gpx}", logger=logger)
        return out
