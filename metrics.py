"""Metrics aligned with the final HALO experiments."""

import numpy as np
import torch
from skimage import measure


class ROCMetric:
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for i_bin in range(self.bins + 1):
            score_thresh = i_bin / self.bins
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(
                preds, labels, self.nclass, score_thresh
            )
            self.tp_arr[i_bin] += i_tp
            self.pos_arr[i_bin] += i_pos
            self.fp_arr[i_bin] += i_fp
            self.neg_arr[i_bin] += i_neg
            self.class_pos[i_bin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 1e-3)
        fp_rates = self.fp_arr / (self.neg_arr + 1e-3)
        recall = self.tp_arr / (self.pos_arr + 1e-3)
        precision = self.tp_arr / (self.class_pos + 1e-3)
        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)


class PD_FA:
    def __init__(self, nclass, bins, img_size=None):
        self.nclass = nclass
        self.bins = bins
        self.img_size = img_size
        self.reset()

    def update(self, preds, labels):
        preds_np = np.array(preds.cpu()).astype("float32")
        labels_np = np.array(labels.cpu()).astype("int64")
        h, w = preds_np.shape[-2], preds_np.shape[-1]
        preds_np = preds_np.reshape(-1, h, w)
        labels_np = labels_np.reshape(-1, h, w)
        self._last_area = h * w

        for i_bin in range(self.bins + 1):
            score_thresh = i_bin * (255 / self.bins)
            for pred_mask, label_mask in zip(preds_np, labels_np):
                pred_mask = (pred_mask > score_thresh).astype("int64")
                image = measure.label(pred_mask, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(label_mask, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[i_bin] += len(coord_label)
                image_area_total = []
                image_area_match = []
                distance_match = []

                for region in coord_image:
                    image_area_total.append(np.array(region.area))

                for label_region in coord_label:
                    centroid_label = np.array(list(label_region.centroid))
                    for idx in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[idx].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        if distance < 3:
                            distance_match.append(coord_image[idx].area)
                            image_area_match.append(coord_image[idx].area)
                            del coord_image[idx]
                            break

                dismatch = [x for x in image_area_total if x not in image_area_match]
                self.FA[i_bin] += np.sum(dismatch)
                self.PD[i_bin] += len(distance_match)

    def get(self, img_num):
        area = self.img_size * self.img_size if self.img_size is not None else self._last_area
        final_fa = self.FA / (area * img_num)
        final_pd = self.PD / self.target
        return final_fa, final_pd

    def reset(self):
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self._last_area = None


class mIoU:
    def __init__(self, nclass, threshold=0.3):
        self.nclass = nclass
        self.threshold = threshold
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels, threshold=self.threshold)
        inter, union = batch_intersection_union(preds, labels, self.nclass, threshold=self.threshold)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pix_acc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return pix_acc, iou.mean()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    intersection = predict * ((predict == target).float())
    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp
    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output, target, threshold=0.3):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    predict = (torch.sigmoid(output) > threshold).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, threshold=0.3):
    predict = (torch.sigmoid(output) > threshold).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    intersection = predict * ((predict == target).float())
    area_inter, _ = np.histogram(intersection.cpu(), bins=1, range=(1, 1))
    area_pred, _ = np.histogram(predict.cpu(), bins=1, range=(1, 1))
    area_lab, _ = np.histogram(target.cpu(), bins=1, range=(1, 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def calculate_mask_to_box_iou(pred_mask, gt_mask, threshold=0.3):
    if isinstance(pred_mask, torch.Tensor):
        pred_np = (torch.sigmoid(pred_mask) > threshold).float().detach().cpu().numpy()
    else:
        pred_np = (pred_mask > threshold).astype(np.float32)
    if isinstance(gt_mask, torch.Tensor):
        gt_np = (gt_mask > 0.5).float().detach().cpu().numpy()
    else:
        gt_np = (gt_mask > 0.5).astype(np.float32)

    batch_size = pred_np.shape[0]
    total_iou = 0.0
    valid_count = 0
    for i in range(batch_size):
        pred_box = _get_bounding_box(pred_np[i, 0])
        gt_box = _get_bounding_box(gt_np[i, 0])
        if gt_box is None:
            continue
        iou = 0.0 if pred_box is None else _compute_box_iou(pred_box, gt_box)
        total_iou += iou
        valid_count += 1
    return 0.0 if valid_count == 0 else total_iou / valid_count


def _get_bounding_box(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]


def _compute_box_iou(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    area_b = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-7)
