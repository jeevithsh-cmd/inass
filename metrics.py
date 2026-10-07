"""metrics.py — DSC, HD95, classification metrics and Grad-CAM IoU (Tables 3, 4, 9, 10)."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

REGIONS = ["TC", "WT", "ET"]     # channel order produced by ConvertToMultiChannelBasedOnBratsClassesd


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-8))


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric Hausdorff distance in mm (1 mm isotropic BraTS voxels)."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(np.linalg.norm(np.array(gt.shape) * np.array(spacing)))   # worst case
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u > 0 else 1.0


def seg_metrics(seg_logits: torch.Tensor, label: torch.Tensor) -> list[dict]:
    """Per-sample DSC / HD95 for TC, WT, ET."""
    pred = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
    gt = (label > 0.5).cpu().numpy()
    rows = []
    for b in range(pred.shape[0]):
        r = {}
        for c, name in enumerate(REGIONS):
            r[f"{name}_DSC"] = dice(pred[b, c], gt[b, c])
            r[f"{name}_HD95"] = hd95(pred[b, c], gt[b, c])
        rows.append(r)
    return rows


def cam_iou(cam_up: torch.Tensor, label: torch.Tensor, thr: float = 0.5) -> list[dict]:
    """Grad-CAM IoU against tumour regions, CAM in [0,1] thresholded at 0.5 (Sec. 5, Table 10)."""
    c = (cam_up[:, 0] > thr).cpu().numpy()
    gt = (label > 0.5).cpu().numpy()
    rows = []
    for b in range(c.shape[0]):
        r = {f"{n}_IoU": iou(c[b], gt[b, i]) for i, n in enumerate(REGIONS)}
        r["Mean_IoU"] = float(np.mean([r[f"{n}_IoU"] for n in REGIONS]))
        rows.append(r)
    return rows


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred = prob.argmax(1)
    out = {"Accuracy": 100 * accuracy_score(y_true, y_pred),
           "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    try:
        out["AUC_ROC"] = roc_auc_score(y_true, prob, multi_class="ovr", average="macro",
                                       labels=list(range(prob.shape[1])))
    except ValueError:
        out["AUC_ROC"] = float("nan")       # a class is absent from this subset
    return out
