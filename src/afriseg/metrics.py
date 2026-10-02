"""Per-case Dice and HD95 with BraTS conventions for empty structures."""
from __future__ import annotations

import numpy as np

# BraTS convention: HD95 when exactly one of prediction / reference is empty
HD95_MAX = 373.13


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return HD95_MAX
    import torch
    from monai.metrics import compute_hausdorff_distance

    p = torch.from_numpy(pred[None, None].astype(np.float32))
    g = torch.from_numpy(gt[None, None].astype(np.float32))
    return float(compute_hausdorff_distance(p, g, include_background=True, percentile=95,
                                            spacing=list(spacing)).item())


def region_scores(pred_regions: np.ndarray, gt_regions: np.ndarray, with_hd: bool = True) -> dict:
    from .labels import REGIONS

    out = {}
    for i, r in enumerate(REGIONS):
        out[f"dice_{r}"] = dice(pred_regions[i], gt_regions[i])
        if with_hd:
            out[f"hd95_{r}"] = hd95(pred_regions[i], gt_regions[i])
    return out
