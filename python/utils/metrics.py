# Author : Ahmed Abdelkader
# Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
# License: MIT
"""
metrics.py
==========
Average Precision (AP) and Average Orientation Similarity (AOS)
via 11-point PASCAL VOC interpolation.

Evaluation scope
----------------
The main evaluation in this repo is BEV-based (2-D Bird's-Eye-View),
not full 3-D.  YOLO labels contain no height or yaw annotations, so:
  - IoU matching uses axis-aligned 2-D BEV boxes (bev_iou_aligned).
  - AOS uses delta_yaw = 0 for all matched detections, making AOS == AP.
This is consistent with the MATLAB evaluate_detector.m script.

For full 3-D IoU matching (when 3-D annotations are available), pass
iou_fn=iou3d from box_utils.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .box_utils import bev_iou_aligned


# ---------------------------------------------------------------------------
# Per-detection orientation similarity
# ---------------------------------------------------------------------------

def orientation_similarity(delta_theta: float) -> float:
    """
    Scalar orientation similarity score (Geiger et al., CVPR 2012).

    Parameters
    ----------
    delta_theta : float — angular error in radians (pred_yaw - gt_yaw).

    Returns
    -------
    float in [0, 1].  1.0 = perfect alignment, 0.0 = 180-degree error.
    """
    return float((1.0 + np.cos(delta_theta)) / 2.0)


# ---------------------------------------------------------------------------
# AP / AOS
# ---------------------------------------------------------------------------

def compute_ap_11point(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    11-point PASCAL VOC Average Precision interpolation.

    Parameters
    ----------
    recall    : np.ndarray, shape (N,) — cumulative recall, ascending.
    precision : np.ndarray, shape (N,) — precision at each recall level.

    Returns
    -------
    float in [0, 1].
    """
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        mask = recall >= thr
        ap  += float(np.max(precision[mask])) if mask.any() else 0.0
    return ap / 11.0


def compute_ap_aos(
    pred_boxes:    np.ndarray,
    pred_scores:   np.ndarray,
    pred_angles:   np.ndarray,
    gt_boxes:      np.ndarray,
    gt_angles:     np.ndarray,
    iou_threshold: float = 0.5,
    iou_fn:        Callable = bev_iou_aligned,
) -> Tuple[float, float]:
    """
    Compute AP and AOS for one class via 11-point PASCAL VOC interpolation.

    Parameters
    ----------
    pred_boxes   : np.ndarray, shape (N, 4) — predicted boxes [x,y,w,h].
                   Pass shape (N, 5) for rotated BEV or (N, 7) for 3-D boxes,
                   matching the signature of iou_fn.
    pred_scores  : np.ndarray, shape (N,) — confidence scores.
    pred_angles  : np.ndarray, shape (N,) — predicted yaw (radians).
                   Pass zeros if yaw is not available (gives AOS == AP).
    gt_boxes     : np.ndarray, shape (M, 4) — ground-truth boxes.
    gt_angles    : np.ndarray, shape (M,) — GT yaw (radians).
    iou_threshold: float — minimum IoU to count as a true positive.
    iou_fn       : callable(b1, b2) -> float — IoU function.
                   Default: bev_iou_aligned (axis-aligned 2-D BEV).

    Returns
    -------
    ap  : float in [0, 1]
    aos : float in [0, 1]

    Notes
    -----
    Detections are matched greedily in descending confidence order.
    Each GT box is matched at most once.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0

    n_pred = len(pred_boxes)
    n_gt   = len(gt_boxes)

    tp  = np.zeros(n_pred, dtype=np.float64)
    fp  = np.zeros(n_pred, dtype=np.float64)
    os  = np.zeros(n_pred, dtype=np.float64)

    matched: set[int] = set()

    # Sort by descending confidence once
    order = np.argsort(-pred_scores)

    for idx in order:
        best_iou, best_g = 0.0, -1

        for g in range(n_gt):
            if g in matched:
                continue
            iou_val = iou_fn(pred_boxes[idx], gt_boxes[g])
            if iou_val > best_iou:
                best_iou, best_g = iou_val, g

        if best_iou >= iou_threshold and best_g >= 0:
            tp[idx] = 1.0
            matched.add(best_g)
            os[idx] = orientation_similarity(
                float(pred_angles[idx]) - float(gt_angles[best_g])
            )
        else:
            fp[idx] = 1.0

    # Accumulate in confidence-descending order
    tp_ord = tp[order]
    fp_ord = fp[order]
    os_ord = os[order]

    cum_tp = np.cumsum(tp_ord)
    cum_fp = np.cumsum(fp_ord)
    cum_os = np.cumsum(os_ord)

    recall    = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    os_score  = cum_os / np.maximum(cum_tp + cum_fp, 1e-9)

    ap  = compute_ap_11point(recall, precision)
    aos = compute_ap_11point(recall, os_score)

    return ap, aos
