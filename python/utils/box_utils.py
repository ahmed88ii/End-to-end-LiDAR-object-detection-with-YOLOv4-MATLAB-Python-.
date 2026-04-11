# Author : Ahmed Abdelkader
# Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
# License: MIT
"""
box_utils.py
============
Bounding-box geometry utilities: BEV IoU (2-D rotated rectangle overlap)
and 3-D volumetric IoU (BEV polygon area x height overlap decomposition).

Note on naming
--------------
The evaluation pipeline in this repo operates exclusively in 2-D
Bird's-Eye-View (BEV) space using axis-aligned boxes derived from YOLO
labels.  The `bev_iou` function handles the rotated-rectangle case
(useful when yaw angles are available), while `bev_iou_aligned` is a
faster special case for axis-aligned boxes.  The `iou3d` function is
provided for completeness but is not used by the main evaluation scripts.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal polygon helpers
# ---------------------------------------------------------------------------

def _poly_area(poly: np.ndarray) -> float:
    """
    Signed polygon area via the shoelace formula.

    Parameters
    ----------
    poly : np.ndarray, shape (N, 2)
        Convex polygon vertices in order.

    Returns
    -------
    float — absolute area.
    """
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _sutherland_hodgman(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """
    Clip a convex polygon (subject) against another convex polygon (clip).

    Parameters
    ----------
    subject : np.ndarray, shape (N, 2) — polygon to clip.
    clip    : np.ndarray, shape (M, 2) — clipping polygon.

    Returns
    -------
    np.ndarray, shape (K, 2) — intersection polygon vertices,
    or an empty array if there is no intersection.
    """
    def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return float((b[0] - a[0]) * (p[1] - a[1])
                     - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

    def _intersect(p1: np.ndarray, p2: np.ndarray,
                   p3: np.ndarray, p4: np.ndarray) -> np.ndarray | None:
        d1 = p2 - p1
        d2 = p4 - p3
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-10:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
        return p1 + t * d1

    output = list(subject.astype(float))
    n_clip = len(clip)

    for i in range(n_clip):
        if not output:
            return np.empty((0, 2), dtype=np.float64)
        a = clip[i].astype(float)
        b = clip[(i + 1) % n_clip].astype(float)
        inp, output = output, []
        for j in range(len(inp)):
            cur  = inp[j]
            prev = inp[j - 1]
            if _inside(cur, a, b):
                if not _inside(prev, a, b):
                    pt = _intersect(prev, cur, a, b)
                    if pt is not None:
                        output.append(pt)
                output.append(cur)
            elif _inside(prev, a, b):
                pt = _intersect(prev, cur, a, b)
                if pt is not None:
                    output.append(pt)

    return np.array(output, dtype=np.float64) if output \
           else np.empty((0, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bev_corners(cx: float, cy: float, length: float,
                width: float, yaw: float) -> np.ndarray:
    """
    Compute the four corners of a rotated rectangle in the BEV plane.

    Parameters
    ----------
    cx, cy  : float — centre coordinates.
    length  : float — extent along the local x-axis.
    width   : float — extent along the local y-axis.
    yaw     : float — rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (4, 2).
    """
    hl, hw = length / 2.0, width / 2.0
    c, s   = np.cos(yaw), np.sin(yaw)
    # Four corners in local frame, rotated and translated
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([cx, cy])


def bev_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    BEV (top-down) IoU for two rotated rectangles.

    Parameters
    ----------
    b1, b2 : array-like, shape (5,) — [cx, cy, length, width, yaw_rad].

    Returns
    -------
    float in [0, 1].
    """
    c1 = bev_corners(*b1)
    c2 = bev_corners(*b2)
    inter = _sutherland_hodgman(c1, c2)
    if len(inter) < 3:
        return 0.0
    ia    = _poly_area(inter)
    union = b1[2] * b1[3] + b2[2] * b2[3] - ia
    return float(ia / union) if union > 0.0 else 0.0


def bev_iou_aligned(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    Fast axis-aligned BEV IoU for [x, y, w, h] boxes (pixel coordinates).

    Used by evaluate_detector.m-compatible evaluation in Python.

    Parameters
    ----------
    b1, b2 : array-like, shape (4,) — [x, y, width, height] (top-left origin).

    Returns
    -------
    float in [0, 1].
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return float(inter / union) if union > 0.0 else 0.0


def iou3d(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    Full 3-D volumetric IoU for rotated boxes.

    Decomposed as: IoU_3D = (BEV_inter_area x height_overlap) / union_volume.

    Parameters
    ----------
    b1, b2 : array-like, shape (7,) — [cx, cy, cz, length, width, height, yaw_rad].

    Returns
    -------
    float in [0, 1].

    Note
    ----
    This function is provided for completeness.  The main evaluation in this
    repo uses axis-aligned 2-D BEV IoU (bev_iou_aligned), not iou3d, because
    the YOLO BEV labels do not contain height or yaw annotations.
    """
    b1, b2 = np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)

    # Height overlap
    h_overlap = max(0.0, min(b1[2] + b1[5], b2[2] + b2[5])
                    - max(b1[2], b2[2]))
    if h_overlap <= 0.0:
        return 0.0

    bev_ov = bev_iou(b1[[0, 1, 3, 4, 6]], b2[[0, 1, 3, 4, 6]])
    if bev_ov <= 0.0:
        return 0.0

    # Recover intersection BEV area from IoU
    a1, a2 = b1[3] * b1[4], b2[3] * b2[4]
    inter_bev = bev_ov * (a1 + a2) / (1.0 + bev_ov)
    inter_vol = inter_bev * h_overlap
    union_vol = a1 * b1[5] + a2 * b2[5] - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0.0 else 0.0
