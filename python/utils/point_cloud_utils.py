# Author : Ahmed Abdelkader
# Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
# License: MIT
"""
point_cloud_utils.py
====================
Point cloud loading, ROI filtering, voxel downsampling, and coordinate
transforms.  All operations are vectorised with NumPy — no Python-level
loops over individual points.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pcd_numpy(path: str) -> np.ndarray:
    """
    Load a PCD file via open3d and return a structured NumPy array.

    Parameters
    ----------
    path : str — path to the .pcd file (ASCII or binary).

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x, y, z, intensity].
        Intensity is taken from the first colour channel if available,
        otherwise set to zero.
    """
    import open3d as o3d
    pcd   = o3d.io.read_point_cloud(path)
    xyz   = np.asarray(pcd.points, dtype=np.float32)
    inten = (np.asarray(pcd.colors, dtype=np.float32)[:, :1]
             if pcd.has_colors()
             else np.zeros((len(xyz), 1), dtype=np.float32))
    return np.hstack([xyz, inten])


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_roi(
    pts: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> np.ndarray:
    """
    Keep only points inside the specified 3-D region of interest.

    Parameters
    ----------
    pts     : np.ndarray, shape (N, 4+) — point cloud.
    x_range : (x_min, x_max)
    y_range : (y_min, y_max)
    z_range : (z_min, z_max)

    Returns
    -------
    np.ndarray, shape (M, 4+) — filtered subset.
    """
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] < z_range[1])
    )
    return pts[mask]


def remove_ground(pts: np.ndarray, z_threshold: float = -1.5) -> np.ndarray:
    """
    Remove points at or below z_threshold (simple height-based ground removal).

    Parameters
    ----------
    pts         : np.ndarray, shape (N, 4+)
    z_threshold : float — points with z <= this value are discarded.

    Returns
    -------
    np.ndarray, shape (M, 4+).
    """
    return pts[pts[:, 2] > z_threshold]


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel-grid downsampling: replace all points in each voxel with their
    centroid.  Fully vectorised — no Python loops over voxels.

    Parameters
    ----------
    pts        : np.ndarray, shape (N, D) where D >= 3.
    voxel_size : float — edge length of each cubic voxel.

    Returns
    -------
    np.ndarray, shape (K, D), dtype float32 — one centroid per occupied voxel.
    """
    if len(pts) == 0:
        return pts

    # Assign each point to a voxel by integer index
    voxel_idx = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Encode the 3-D index as a single integer for fast grouping
    # (shift to non-negative then linearise)
    offset = voxel_idx.min(axis=0)
    shifted = voxel_idx - offset
    span   = shifted.max(axis=0) + 1
    linear = (shifted[:, 0] * span[1] * span[2]
              + shifted[:, 1] * span[2]
              + shifted[:, 2])

    # Sort by voxel key so equal keys are contiguous
    sort_order = np.argsort(linear)
    sorted_lin = linear[sort_order]
    sorted_pts = pts[sort_order]

    # Find voxel boundaries with np.unique
    _, first_idx, counts = np.unique(
        sorted_lin, return_index=True, return_counts=True
    )

    # Compute centroid per voxel using cumulative sum trick (vectorised)
    cum = np.cumsum(sorted_pts, axis=0)
    # Sum within each voxel: sum[end] - sum[start-1]
    end_idx = first_idx + counts - 1
    total   = cum[end_idx]
    total[1:] -= cum[first_idx[1:] - 1]   # subtract preceding cumulative sum

    centroids = (total / counts[:, np.newaxis]).astype(np.float32)
    return centroids


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def apply_extrinsic(
    pts:   np.ndarray,
    roll:  float,
    pitch: float,
    yaw:   float,
    tx:    float,
    ty:    float,
    tz:    float,
) -> np.ndarray:
    """
    Apply a rigid extrinsic calibration transform to a point cloud.

    Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (intrinsic ZYX).

    Parameters
    ----------
    pts   : np.ndarray, shape (N, 4+) — point cloud (xyz in first 3 cols).
    roll  : float — rotation about X in radians.
    pitch : float — rotation about Y in radians.
    yaw   : float — rotation about Z in radians.
    tx, ty, tz : float — translation in metres.

    Returns
    -------
    np.ndarray, shape (N, 4+) — transformed point cloud.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cr, -sr],
                   [0,  sr,  cr]], dtype=np.float32)
    Ry = np.array([[ cp, 0,  sp],
                   [  0, 1,   0],
                   [-sp, 0,  cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float32)

    R   = Rz @ Ry @ Rx
    t   = np.array([tx, ty, tz], dtype=np.float32)
    out = pts.copy()
    out[:, :3] = pts[:, :3] @ R.T + t
    return out
