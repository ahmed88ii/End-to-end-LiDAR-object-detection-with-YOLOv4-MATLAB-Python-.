# Author : Ahmed Abdelkader
# Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
# License: MIT
"""
pcd_to_bev.py
=============
Generate Bird's-Eye-View (BEV) images from .pcd point clouds.

BEV image is a 3-channel uint8 PNG:
  Channel 0 (R) : normalised max height     in the voxel column
  Channel 1 (G) : mean reflectivity / intensity
  Channel 2 (B) : normalised point density  (saturates at 64 pts/pixel)

Default spatial extent (matches MATLAB generate_bev.m and the YAML config):
    x_range    : [0, 70.4] m   (forward)  -> 704 pixels
    y_range    : [-40, 40] m   (lateral)  -> 800 pixels
    z_range    : [-3, 1] m     (height band)
    resolution : 0.1 m/pixel

Usage
-----
    python python/pcd_to_bev.py --input data/pcd/ --output data/bev/images/ \
                                 --config configs/yolov4_lidar.yaml
    python python/pcd_to_bev.py --input scan.pcd  --output scan_bev.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
import yaml

# ---------------------------------------------------------------------------
# Default BEV parameters — can be overridden via YAML config
# ---------------------------------------------------------------------------
DEFAULTS: Dict = dict(
    x_range      = (0.0, 70.4),
    y_range      = (-40.0, 40.0),
    z_range      = (-3.0, 1.0),
    resolution   = 0.1,
    height_scale = 4.0,   # z_range span for normalisation
)

MAX_DENSITY = 64  # point count per pixel above which density saturates to 1.0


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """
    Load BEV parameters from a YAML config file.

    Falls back to DEFAULTS for any missing key.

    Parameters
    ----------
    config_path : str
        Path to the YAML file. Pass an empty string to use all defaults.

    Returns
    -------
    dict
        Keys: x_range, y_range, z_range, resolution, height_scale.
    """
    params = DEFAULTS.copy()
    if not config_path or not Path(config_path).exists():
        return params

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    bev = cfg.get('bev', {})
    for key in ('x_range', 'y_range', 'z_range'):
        if key in bev:
            params[key] = tuple(bev[key])
    for key in ('resolution', 'height_scale'):
        if key in bev:
            params[key] = float(bev[key])
    return params


# ---------------------------------------------------------------------------
# PCD loader
# ---------------------------------------------------------------------------

def load_pcd(path: str) -> np.ndarray:
    """
    Read an ASCII PCD file (PCL v0.7 format).

    Falls back to open3d for binary PCDs.

    Parameters
    ----------
    path : str
        Path to the .pcd file.

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x, y, z, intensity].
        Returns a zero-row array if the file is empty or unreadable.
    """
    with open(path, 'r', errors='ignore') as fh:
        lines = fh.readlines()

    data_line = 0
    data_type = 'ascii'

    for i, line in enumerate(lines):
        tok = line.strip().lower()
        if tok.startswith('data'):
            data_type = tok.split()[1]
            data_line = i + 1
            break

    if data_type != 'ascii':
        import open3d as o3d   # lazy import — not required for ASCII PCDs
        pcd   = o3d.io.read_point_cloud(path)
        xyz   = np.asarray(pcd.points, dtype=np.float32)
        inten = np.zeros((len(xyz), 1), dtype=np.float32)
        return np.hstack([xyz, inten])

    rows = []
    for line in lines[data_line:]:
        vals = line.strip().split()
        if len(vals) < 3:
            continue
        try:
            row = [float(v) for v in vals[:4]] if len(vals) >= 4 \
                  else [float(v) for v in vals[:3]] + [0.0]
            rows.append(row)
        except ValueError:
            continue

    return np.array(rows, dtype=np.float32) if rows \
           else np.zeros((0, 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# BEV projection — fully vectorised with numpy
# ---------------------------------------------------------------------------

def point_cloud_to_bev(points: np.ndarray, params: Dict) -> np.ndarray:
    """
    Project a point cloud onto a 3-channel BEV image.

    All per-point operations are vectorised with NumPy (np.maximum.at and
    np.add.at) — no Python-level loops over individual points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Point cloud with columns [x, y, z, intensity].
    params : dict
        BEV parameters from load_config().

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
        Three-channel BEV image: [height, intensity, density].
    """
    x_min, x_max = params['x_range']
    y_min, y_max = params['y_range']
    z_min, z_max = params['z_range']
    res   = float(params['resolution'])
    h_scl = float(params['height_scale'])

    W = int(round((x_max - x_min) / res))
    H = int(round((y_max - y_min) / res))

    # --- Spatial ROI filter ----------------------------------------------
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]

    if len(pts) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # --- Pixel indices ---------------------------------------------------
    xi      = np.clip(np.floor((pts[:, 0] - x_min) / res).astype(np.int32), 0, W - 1)
    yi      = np.clip(np.floor((pts[:, 1] - y_min) / res).astype(np.int32), 0, H - 1)
    yi_flip = H - 1 - yi          # flip y so that forward direction is image-up

    z_norm    = np.clip((pts[:, 2] - z_min) / h_scl, 0.0, 1.0)
    intensity = np.clip(pts[:, 3],             0.0, 1.0)

    # --- Vectorised accumulation -----------------------------------------
    height_map  = np.zeros((H, W), dtype=np.float32)
    inten_sum   = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)

    # np.maximum.at: unbuffered max — correct for multiple points per pixel
    np.maximum.at(height_map, (yi_flip, xi), z_norm)
    np.add.at(inten_sum,      (yi_flip, xi), intensity)
    np.add.at(density_map,    (yi_flip, xi), 1.0)

    # Mean intensity per occupied pixel
    occupied   = density_map > 0
    inten_mean = np.where(occupied, inten_sum / np.maximum(density_map, 1.0), 0.0)

    # Density normalised to [0, 1]; saturates at MAX_DENSITY pts/pixel
    density_norm = np.minimum(density_map / MAX_DENSITY, 1.0)

    # --- Pack into uint8 RGB image ---------------------------------------
    bev = np.stack([
        np.clip(height_map, 0.0, 1.0),
        np.clip(inten_mean, 0.0, 1.0),
        density_norm,
    ], axis=-1)
    return (bev * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# File-level helper
# ---------------------------------------------------------------------------

def convert_file(pcd_path: str, out_path: str, params: Dict) -> None:
    """Convert a single PCD file to a BEV PNG."""
    points = load_pcd(pcd_path)
    bev    = point_cloud_to_bev(points, params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, bev)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .pcd point clouds to 3-channel BEV PNG images"
    )
    parser.add_argument('--input',  required=True,
                        help=".pcd file or directory of .pcd files")
    parser.add_argument('--output', required=True,
                        help="Output .png file or output directory")
    parser.add_argument('--config', default='configs/yolov4_lidar.yaml',
                        help="YAML config file (optional)")
    args = parser.parse_args()

    params = load_config(args.config)
    in_p   = Path(args.input)

    if in_p.is_dir():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        pcd_files = sorted(in_p.glob('**/*.pcd'))
        if not pcd_files:
            print(f"No .pcd files found in {in_p}")
            return
        print(f"Converting {len(pcd_files)} file(s) ...")
        for pcd in pcd_files:
            out = out_dir / (pcd.stem + '.png')
            convert_file(str(pcd), str(out), params)
            print(f"  {pcd.name} -> {out.name}")
    else:
        convert_file(str(in_p), args.output, params)
        print(f"Saved BEV to {args.output}")


if __name__ == '__main__':
    main()
