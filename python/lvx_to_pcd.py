"""
lvx_to_pcd.py
=============
Convert Livox .lvx binary files (v1.0 / v1.1) to ASCII .pcd files using
Python's struct module to parse the raw binary format directly.

Author : Ahmed Abdelkader
Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
License: MIT

Usage
-----
    # Convert a whole directory
    python lvx_to_pcd.py --input data/raw/ --output data/pcd/

    # Convert a single file
    python lvx_to_pcd.py --input data/raw/scan.lvx --output data/pcd/scan.pcd

LVX Binary Format (little-endian, packed structs)
-------------------------------------------------
Public Header Block  (24 bytes)
  signature   : 16 bytes  ("livox_tech\\0...")
  version     : 4 bytes   (major, minor, patch, build)
  magic_code  : 4 bytes   (0xAC0EA767)

Device Info Block
  device_count : 1 byte
  per device   : 59 bytes each
    lidar_sn   : 16 bytes
    hub_sn     : 16 bytes
    device_idx : 4 bytes  (uint32)
    device_type: 4 bytes  (uint32)
    extrinsic  : 1 byte   (bool)
    roll/pitch/yaw : 3 x float32  (degrees)
    x/y/z      : 3 x float32  (metres)

Data Block  (repeated frames until EOF)
  Frame header: 24 bytes
    current_offset : uint64
    next_offset    : uint64
    frame_index    : uint64
  Inside each frame -- repeated packets until next_offset:
    Packet header: 19 bytes
      device_index   : uint8
      version        : uint8
      slot_id        : uint8
      lidar_id       : uint8
      reserved       : uint8
      err_code       : uint32
      timestamp_type : uint8
      data_type      : uint8
      timestamp      : 8 bytes (raw)
    Point payload (depends on data_type):
      type 0 -- 100 x LivoxRawPoint (14 bytes):
                int32 x, y, z (millimetres); uint8 reflectivity, tag
      type 2 -- 100 x LivoxExtendRawPoint (16 bytes):
                float32 x, y, z (metres); uint8 reflectivity, tag; uint16 pad
"""

import argparse
import struct
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAGIC             = 0xAC0EA767
PUBLIC_HEADER_FMT = '<16sBBBBI'       # signature(16) + 4 version bytes + magic = 24 bytes
DEVICE_INFO_FMT   = '<16s16sIIBffffff'  # 59 bytes per device
FRAME_HEADER_FMT  = '<QQQ'             # 3 x uint64 = 24 bytes
PKG_HEADER_FMT    = '<BBBBBIB8s'       # 19 bytes
POINT_T0_FMT      = '<iiiBB'           # 14 bytes: x,y,z int32 (mm), refl, tag
POINT_T2_FMT      = '<fffBBH'          # 16 bytes: x,y,z float32 (m), refl, tag, pad
POINTS_PER_PACKET = 100


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_header(data: bytes, offset: int):
    """Parse the 24-byte public header; validate magic number."""
    size = struct.calcsize(PUBLIC_HEADER_FMT)
    unpacked = struct.unpack_from(PUBLIC_HEADER_FMT, data, offset)
    magic = unpacked[-1]
    if magic != MAGIC:
        raise ValueError(
            f"LVX magic mismatch: got 0x{magic:08X}, expected 0x{MAGIC:08X}. "
            "Is this a valid Livox .lvx file?"
        )
    # return (major_version, minor_version, new_offset)
    return unpacked[1], unpacked[2], offset + size


def _parse_devices(data: bytes, offset: int):
    """Parse device info block; return list of device dicts and new offset."""
    dev_count = data[offset]
    offset += 1
    sz = struct.calcsize(DEVICE_INFO_FMT)
    devices = []
    for _ in range(dev_count):
        f = struct.unpack_from(DEVICE_INFO_FMT, data, offset)
        devices.append({
            'lidar_sn':    f[0].rstrip(b'\x00').decode('ascii', errors='replace'),
            'device_type': f[3],
            'extrinsic':   bool(f[4]),
            'roll':  f[5], 'pitch': f[6], 'yaw': f[7],
            'x':     f[8], 'y':    f[9], 'z':   f[10],
        })
        offset += sz
    return devices, offset


def _read_t0(data: bytes, offset: int):
    """Parse 100 LivoxRawPoint (type 0) records; coordinates in millimetres."""
    sz  = struct.calcsize(POINT_T0_FMT)
    pts = []
    for _ in range(POINTS_PER_PACKET):
        x_mm, y_mm, z_mm, refl, _tag = struct.unpack_from(POINT_T0_FMT, data, offset)
        # Convert mm → m; normalise reflectivity to [0, 1]
        pts.append([x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3, refl / 255.0])
        offset += sz
    return np.array(pts, dtype=np.float32), offset


def _read_t2(data: bytes, offset: int):
    """Parse 100 LivoxExtendRawPoint (type 2) records; coordinates in metres."""
    sz  = struct.calcsize(POINT_T2_FMT)
    pts = []
    for _ in range(POINTS_PER_PACKET):
        x, y, z, refl, _tag, _pad = struct.unpack_from(POINT_T2_FMT, data, offset)
        pts.append([x, y, z, refl / 255.0])
        offset += sz
    return np.array(pts, dtype=np.float32), offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_lvx(path: str) -> np.ndarray:
    """
    Parse a Livox .lvx file using struct.

    Parameters
    ----------
    path : str
        Path to the .lvx file.

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x_m, y_m, z_m, reflectivity_0_to_1]
        Returns a zero-row array if the file contains no valid points.
    """
    with open(path, 'rb') as fh:
        data = fh.read()

    offset = 0
    _, _, offset = _parse_header(data, offset)
    _devices, offset = _parse_devices(data, offset)

    all_points: List[np.ndarray] = []
    frame_sz = struct.calcsize(FRAME_HEADER_FMT)
    pkg_sz   = struct.calcsize(PKG_HEADER_FMT)

    while offset + frame_sz <= len(data):
        _cur_off, nxt_off, _frame_idx = struct.unpack_from(FRAME_HEADER_FMT, data, offset)
        offset   += frame_sz
        frame_end = int(nxt_off) if 0 < nxt_off <= len(data) else len(data)

        while offset + pkg_sz <= frame_end:
            (_dev_idx, _ver, _slot, _lid, _res,
             _err, _ts_type, dtype, _ts) = struct.unpack_from(PKG_HEADER_FMT, data, offset)
            offset += pkg_sz

            try:
                if dtype == 0:
                    pts, offset = _read_t0(data, offset)
                elif dtype == 2:
                    pts, offset = _read_t2(data, offset)
                else:
                    # Unknown data type — skip the rest of this frame
                    offset = frame_end
                    break

                # Filter out invalid / origin points
                valid = (
                    np.isfinite(pts).all(axis=1) &
                    (np.linalg.norm(pts[:, :3], axis=1) > 1e-3)
                )
                if valid.any():
                    all_points.append(pts[valid])

            except struct.error:
                # Truncated packet — stop parsing this frame
                break

        # Advance to the start of the next frame
        offset = max(offset, frame_end)

    if not all_points:
        return np.zeros((0, 4), dtype=np.float32)
    return np.vstack(all_points)


def write_pcd(points: np.ndarray, out_path: str) -> None:
    """
    Write a point cloud to an ASCII .pcd file (PCL v0.7 format).

    Parameters
    ----------
    points   : np.ndarray, shape (N, 4) — [x, y, z, intensity]
    out_path : str — destination file path (parent dirs are created if needed)
    """
    n = len(points)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write("# .PCD v0.7 - Point Cloud Data\n")
        fh.write("# Generated by lvx_to_pcd.py (Ahmed Abdelkader)\n")
        fh.write("VERSION 0.7\n")
        fh.write("FIELDS x y z intensity\n")
        fh.write("SIZE 4 4 4 4\n")
        fh.write("TYPE F F F F\n")
        fh.write("COUNT 1 1 1 1\n")
        fh.write(f"WIDTH {n}\n")
        fh.write("HEIGHT 1\n")
        fh.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        fh.write(f"POINTS {n}\n")
        fh.write("DATA ascii\n")
        for x, y, z, intensity in points:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Livox .lvx binary files to ASCII .pcd (struct-based parser)"
    )
    parser.add_argument('--input',  required=True,
                        help=".lvx file or directory containing .lvx files")
    parser.add_argument('--output', required=True,
                        help="Output .pcd file (single) or output directory (batch)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print per-file statistics")
    args = parser.parse_args()

    in_p = Path(args.input)

    if in_p.is_dir():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        lvx_files = sorted(in_p.glob('**/*.lvx'))
        if not lvx_files:
            print(f"No .lvx files found in {in_p}")
            return
        print(f"Converting {len(lvx_files)} file(s) ...")
        for lvx in lvx_files:
            pcd = out_dir / (lvx.stem + '.pcd')
            pts = parse_lvx(str(lvx))
            write_pcd(pts, str(pcd))
            if args.verbose:
                print(f"  {lvx.name:40s}  ->  {pcd.name}  ({len(pts):,} pts)")
        print("Done.")
    else:
        pts = parse_lvx(str(in_p))
        write_pcd(pts, args.output)
        print(f"Wrote {len(pts):,} points  ->  {args.output}")


if __name__ == '__main__':
    main()
