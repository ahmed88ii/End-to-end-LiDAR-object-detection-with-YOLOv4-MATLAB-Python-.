# Author : Ahmed Abdelkader
# Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
# License: MIT
"""
visualize.py -- BEV image and 3-D point cloud visualisation tools.

Usage:
    python visualize.py bev --image data/bev/images/000001.png --label data/bev/labels/000001.txt
    python visualize.py pcd --pcd   data/pcd/000001.pcd
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

PALETTE = {
    "Car":        (0,   255,  0),
    "Pedestrian": (255, 128,  0),
    "Cyclist":    (0,   128, 255),
    "default":    (255,   0,  0),
}


def draw_boxes(img, boxes, names, scores=None):
    out = img.copy()
    for i, (b, cls) in enumerate(zip(boxes, names)):
        x, y, w, h = [int(v) for v in b]
        col = PALETTE.get(cls, PALETTE["default"])
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        lbl = cls if scores is None else "{} {:.2f}".format(cls, scores[i])
        cv2.putText(out, lbl, (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def load_yolo(label_path, img_h, img_w):
    boxes, cls_ids = [], []
    if not Path(label_path).exists():
        return boxes, cls_ids
    for ln in Path(label_path).read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        c  = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append([int((cx - bw / 2) * img_w),
                      int((cy - bh / 2) * img_h),
                      int(bw * img_w),
                      int(bh * img_h)])
        cls_ids.append(c)
    return boxes, cls_ids


def show_bev(img_path, label_path=None, class_names=None, save=None):
    class_names = class_names or ["Car", "Pedestrian", "Cyclist"]
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError("Cannot read: {}".format(img_path))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if label_path:
        boxes, ids = load_yolo(label_path, h, w)
        names = [class_names[i] if i < len(class_names) else "?" for i in ids]
        img = draw_boxes(img, boxes, names)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(Path(img_path).stem)
    plt.axis("off")
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print("Saved {}".format(save))
    else:
        plt.show()
    plt.close()


def show_pcd(pcd_path):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed. Run: pip install open3d")
        return
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_colors():
        pts = np.asarray(pcd.points)
        z   = pts[:, 2]
        zn  = (z - z.min()) / (z.max() - z.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(zn)[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                       window_name=Path(pcd_path).stem,
                                       width=1280, height=720)


def main():
    ap = argparse.ArgumentParser(description="LiDAR visualisation tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bev", help="Show BEV image with optional labels")
    bp.add_argument("--image",   required=True)
    bp.add_argument("--label",   default=None)
    bp.add_argument("--classes", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    bp.add_argument("--save",    default=None)

    pp = sub.add_parser("pcd", help="Show 3-D point cloud (open3d)")
    pp.add_argument("--pcd", required=True)

    a = ap.parse_args()
    if a.cmd == "bev":
        show_bev(a.image, a.label, a.classes, a.save)
    else:
        show_pcd(a.pcd)


if __name__ == "__main__":
    main()
