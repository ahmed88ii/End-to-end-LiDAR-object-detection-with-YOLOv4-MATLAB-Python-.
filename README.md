# End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)

**Author:** Ahmed Abdelkader  
**License:** MIT  
**Platform:** MATLAB R2021a+ · Python 3.8+

> Bachelor's thesis project implementing an end-to-end 3D object detection
> pipeline for Livox LiDAR sensors.  Raw `.lvx` scans are converted to
> Bird's-Eye-View (BEV) images, a Complex-YOLOv4 model is trained in
> MATLAB, and performance is evaluated using AP, AOS, Precision, Recall,
> and BEV IoU metrics.

---

## Pipeline Overview

```
.lvx files          .pcd files         BEV images (.png)
(Livox raw)  -->  (point cloud)  -->  (3-channel map)  -->  Complex-YOLOv4 (MATLAB)
                                                                    |
                                                  Detections  <-----+
                                                  AP / AOS / BEV IoU metrics
```

| Step | Script | Language |
|------|--------|----------|
| 1 · LVX -> PCD conversion | `python/lvx_to_pcd.py` | Python |
| 2 · PCD -> BEV image | `matlab/generate_bev.m` / `python/pcd_to_bev.py` | MATLAB / Python |
| 3 · Dataset splitting | `python/generate_splits.py` | Python |
| 4 · Prepare datastores + anchors | `matlab/prepare_training_data.m` | MATLAB |
| 5 · Train Complex-YOLOv4 | `matlab/train_yolov4.m` | MATLAB |
| 6 · Inference | `matlab/detect_objects.m` | MATLAB |
| 7 · Export GT | `matlab/export_ground_truth.m` | MATLAB |
| 8 · Evaluate AP / AOS / BEV IoU | `matlab/evaluate_detector.m` | MATLAB |
| Visualise BEV + boxes | `python/visualize.py` | Python |

---

## Repository Structure

```
.
|-- configs/
|   +-- yolov4_lidar.yaml          # BEV params + exact training hyperparameters
|-- data/
|   |-- raw/                       # Place .lvx files here (git-ignored)
|   |-- pcd/                       # Generated .pcd files  (git-ignored)
|   |-- bev/
|   |   |-- images/                # Generated BEV .png    (git-ignored)
|   |   +-- labels/                # YOLO .txt labels      (git-ignored)
|   +-- splits/
|       |-- train.txt
|       |-- val.txt
|       +-- test.txt
|-- docs/
|   +-- Abdelkader-_Ahmed_Thesis final submission.pdf
|-- matlab/
|   |-- generate_bev.m             # PCD -> 3-channel BEV PNG (height/intensity/density)
|   |-- prepare_training_data.m    # Build datastores, estimate anchors
|   |-- train_yolov4.m             # Complex-YOLOv4 (thesis hyperparameters)
|   |-- detect_objects.m           # Batch inference -> detection .txt files
|   |-- export_ground_truth.m      # YOLO GT -> KITTI-inspired format
|   |-- evaluate_detector.m        # PR curves, AP, AOS, BEV IoU plots
|   +-- utils/
|       |-- calculate3DIoU.m       # Full volumetric IoU (Sutherland-Hodgman)
|       |-- compute_iou_3d.m       # Legacy alias for calculate3DIoU
|       |-- compute_ap.m           # 11-point / AUC Average Precision
|       |-- compute_aos.m          # Average Orientation Similarity
|       +-- visualize_detections.m # Overlay boxes on BEV image
|-- python/
|   |-- lvx_to_pcd.py              # Livox binary parser (struct) -> ASCII PCD
|   |-- pcd_to_bev.py              # BEV 3-channel image generation (vectorised)
|   |-- generate_splits.py         # Dataset train/val/test split
|   |-- visualize.py               # BEV + box visualisation
|   +-- utils/
|       |-- box_utils.py           # BEV IoU (rotated + aligned), 3-D IoU
|       |-- metrics.py             # AP and AOS computation
|       +-- point_cloud_utils.py   # Load, filter, downsample, transform
|-- results/                       # Evaluation outputs (git-ignored)
|-- requirements.txt
|-- LICENSE
+-- README.md
```

---

## Installation

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.8 |
| MATLAB | R2021a |
| Deep Learning Toolbox | (bundled with MATLAB) |
| Computer Vision Toolbox | (bundled with MATLAB) |

### 1 · Clone the repository

```bash
git clone https://github.com/ahmed88ii/End-to-end-LiDAR-object-detection-with-YOLOv4-MATLAB-Python-.git
cd End-to-end-LiDAR-object-detection-with-YOLOv4-MATLAB-Python-
```

### 2 · Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 · MATLAB path

```matlab
addpath(genpath('matlab'));
```

---

## Usage

### Step 1 — Convert LVX to PCD

```bash
python python/lvx_to_pcd.py --input data/raw/ --output data/pcd/ --verbose
```

### Step 2 — Generate BEV images

```matlab
run('matlab/generate_bev.m')
```

or in Python:

```bash
python python/pcd_to_bev.py --input data/pcd/ --output data/bev/images/ \
                              --config configs/yolov4_lidar.yaml
```

### Step 3 — Generate dataset splits

```bash
python python/generate_splits.py --bev_dir data/bev/images/ \
                                  --config configs/yolov4_lidar.yaml
```

### Step 4 — Prepare datastores (MATLAB)

```matlab
run('matlab/prepare_training_data.m')
```

### Step 5 — Train Complex-YOLOv4 (MATLAB)

```matlab
run('matlab/train_yolov4.m')
```

Exact thesis hyperparameters (also recorded in `configs/yolov4_lidar.yaml`):

| Hyperparameter | Value |
|---|---|
| `maxEpochs` | 50 |
| `miniBatchSize` | 16 |
| `learningRate` | 0.001 |
| `warmupPeriod` | 500 iterations |
| `l2Regularization` | 0.001 |
| `penaltyThreshold` | 0.5 |
| Backbone | CSP-DarkNet53 (COCO pre-trained) |
| Input size | 416 x 416 x 3 |

### Step 6 — Inference

```matlab
run('matlab/detect_objects.m')
```

### Step 7 — Export ground truth (KITTI-inspired format)

```matlab
run('matlab/export_ground_truth.m')
```

### Step 8 — Evaluate

```matlab
run('matlab/evaluate_detector.m')
```

---

## BEV Image Format

Each BEV image is a **3-channel uint8 PNG** (default 704 x 800 px):

| Channel | Content | Normalisation |
|---------|---------|---------------|
| R | Maximum height in voxel column | `(z - z_min) / (z_max - z_min)` -> [0, 255] |
| G | Mean reflectivity / intensity | [0, 1] -> [0, 255] |
| B | Point density (count) | `count / 64` clamped to [0, 1] -> [0, 255] |

**Spatial extents** (from `configs/yolov4_lidar.yaml`):

| Axis | Range | Resolution | Pixels |
|------|-------|------------|--------|
| X (forward) | 0 -> 70.4 m | 0.1 m/px | 704 |
| Y (lateral) | -40 -> 40 m | 0.1 m/px | 800 |
| Z (height) | -3 -> 1 m | — | — |

---

## Evaluation Scope

All evaluation in this repo is **BEV-based (2-D Bird's-Eye-View)**, not
full 3-D volumetric.  This is by design: YOLO labels contain 2-D bounding
boxes only (no height or yaw annotations).

| Aspect | Implementation |
|--------|---------------|
| IoU matching | Axis-aligned 2-D BEV rectangle overlap |
| IoU thresholds | Car: 0.7 / Pedestrian: 0.5 / Cyclist: 0.5 (KITTI-inspired) |
| AP | 11-point PASCAL VOC interpolation |
| AOS | Computed with delta_yaw = 0 for all matched detections (no yaw labels available), so **AOS == AP** in practice |
| GT export format | KITTI-*inspired* field order; 3-D fields (height, location, yaw) are placeholder zeros |

The `calculate3DIoU.m` / `iou3d` utilities implement full Sutherland-Hodgman
volumetric IoU and are provided for future use with annotated 3-D datasets.

---

## Results

BEV-based evaluation on the own Livox dataset (collected and annotated as
part of the thesis). IoU thresholds follow KITTI convention.

| Class | AP (BEV) | AOS | IoU threshold |
|-------|---------|-----|---------------|
| **Car** | **0.496** | **0.496** | 0.7 |
| **Pedestrian** | **0.250** | **0.250** | 0.5 |
| Cyclist | — | — | 0.5 (insufficient samples) |

> AP and AOS computed via 11-point PASCAL VOC interpolation on BEV detections.
> AOS equals AP because BEV labels carry no yaw annotation (delta_yaw = 0
> for all matched detections). See the Evaluation Scope section above.

---

## Metrics

### Average Precision (AP)

11-point PASCAL VOC interpolation over recall levels {0, 0.1, ..., 1.0}:

```
AP = (1/11) * sum_{r in {0, 0.1, ..., 1}} max_{r' >= r} Precision(r')
```

### Average Orientation Similarity (AOS)

Introduced by Geiger et al. (CVPR 2012):

```
s(delta_theta) = (1 + cos(delta_theta)) / 2

AOS = (1/11) * sum_{r in {0, 0.1, ..., 1}} max_{r' >= r} mean_{d: recall(d) >= r'} s(delta_theta_d)
```

In this repo delta_theta = 0 for all detections (no yaw labels), so AOS = AP.

### BEV IoU (2-D axis-aligned)

```
IoU_BEV = intersection_area / union_area
```

where both boxes are axis-aligned rectangles in the BEV image plane.

---

## Reproducibility

**Raw data and trained weights are not included in this repository** due to
file-size constraints.  The full pipeline code is provided so that results
can be reproduced given the original data.

| Item | Status |
|------|--------|
| LVX -> PCD converter | Included (`python/lvx_to_pcd.py`) |
| BEV generation | Included (MATLAB + Python) |
| Training script with exact hyperparameters | Included (`matlab/train_yolov4.m`) |
| Hyperparameter config | Included (`configs/yolov4_lidar.yaml`) |
| Evaluation scripts | Included (`matlab/evaluate_detector.m`) |
| Raw `.lvx` scan data | **Not included** (dataset size) |
| Trained model weights (`.mat`) | **Not included** (file size) |
| Annotated BEV labels | **Not included** (dataset size) |

> **Data sharing:** The raw scan data and trained model weights can be
> shared on request for academic purposes.  Please open a GitHub issue or
> contact the author directly.

The MATLAB training script and YAML config are kept in sync — both record
`maxEpochs=50, miniBatchSize=16, learningRate=0.001, warmupPeriod=500,
l2Regularization=0.001, penaltyThreshold=0.5`.

---

## Requirements

```
numpy>=1.21
open3d>=0.15
opencv-python>=4.5
matplotlib>=3.4
scipy>=1.7
PyYAML>=5.4
```

Install with:

```bash
pip install -r requirements.txt
```

---

## References

1. **Bochkovskiy, A., Wang, C.-Y., and Liao, H.-Y. M.** (2020).
   *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv:2004.10934.

2. **Simon, M., Milz, S., Amende, K., and Gross, H.-M.** (2019).
   *Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection
   on Point Clouds.* arXiv:1803.06199.

3. **Geiger, A., Lenz, P., and Urtasun, R.** (2012).
   *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.*
   CVPR 2012.

4. **MathWorks.** (2021). *Train YOLO v4 Object Detector.*
   MATLAB Deep Learning Toolbox documentation.

5. **Livox Technology.** (2020). *Livox SDK and LVX File Format Specification.*
   github.com/Livox-SDK/Livox-SDK

---

## License

MIT © 2024 Ahmed Abdelkader — see [LICENSE](LICENSE) for full text.
