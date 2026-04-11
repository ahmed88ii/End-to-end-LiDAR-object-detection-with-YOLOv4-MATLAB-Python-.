"""
generate_splits.py
==================
Split BEV image filenames into train / val / test sets and write
data/splits/{train,val,test}.txt (one stem per line, no extension).

Author : Ahmed Abdelkader
Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
License: MIT

Usage
-----
    # Default (70 / 15 / 15 split, config from YAML)
    python python/generate_splits.py \\
        --bev_dir data/bev/images/ \\
        --config  configs/yolov4_lidar.yaml

    # Override ratios on the command line
    python python/generate_splits.py \\
        --bev_dir data/bev/images/ \\
        --train 0.75 --val 0.15 --test 0.10 --seed 0

Output
------
    data/splits/train.txt
    data/splits/val.txt
    data/splits/test.txt
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config_splits(config_path: str) -> Dict[str, float | int]:
    """Load split ratios and random seed from a YAML config file."""
    defaults: Dict[str, float | int] = {
        'train': 0.70, 'val': 0.15, 'test': 0.15, 'seed': 42
    }
    if not config_path or not Path(config_path).exists():
        return defaults

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    s = cfg.get('splits', {})
    return {
        'train': float(s.get('train', defaults['train'])),
        'val':   float(s.get('val',   defaults['val'])),
        'test':  float(s.get('test',  defaults['test'])),
        'seed':  int(s.get('seed',    defaults['seed'])),
    }


def split_dataset(
    stems:      List[str],
    train_frac: float,
    val_frac:   float,
    test_frac:  float,
    seed:       int,
) -> Dict[str, List[str]]:
    """
    Randomly partition a list of stems into train / val / test subsets.

    Parameters
    ----------
    stems      : sorted list of file stems (no extension)
    train_frac : fraction for training (e.g. 0.70)
    val_frac   : fraction for validation
    test_frac  : fraction for test  (ignored if fractions do not sum to 1)
    seed       : random seed for reproducibility

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    total = len(stems)
    if total == 0:
        raise ValueError("Empty dataset — no .png files found.")

    # Normalise fractions so they sum to exactly 1.0
    frac_sum = train_frac + val_frac + test_frac
    train_frac /= frac_sum
    val_frac   /= frac_sum

    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)

    n_val   = max(1, int(total * val_frac))
    n_test  = max(1, total - int(total * train_frac) - n_val)
    n_train = total - n_val - n_test

    # Safety: ensure at least 1 sample per split
    n_train = max(1, n_train)
    n_val   = max(1, n_val)
    n_test  = max(1, total - n_train - n_val)

    return {
        'train': shuffled[:n_train],
        'val':   shuffled[n_train:n_train + n_val],
        'test':  shuffled[n_train + n_val:],
    }


def write_splits(splits: Dict[str, List[str]], out_dir: Path) -> None:
    """Write each split to a .txt file (one stem per line)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        path = out_dir / f"{name}.txt"
        path.write_text('\n'.join(stems) + '\n', encoding='utf-8')
        print(f"  {name:5s}: {len(stems):5d} samples  →  {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split BEV image filenames into train/val/test sets"
    )
    parser.add_argument('--bev_dir', default='data/bev/images/',
                        help="Directory containing BEV .png files")
    parser.add_argument('--out_dir', default='data/splits/',
                        help="Output directory for split .txt files")
    parser.add_argument('--config', default='configs/yolov4_lidar.yaml',
                        help="YAML config file (optional; CLI args override)")
    parser.add_argument('--train', type=float, default=None,
                        help="Training fraction (overrides config)")
    parser.add_argument('--val',   type=float, default=None,
                        help="Validation fraction (overrides config)")
    parser.add_argument('--test',  type=float, default=None,
                        help="Test fraction (overrides config)")
    parser.add_argument('--seed',  type=int,   default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    # Merge config with CLI overrides
    cfg = load_config_splits(args.config)
    train_frac = args.train if args.train is not None else cfg['train']
    val_frac   = args.val   if args.val   is not None else cfg['val']
    test_frac  = args.test  if args.test  is not None else cfg['test']
    seed       = args.seed  if args.seed  is not None else cfg['seed']

    bev_dir = Path(args.bev_dir)
    stems   = sorted(p.stem for p in bev_dir.glob('*.png'))

    if not stems:
        print(f"No .png files found in {bev_dir}")
        return

    print(f"Found {len(stems)} BEV images in {bev_dir}")
    print(f"Split ratios  — train: {train_frac:.0%}  val: {val_frac:.0%}  "
          f"test: {test_frac:.0%}  (seed={seed})\n")

    splits = split_dataset(stems, train_frac, val_frac, test_frac, seed)
    write_splits(splits, Path(args.out_dir))

    print(f"\nTotal: {len(stems)} samples")


if __name__ == '__main__':
    main()
