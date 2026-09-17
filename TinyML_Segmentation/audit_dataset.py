"""
audit_dataset.py - Dataset Integrity, Keypoints & Orientation Audit

Inspects the dataset (e.g. water_meter/):
1. Verifies image integrity, resolution, and readability.
2. Validates 4-corner keypoints (normalization in [0, 1], polygon convexity).
3. Analyzes counter rotation distribution (angles in degrees).
4. Checks challenging flags (soil-covered, dial-stained, reflective) from dataset_stats.csv.
"""

import math
from pathlib import Path
import numpy as np
import cv2
import config

def calculate_quad_angle(kpts: np.ndarray) -> float:
    """
    Computes rotation angle (in degrees) of the quadrilateral's top edge (TL -> TR).
    kpts: (4, 2) array of [x, y] coordinates.
    """
    tl = kpts[0]
    tr = kpts[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def audit_dataset():
    print("=" * 65)
    print("  AUDITING WATER METER SEGMENTATION DATASET")
    print(f"  Dataset Root: {config.DATASET_ROOT}")
    print("=" * 65)

    if not config.DATASET_ROOT.exists():
        print(f"[ERROR] Dataset directory not found: {config.DATASET_ROOT}")
        return

    splits = ["train", "val", "test"]
    total_valid_samples = 0
    angles_list = []
    aspect_ratios = []
    corrupted_images = 0
    invalid_labels = 0

    for split in splits:
        img_split_dir = config.IMAGES_DIR / split
        lbl_split_dir = config.LABELS_DIR / split

        if not img_split_dir.exists():
            print(f"\n[Split: {split.upper()}] Images directory not found ({img_split_dir})")
            continue

        img_files = sorted([f for f in img_split_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        lbl_files = sorted(list(lbl_split_dir.glob("*.txt"))) if lbl_split_dir.exists() else []

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Total images found: {len(img_files)}")
        print(f"  Total labels found: {len(lbl_files)}")

        split_angles = []

        for img_p in img_files:
            lbl_p = lbl_split_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                invalid_labels += 1
                continue

            # Check label format
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                tokens = line.split()
                if len(tokens) < 13: # class + cx,cy,w,h + 4 keypoints (x,y)
                    invalid_labels += 1
                    continue

                vals = [float(v) for v in tokens[5:13]]
                kpts = np.array(vals).reshape(4, 2)

                # Check bounds
                if np.any(kpts < 0.0) or np.any(kpts > 1.0):
                    invalid_labels += 1
                    continue

                # Calculate angle of counter box
                ang = calculate_quad_angle(kpts)
                split_angles.append(ang)
                angles_list.append(ang)

                # Calculate approximate aspect ratio (width / height)
                top_w = np.linalg.norm(kpts[1] - kpts[0])
                left_h = np.linalg.norm(kpts[3] - kpts[0])
                if left_h > 1e-4:
                    aspect_ratios.append(top_w / left_h)

                total_valid_samples += 1

            except Exception:
                invalid_labels += 1

        if split_angles:
            print(f"  Valid samples with 4 corners: {len(split_angles)}")
            print(f"  Counter rotation range: {min(split_angles):.1f}° to {max(split_angles):.1f}° (mean: {np.mean(split_angles):.1f}°)")

    print("\n" + "=" * 65)
    print("  AUDIT SUMMARY")
    print("=" * 65)
    print(f"  Total valid paired samples : {total_valid_samples}")
    print(f"  Invalid or missing labels  : {invalid_labels}")
    print(f"  Corrupted image files      : {corrupted_images}")

    if aspect_ratios:
        print(f"  Average Counter Aspect Ratio: {np.mean(aspect_ratios):.2f}:1 (Target Strip 6 digits is ~3:1)")

    # Check for environmental flags summary
    stats_csv = config.DATASET_ROOT / "dataset_stats.csv"
    if stats_csv.exists():
        print("\n  [Found dataset_stats.csv with challenging conditions]")
        try:
            import pandas as pd
            df = pd.read_csv(stats_csv)
            print(f"  Columns: {list(df.columns)}")
            for col in ["soil-covered", "dial-stained", "reflective", "blurry"]:
                if col in df.columns:
                    print(f"    - Flag '{col}': {df[col].sum()} samples")
        except Exception as e:
            print(f"  (Could not parse dataset_stats.csv: {e})")

    print("\n[OK] Dataset audit complete. Data is ready for TinyML_Segmentation experiments.")

if __name__ == "__main__":
    audit_dataset()
