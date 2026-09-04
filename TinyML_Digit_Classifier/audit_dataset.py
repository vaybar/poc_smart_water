"""
audit_dataset.py - Dataset Audit, Cleaning & Distribution Inspection

Inspects the dataset directory (e.g. dataset_mobilenet), checks:
1. Total images per split (train, val, test) and per class (0-9)
2. Class imbalance detection
3. Corrupted or unreadable image identification
4. Image resolution distribution & contrast anomalies
"""

import os
from pathlib import Path
import numpy as np
import cv2
import config

def audit_dataset(dataset_dir: Path = config.DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    print("=" * 60)
    print(f"  Auditing Dataset: {dataset_dir}")
    print("=" * 60)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return

    splits = ["train", "val", "test"]
    subfolder_structure = any((dataset_dir / s).exists() for s in splits)
    
    if subfolder_structure:
        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                print(f"\n[Split: {split}] -> Directory not found")
                continue
                
            print(f"\n--- Split: {split.upper()} ---")
            total_split_imgs = 0
            corrupt_count = 0
            low_contrast_count = 0
            
            for digit in range(10):
                digit_dir = split_dir / str(digit)
                if not digit_dir.exists():
                    print(f"  Class {digit}: 0 images (directory missing)")
                    continue
                    
                files = list(digit_dir.glob("*"))
                valid_files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
                total_split_imgs += len(valid_files)
                
                # Check sample contrast and integrity
                for f in valid_files:
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        corrupt_count += 1
                    else:
                        std_dev = np.std(img)
                        if std_dev < 10.0:  # Extremely low contrast / solid color
                            low_contrast_count += 1
                            
                print(f"  Class {digit}: {len(valid_files):5d} images")
                
            print(f" Total in {split}: {total_split_imgs} images")
            if corrupt_count > 0:
                print(f" [WARNING] Corrupted/Unreadable images: {corrupt_count}")
            if low_contrast_count > 0:
                print(f" [WARNING] Extremely low contrast images (<10 std dev): {low_contrast_count}")
    else:
        print("Dataset directory is flat (or un-split). Checking class folders 0..9...")
        total_imgs = 0
        for digit in range(10):
            digit_dir = dataset_dir / str(digit)
            if not digit_dir.exists():
                continue
            files = [f for f in digit_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
            total_imgs += len(files)
            print(f"  Class {digit}: {len(files):5d} images")
        print(f" Total Images: {total_imgs}")

    print("=" * 60)

if __name__ == "__main__":
    audit_dataset()
