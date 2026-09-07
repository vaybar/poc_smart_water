"""
clean_dataset.py - Automated Dataset Cleaning & Discarded Image Isolation

Audits and cleans the digit dataset:
1. Identifies corrupt/unreadable images and ultra-low contrast images (std dev < 10.0).
2. MOVES all discarded images to a dedicated folder `discarded_images/` preserving class and split structure for subsequent inspection.
3. Reports precise counts of cleaned and preserved images per split and digit.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import config

def clean_dataset(
    dataset_dir: Path = config.DATASET_DIR,
    min_std_dev: float = 10.0,
    discarded_dir_name: str = "discarded_images"
):
    dataset_dir = Path(dataset_dir)
    discarded_root = dataset_dir / discarded_dir_name
    discarded_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print(f"  Cleaning Dataset: {dataset_dir}")
    print(f"  Discarded Images Folder: {discarded_root}")
    print(f"  Filtering threshold: Image std dev < {min_std_dev}")
    print("=" * 65)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return

    splits = ["train", "val", "test"]
    total_moved = 0
    total_kept = 0
    stats_per_class = {d: 0 for d in range(10)}

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n--- Cleaning Split: {split.upper()} ---")
        split_moved = 0
        split_kept = 0
        
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if not digit_dir.exists():
                continue
                
            discard_dest_dir = discarded_root / split / str(digit)
            
            files = list(digit_dir.glob("*"))
            valid_files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
            
            digit_moved = 0
            digit_kept = 0
            
            for f in valid_files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                
                # Check condition: corrupt or std_dev < 10.0
                is_corrupt = (img is None)
                is_low_contrast = (img is not None and np.std(img) < min_std_dev)
                
                if is_corrupt or is_low_contrast:
                    discard_dest_dir.mkdir(parents=True, exist_ok=True)
                    target_path = discard_dest_dir / f.name
                    
                    # Handle name collision if file exists
                    if target_path.exists():
                        target_path = discard_dest_dir / f"{f.stem}_dup{f.suffix}"
                        
                    shutil.move(str(f), str(target_path))
                    digit_moved += 1
                    stats_per_class[digit] += 1
                else:
                    digit_kept += 1
                    
            split_moved += digit_moved
            split_kept += digit_kept
            print(f"  Class {digit}: Kept {digit_kept:5d} | Discarded & Moved: {digit_moved:3d}")
            
        print(f" Split {split.upper()} Summary: Kept {split_kept} images, Discarded {split_moved} images.")
        total_moved += split_moved
        total_kept += split_kept

    print("\n" + "=" * 65)
    print("                    CLEANING COMPLETE SUMMARY")
    print("=" * 65)
    print(f" Total Images Retained  : {total_kept}")
    print(f" Total Images Discarded : {total_moved}")
    print(f" Discarded Location     : {discarded_root}")
    print(" Desglose de descartados por dígito:")
    for d in range(10):
        print(f"   - Dígito '{d}': {stats_per_class[d]} imágenes aisladas")
    print("=" * 65)

if __name__ == "__main__":
    clean_dataset()
