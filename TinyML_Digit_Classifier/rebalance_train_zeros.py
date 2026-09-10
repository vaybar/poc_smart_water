"""
rebalance_train_zeros.py

Safely rebalances the '0' class in dataset_mobilenet/train/0:
- Keeps a designated target number of representative images (default: 2000).
- Moves the excess images to dataset_mobilenet/discarded_images/train/0/
- Logs all moved files to a JSON file so the operation is 100% reversible.
"""

import os
import json
import random
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset_mobilenet"
TRAIN_ZERO_DIR = DATASET_DIR / "train" / "0"
DISCARD_ZERO_DIR = DATASET_DIR / "discarded_images" / "train" / "0"
LOG_PATH = BASE_DIR / "rebalance_train_zeros.log.json"

TARGET_COUNT = 2000
SEED = 42

def rebalance_zeros(target_count=TARGET_COUNT, seed=SEED):
    print("=" * 65)
    print("  REBALANCING DIGIT '0' IN TRAIN DATASET")
    print(f"  Source directory : {TRAIN_ZERO_DIR}")
    print(f"  Target directory : {DISCARD_ZERO_DIR}")
    print(f"  Target keep count: {target_count}")
    print("=" * 65)

    if not TRAIN_ZERO_DIR.exists():
        print(f"Error: Directory {TRAIN_ZERO_DIR} does not exist.")
        return

    DISCARD_ZERO_DIR.mkdir(parents=True, exist_ok=True)

    # List all valid image files
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    all_files = sorted([f for f in TRAIN_ZERO_DIR.glob("*") if f.suffix.lower() in valid_exts])
    total_found = len(all_files)
    print(f"Found {total_found} images in {TRAIN_ZERO_DIR.name}")

    if total_found <= target_count:
        print(f"Nothing to move. Current count ({total_found}) <= target ({target_count}).")
        return

    # Deterministic shuffle to pick samples to keep
    random.seed(seed)
    indices = list(range(total_found))
    random.shuffle(indices)

    keep_indices = set(indices[:target_count])
    files_to_keep = [all_files[i] for i in keep_indices]
    files_to_move = [all_files[i] for i in indices[target_count:]]

    print(f"Selected {len(files_to_keep)} images to keep.")
    print(f"Moving {len(files_to_move)} images to {DISCARD_ZERO_DIR}...")

    moved_records = []
    for f in files_to_move:
        dest_file = DISCARD_ZERO_DIR / f.name
        if dest_file.exists():
            dest_file = DISCARD_ZERO_DIR / f"{f.stem}_rebalanced{f.suffix}"
        shutil.move(str(f), str(dest_file))
        moved_records.append({"source": str(f), "destination": str(dest_file)})

    # Save log for reversibility
    with open(LOG_PATH, "w", encoding="utf-8") as out_f:
        json.dump(moved_records, out_f, indent=2)

    # Verify final count
    remaining = len([f for f in TRAIN_ZERO_DIR.glob("*") if f.suffix.lower() in valid_exts])
    in_discarded = len([f for f in DISCARD_ZERO_DIR.glob("*") if f.suffix.lower() in valid_exts])

    print("=" * 65)
    print("  REBALANCE COMPLETED SUCCESSFULLY")
    print(f"  Images remaining in train/0            : {remaining}")
    print(f"  Images now in discarded_images/train/0 : {in_discarded}")
    print(f"  Reversibility log saved to             : {LOG_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    rebalance_zeros()
