"""
convert_image_detection_original.py - Converts raw dataset to YOLO/TinyML_Segmentation format

Reads raw images and segmentation masks from 'image_detection_original/' (in project root or relative path),
extracts bounding box and 4 corner keypoints, normalizes coordinates, and exports to 'water_meter/' structure.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import config

CSV_COLUMNS = [
    "filename",
    "clear",
    "blurry",
    "dial_stained",
    "soil_covered",
    "dark",
    "reflective",
    "six_digit",
]

def load_label_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] == len(CSV_COLUMNS):
        df.columns = CSV_COLUMNS
    else:
        df = pd.read_csv(csv_path)
        if len(df.columns) >= 8:
            df = df.iloc[:, :8]
            df.columns = CSV_COLUMNS

    for col in CSV_COLUMNS[1:]:
        df[col] = df[col].fillna(0).astype(int)
    return df

def sort_quad_vertices(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4 corner points into consistent order:
    TL (top-left), TR (top-right), BR (bottom-right), BL (bottom-left)
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1) # x + y
    d = np.diff(pts, axis=1)[:, 0] # y - x

    ordered[0] = pts[np.argmin(s)]   # TL (smallest x+y)
    ordered[2] = pts[np.argmax(s)]   # BR (largest x+y)
    ordered[1] = pts[np.argmin(d)]   # TR (smallest y-x)
    ordered[3] = pts[np.argmax(d)]   # BL (largest y-x)

    return ordered

def extract_bbox_and_keypoints(mask: np.ndarray, img_w: int, img_h: int):
    """
    Extracts normalized (cx, cy, w, h) and 4-corner keypoints from binary segmentation mask.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[1] != img_w or mask.shape[0] != img_h:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 10:
        return None, None

    # Bounding box
    x, y, w, h = cv2.boundingRect(main_contour)
    cx = (x + w / 2.0) / float(img_w)
    cy = (y + h / 2.0) / float(img_h)
    w_n = float(w) / float(img_w)
    h_n = float(h) / float(img_h)

    bbox = (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w_n)),
        max(0.0, min(1.0, h_n)),
    )

    # 4 Corners keypoints via minAreaRect
    rect = cv2.minAreaRect(main_contour)
    vertices = cv2.boxPoints(rect).astype(np.float32)
    vertices = sort_quad_vertices(vertices)

    keypoints = [
        (
            max(0.0, min(1.0, float(pt[0] / img_w))),
            max(0.0, min(1.0, float(pt[1] / img_h))),
        )
        for pt in vertices
    ]

    return bbox, keypoints

def process_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    max_samples: int = None,
    seed: int = 42
):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    print("=" * 65)
    print("  CONVERTING IMAGE_DETECTION_ORIGINAL TO WATER_METER FORMAT")
    print(f"  Source Directory: {source_dir}")
    print(f"  Output Directory: {output_dir}")
    if max_samples:
        print(f"  Max Samples Limit: {max_samples}")
    print("=" * 65)

    train_src_img = source_dir / "train" / "train_img"
    train_src_lbl = source_dir / "train" / "train_seg_label"
    train_csv = source_dir / "train" / "train_class_label_CSV.csv"

    test_src_img = source_dir / "test" / "test_img"
    test_src_lbl = source_dir / "test" / "test_seg_label"
    test_csv = source_dir / "test" / "test_class_label_CSV.csv"

    if not train_src_img.exists() or not train_csv.exists():
        print(f"[ERROR] Source directories missing in {source_dir}")
        return

    # Create destination folders
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    df_train = load_label_csv(train_csv)
    df_test = load_label_csv(test_csv) if test_csv.exists() else pd.DataFrame()

    print(f"Loaded train CSV: {len(df_train)} records")
    print(f"Loaded test CSV : {len(df_test)} records")

    # Split train into train/val
    random.seed(seed)
    train_indices = list(range(len(df_train)))
    random.shuffle(train_indices)

    num_val = int(len(train_indices) * val_ratio)
    val_set_indices = set(train_indices[:num_val])

    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    stats_records = []

    # Helper to convert a batch
    def process_records(df: pd.DataFrame, img_dir: Path, mask_dir: Path, default_split_is_test: bool = False):
        count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), unit="img"):
            if max_samples and (stats["train"] + stats["val"] + stats["test"]) >= max_samples:
                break

            filename = row["filename"]
            stem = Path(filename).stem
            img_path = img_dir / filename
            mask_path = mask_dir / filename

            if not img_path.exists() or not mask_path.exists():
                stats["skipped"] += 1
                continue

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                stats["skipped"] += 1
                continue

            h, w = img.shape[:2]
            bbox, kpts = extract_bbox_and_keypoints(mask, w, h)
            if bbox is None or kpts is None:
                stats["skipped"] += 1
                continue

            # Target split
            if default_split_is_test:
                split = "test"
            else:
                split = "val" if idx in val_set_indices else "train"

            # YOLO pose line format: 0 cx cy w h x1 y1 x2 y2 x3 y3 x4 y4
            cx, cy, w_n, h_n = bbox
            line_parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
            for kx, ky in kpts:
                line_parts.extend([f"{kx:.6f}", f"{ky:.6f}"])
            yolo_line = " ".join(line_parts)

            # Copy image & save label
            dst_img_path = output_dir / "images" / split / filename
            dst_lbl_path = output_dir / "labels" / split / f"{stem}.txt"

            shutil.copy2(str(img_path), str(dst_img_path))
            dst_lbl_path.write_text(yolo_line + "\n", encoding="utf-8")

            stats[split] += 1
            count += 1

            record = {col: row[col] for col in CSV_COLUMNS}
            record["split"] = split
            stats_records.append(record)

        return count

    print("\n[Processing Train & Val Splits...]")
    process_records(df_train, train_src_img, train_src_lbl, default_split_is_test=False)

    if len(df_test) > 0 and (not max_samples or (stats["train"] + stats["val"] + stats["test"]) < max_samples):
        print("\n[Processing Test Split...]")
        process_records(df_test, test_src_img, test_src_lbl, default_split_is_test=True)

    # Save stats CSV and YAML configuration
    df_stats = pd.DataFrame(stats_records)
    df_stats.to_csv(output_dir / "dataset_stats.csv", index=False)

    yaml_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: water_meter_dial

kpt_shape: [4, 2]
"""
    (output_dir / "water_meter.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("  CONVERSION COMPLETE")
    print("=" * 65)
    print(f"  Train samples: {stats['train']:>6}")
    print(f"  Val samples  : {stats['val']:>6}")
    print(f"  Test samples : {stats['test']:>6}")
    print(f"  Skipped      : {stats['skipped']:>6}")
    print(f"  Total        : {stats['train'] + stats['val'] + stats['test']:>6}")
    print(f"  Output YAML  : {output_dir / 'water_meter.yaml'}")
    print(f"  Output Stats : {output_dir / 'dataset_stats.csv'}")

if __name__ == "__main__":
    default_source = config.ROOT_DIR / "image_detection_original"
    if not default_source.exists() and Path("image_detection_original").exists():
        default_source = Path("image_detection_original")

    default_output = config.DATASET_ROOT

    parser = argparse.ArgumentParser(description="Convert image_detection_original to water_meter format")
    parser.add_argument("--source-dir", type=Path, default=default_source, help="Source dataset path")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output dataset path")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio from train split")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for dry-run testing")
    args = parser.parse_args()

    process_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples
    )
