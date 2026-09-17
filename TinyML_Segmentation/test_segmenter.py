"""
test_segmenter.py - Visual Verification & Contract Testing for TinyML_Segmentation

Verifies:
1. Shape contract: exactly N output images with shape (64, 32, 1) and dtype uint8.
2. Value contract: pixel values in valid range [0, 255].
3. Visual diagnostics: saves annotated original, rectified strip, and sliced digits.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

import config
from segmenter import TinyMLSegmenter

def run_test(image_path: Path | str | None = None, num_digits: int = config.DEFAULT_NUM_DIGITS):
    debug_dir = config.BASE_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  TESTING TinyML_Segmentation PIPELINE")
    print("=" * 65)

    # 1. Resolve test image
    if image_path is None or not Path(image_path).exists():
        # Search for available images in dataset
        candidates = list(config.IMAGES_DIR.glob("**/*.png")) + list(config.IMAGES_DIR.glob("**/*.jpg"))
        if not candidates:
            # Check parent folder for sample images
            candidates = list(config.ROOT_DIR.glob("*.jpg")) + list(config.ROOT_DIR.glob("*.png"))

        if candidates:
            image_path = candidates[0]
            print(f"[INFO] Using test image: {image_path}")
        else:
            print("[WARN] No real image found. Generating a synthetic water meter image.")
            # Synthetic 480x640 image with rotated counter box
            syn = np.full((480, 640), 120, dtype=np.uint8)
            cv2.circle(syn, (320, 240), 180, (80,), -1) # Dial body
            # Rotated white rectangle for counter
            rect = ((320, 200), (160, 45), 20) # Center, size, angle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(syn, [box], 0, (240,), -1)
            image_path = debug_dir / "synthetic_meter.png"
            cv2.imwrite(str(image_path), syn)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    print(f"Input image shape: {img.shape}")

    # 2. Instantiate Segmenter
    segmenter = TinyMLSegmenter(num_digits=num_digits)

    # 3. Run Segmentation
    result = segmenter.segment(img)
    print(f"Pipeline Execution Time: {result.execution_time_ms:.1f} ms")

    # 4. Validate Contracts
    print("\n--- Validating Contract with TinyML_Digit_Classifier ---")
    assert len(result.digits) == num_digits, f"Expected {num_digits} digits, got {len(result.digits)}"
    print(f"[PASS] Number of digits: {len(result.digits)}")

    for idx, d in enumerate(result.digits):
        assert d.shape == (config.DIGIT_H, config.DIGIT_W, 1), f"Digit {idx} shape mismatch: {d.shape}"
        assert d.dtype == np.uint8, f"Digit {idx} dtype mismatch: {d.dtype}"
        assert d.min() >= 0 and d.max() <= 255, f"Digit {idx} value out of range [0, 255]"

    print(f"[PASS] All {num_digits} digit crops strictly conform to ({config.DIGIT_H}, {config.DIGIT_W}, 1) uint8!")

    # 5. Save Visual Diagnostics
    # Draw polygon on original
    annotated = img.copy()
    pts = result.corners_original.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, pt in enumerate(result.corners_original):
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        cv2.putText(annotated, f"P{i}", (int(pt[0]) + 5, int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imwrite(str(debug_dir / "01_detected_counter_corners.png"), annotated)
    cv2.imwrite(str(debug_dir / "02_rectified_strip_64x192.png"), result.rectified_strip)

    for idx, d in enumerate(result.digits):
        cv2.imwrite(str(debug_dir / f"03_digit_slot_{idx}_64x32.png"), d[:, :, 0])

    print(f"\n[OK] Diagnostic images successfully saved to: {debug_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TinyML_Segmentation pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--digits", type=int, default=config.DEFAULT_NUM_DIGITS, help="Number of digits")
    args = parser.parse_args()

    run_test(image_path=args.image, num_digits=args.digits)
