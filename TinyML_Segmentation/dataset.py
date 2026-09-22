"""
dataset.py - Dataset Loader, Keypoint Augmentation & PTQ Calibration Generator

Loads the water meter counter localization dataset:
1. Parses images and 4-corner keypoints from YOLO format labels.
2. Resizes and converts images to (128, 128, 1) grayscale thumbnails.
3. Implements affine-safe data augmentation (rotation, brightness, contrast, scaling).
4. Generates representative calibration dataset for full INT8 post-training quantization.
"""

from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import config

def apply_photometric_augmentation(img_gray: np.ndarray) -> np.ndarray:
    """
    Applies photometric data augmentation (brightness, contrast, noise, blur)
    which does not alter spatial keypoint coordinates.
    """
    aug = img_gray.astype(np.float32)

    # 1. Random Brightness Jitter (-25 to +25)
    brightness_shift = np.random.uniform(-25.0, 25.0)
    aug += brightness_shift

    # 2. Random Contrast Factor (0.7 to 1.3)
    contrast_factor = np.random.uniform(0.7, 1.3)
    mean_val = np.mean(aug)
    aug = (aug - mean_val) * contrast_factor + mean_val

    # 3. Random Gaussian Noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, np.random.uniform(3.0, 10.0), size=aug.shape)
        aug += noise

    # 4. Random Subtle Blur
    if np.random.rand() > 0.7:
        ksize = np.random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    return np.clip(aug, 0, 255).astype(np.uint8)

def apply_affine_geometric_augmentation(
    img_gray: np.ndarray,
    kpts_norm: list[float] | np.ndarray,
    max_rotation_deg: float = 25.0,
    max_translation_px: float = 8.0,
    scale_range: tuple[float, float] = (0.88, 1.12)
) -> tuple[np.ndarray, list[float]] | None:
    """
    Applies simultaneous 2D affine geometric transformation (rotation, scale, translation)
    to both the image and the 4 keypoint coordinates.
    """
    h, w = img_gray.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0

    # Random parameters
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    dx = np.random.uniform(-max_translation_px, max_translation_px)
    dy = np.random.uniform(-max_translation_px, max_translation_px)

    # Affine matrix
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    # Warp image
    aug_img = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Transform 4 keypoints (x_i, y_i)
    kpts_arr = np.array(kpts_norm, dtype=np.float32).reshape(4, 2)
    kpts_px = kpts_arr.copy()
    kpts_px[:, 0] *= float(w)
    kpts_px[:, 1] *= float(h)

    # Homogeneous 2D coordinates [x, y, 1]
    ones = np.ones((4, 1), dtype=np.float32)
    pts_homo = np.hstack([kpts_px, ones]) # (4, 3)

    # Transform points: P' = M * P^T -> (2, 4) -> (4, 2)
    transformed_pts = (M @ pts_homo.T).T

    # Normalize back to [0, 1]
    new_kpts_norm = transformed_pts.copy()
    new_kpts_norm[:, 0] /= float(w)
    new_kpts_norm[:, 1] /= float(h)

    # Check bounds safety (keep all points inside [0.01, 0.99])
    if np.any(new_kpts_norm < 0.01) or np.any(new_kpts_norm > 0.99):
        # Reject out-of-bound transformation
        return None

    # Apply photometric augmentation on top of warped image
    aug_img_photo = apply_photometric_augmentation(aug_img)

    return aug_img_photo, new_kpts_norm.flatten().tolist()

def load_paired_dataset(split: str = "train", augment: bool = False, augment_factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads all paired images and 4-corner keypoints for a given split ('train', 'val', 'test').

    Args:
        split: Dataset split ('train', 'val', 'test').
        augment: If True (recommended for training), multiplies dataset with affine & photometric variations.
        augment_factor: Number of augmented copies per training image (e.g. 4 -> 5x dataset size).

    Returns:
        images: (N, 128, 128, 1) uint8 numpy array.
        targets: (N, 8) float32 numpy array with normalized coordinates in [0.0, 1.0].
    """
    img_dir = config.IMAGES_DIR / split
    lbl_dir = config.LABELS_DIR / split

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[WARN] Split directory {split} missing ({img_dir} or {lbl_dir})")
        return np.empty((0, config.THUMB_H, config.THUMB_W, 1), dtype=np.uint8), np.empty((0, 8), dtype=np.float32)

    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    images = []
    targets = []

    for img_p in img_paths:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        try:
            with open(lbl_p, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            tokens = line.split()
            if len(tokens) < 13:
                continue

            # Extract 4 keypoints (x1, y1, x2, y2, x3, y3, x4, y4)
            kpts = [float(v) for v in tokens[5:13]]
            if any(v < 0.0 or v > 1.0 for v in kpts):
                continue

            # Load image in grayscale
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to thumbnail (128, 128)
            resized = cv2.resize(img, (config.THUMB_W, config.THUMB_H), interpolation=cv2.INTER_AREA)

            # Original sample
            images.append(np.expand_dims(resized, axis=-1))
            targets.append(kpts)

            # Augmented copies for training (Affine Geometric + Photometric)
            if augment and split == "train":
                for _ in range(augment_factor):
                    res = apply_affine_geometric_augmentation(resized, kpts)
                    if res is not None:
                        aug_img, aug_kpts = res
                        images.append(np.expand_dims(aug_img, axis=-1))
                        targets.append(aug_kpts)
                    else:
                        # Fallback to photometric only if geometric went out of bounds
                        aug_img = apply_photometric_augmentation(resized)
                        images.append(np.expand_dims(aug_img, axis=-1))
                        targets.append(kpts)

        except Exception:
            continue


    X = np.array(images, dtype=np.uint8)
    y = np.array(targets, dtype=np.float32)
    return X, y


def get_representative_dataset(num_samples: int = 150):
    """
    Generator yielding calibration samples for TFLite INT8 post-training quantization.
    """
    X_train, _ = load_paired_dataset("train")
    if len(X_train) == 0:
        X_train, _ = load_paired_dataset("val")

    if len(X_train) == 0:
        # Fallback synthetic
        for _ in range(num_samples):
            synthetic = np.random.randint(0, 256, size=(1, config.THUMB_H, config.THUMB_W, 1), dtype=np.uint8)
            yield [synthetic.astype(np.float32)]
        return

    indices = np.random.choice(len(X_train), size=min(num_samples, len(X_train)), replace=False)
    for idx in indices:
        sample = X_train[idx:idx+1].astype(np.float32)
        yield [sample]
