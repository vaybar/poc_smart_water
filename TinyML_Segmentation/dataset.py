"""
dataset.py - Dataset Loader, Keypoint Augmentation & PTQ Calibration Generator

Loads the water meter counter localization dataset:
1. Parses images and 4-corner keypoints from YOLO format labels.
2. Resizes and converts images to (128, 128, 1) grayscale thumbnails.
3. Implements affine-safe data augmentation (rotation, brightness, contrast, scaling).
4. Implements spatial geometric augmentation (rotation, scale, translation, flip)
   that co-transforms images AND keypoint coordinates consistently.
5. Implements CutOut (random erasing) to simulate partial occlusions.
6. Generates representative calibration dataset for full INT8 post-training quantization.
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


def apply_cutout(img_gray: np.ndarray, max_patches: int = 3,
                 patch_ratio_range: tuple = (0.05, 0.15)) -> np.ndarray:
    """
    Random Erasing / CutOut augmentation — masks rectangular patches to simulate
    partial occlusions (mud, water droplets, insects, shadows over the meter).

    Only modifies the image, NOT keypoint coordinates (occlusions don't move corners).

    Args:
        img_gray: (H, W) uint8 grayscale image.
        max_patches: Maximum number of rectangular patches to erase (1 to max_patches).
        patch_ratio_range: (min, max) fraction of image dimension for patch size.

    Returns:
        Augmented image with random gray-filled rectangular patches.
    """
    h, w = img_gray.shape[:2]
    aug = img_gray.copy()

    n_patches = np.random.randint(1, max_patches + 1)
    for _ in range(n_patches):
        ratio_h = np.random.uniform(*patch_ratio_range)
        ratio_w = np.random.uniform(*patch_ratio_range)
        ph = max(1, int(h * ratio_h))
        pw = max(1, int(w * ratio_w))
        y0 = np.random.randint(0, max(1, h - ph))
        x0 = np.random.randint(0, max(1, w - pw))
        # Fill with random mid-gray (avoids strong black/white artifacts)
        aug[y0:y0 + ph, x0:x0 + pw] = np.random.randint(50, 200)

    return aug


def apply_spatial_augmentation(
    img_gray: np.ndarray,
    kpts: np.ndarray,
    max_rotation_deg: float = 15.0,
    max_scale_delta: float = 0.10,
    max_translate_frac: float = 0.05,
    flip_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies geometric spatial augmentation that co-transforms image AND keypoints.

    Operations (applied in order):
    1. Random rotation ±max_rotation_deg around image center
    2. Random uniform scale (1 ± max_scale_delta)
    3. Random translation ±max_translate_frac of image size
    4. Random horizontal flip with probability flip_prob

    Args:
        img_gray: (H, W) uint8 grayscale image.
        kpts: (8,) float32 normalized keypoint coordinates [x1,y1,...,x4,y4] in [0,1].
        max_rotation_deg: Maximum rotation angle in degrees.
        max_scale_delta: Maximum scale deviation (e.g. 0.10 → scale in [0.90, 1.10]).
        max_translate_frac: Maximum translation as fraction of image size.
        flip_prob: Probability of horizontal flip.

    Returns:
        aug_img: (H, W) uint8 augmented grayscale image.
        aug_kpts: (8,) float32 augmented normalized keypoints, clamped to [0,1].
    """
    h, w = img_gray.shape[:2]
    pts = kpts.reshape(4, 2).copy()

    # Convert normalized coords to pixel coords
    pts_px = pts.copy()
    pts_px[:, 0] *= w
    pts_px[:, 1] *= h

    cx, cy = w / 2.0, h / 2.0

    # 1. Random Rotation
    angle_deg = np.random.uniform(-max_rotation_deg, max_rotation_deg)
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # 2. Random Scale
    scale = np.random.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta)
    M_rot[:, :2] *= scale

    # 3. Random Translation
    tx = np.random.uniform(-max_translate_frac, max_translate_frac) * w
    ty = np.random.uniform(-max_translate_frac, max_translate_frac) * h
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # Apply affine transform to image
    aug_img = cv2.warpAffine(
        img_gray, M_rot, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Apply affine transform to keypoints
    pts_hom = np.hstack([pts_px, np.ones((4, 1))])  # (4, 3)
    pts_transformed = (M_rot @ pts_hom.T).T  # (4, 2)

    # 4. Random Horizontal Flip
    if np.random.rand() < flip_prob:
        aug_img = cv2.flip(aug_img, 1)
        pts_transformed[:, 0] = w - 1 - pts_transformed[:, 0]
        # After horizontal flip, swap TL↔TR and BL↔BR to maintain canonical order:
        # Original order: [TL, TR, BR, BL] → Flipped: [TR', TL', BL', BR']
        # Reorder to: [TL', TR', BR', BL']
        pts_transformed = pts_transformed[[1, 0, 3, 2], :]

    # Normalize back to [0, 1]
    aug_pts = pts_transformed.copy()
    aug_pts[:, 0] /= w
    aug_pts[:, 1] /= h

    # Clamp to valid range and check if quad is still mostly visible
    aug_pts = np.clip(aug_pts, 0.0, 1.0)

    # Reject if any corner is pushed too far to the edge (quad collapsed)
    quad_w = np.max(aug_pts[:, 0]) - np.min(aug_pts[:, 0])
    quad_h = np.max(aug_pts[:, 1]) - np.min(aug_pts[:, 1])
    if quad_w < 0.05 or quad_h < 0.02:
        # Augmentation collapsed the quad — return original unchanged
        return img_gray.copy(), kpts.copy()

    return aug_img, aug_pts.flatten().astype(np.float32)


def load_paired_dataset(split: str = "train", augment: bool = False, augment_factor: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads all paired images and 4-corner keypoints for a given split ('train', 'val', 'test').

    Args:
        split: Dataset split ('train', 'val', 'test').
        augment: If True (recommended for training), multiplies dataset with
                 combined spatial + photometric + cutout augmentation.
        augment_factor: Number of augmented copies per training image (default 6).

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
            kpts = np.array([float(v) for v in tokens[5:13]], dtype=np.float32)
            if np.any(kpts < 0.0) or np.any(kpts > 1.0):
                continue

            # Load image in grayscale
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to thumbnail (128, 128)
            resized = cv2.resize(img, (config.THUMB_W, config.THUMB_H), interpolation=cv2.INTER_AREA)

            # Original sample (always included)
            images.append(np.expand_dims(resized, axis=-1))
            targets.append(kpts.tolist())

            # Augmented copies for training: spatial + photometric + cutout
            if augment and split == "train":
                for _ in range(augment_factor):
                    # First: spatial augmentation (transforms both image and keypoints)
                    aug_img, aug_kpts = apply_spatial_augmentation(resized, kpts)
                    # Then: photometric augmentation (doesn't change keypoints)
                    aug_img = apply_photometric_augmentation(aug_img)
                    # Then: CutOut with 50% probability (doesn't change keypoints)
                    if np.random.rand() > 0.5:
                        aug_img = apply_cutout(aug_img)
                    images.append(np.expand_dims(aug_img, axis=-1))
                    targets.append(aug_kpts.tolist())

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
