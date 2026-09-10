"""
dataset.py - Dataset Loading, Augmentation & Representative Calibration Generator

Handles dataset loading for digit classification (0-9). Supports:
1. Parent workspace digit dataset (`dataset_mobilenet`, `dataset_digitos_5`, `dataset_digitos_6`)
2. Keras MNIST fallback dataset if custom directory is not provided
3. Grayscale 32x32 image preprocessing
4. Data augmentation pipeline
5. Representative data generator for INT8 post-training quantization
"""

import os
import glob
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import config

def preprocess_digit_image(img: np.ndarray, target_size=(config.IMG_W, config.IMG_H)) -> np.ndarray:
    """
    Preprocess a single image BGR/RGB/Grayscale to (IMG_H, IMG_W, 1) uint8 numpy array.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img
        
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    resized_3d = np.expand_dims(resized, axis=-1)  # (32, 32, 1)
    return resized_3d.astype(np.uint8)

def load_images_from_folder(folder_path: Path):
    """Utility to load all digit images from 0..9 subfolders."""
    images, labels = [], []
    for digit in range(10):
        digit_dir = folder_path / str(digit)
        if not digit_dir.exists():
            continue
        # Support png, jpg, jpeg, bmp
        for img_file in digit_dir.glob("*"):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                img = cv2.imread(str(img_file))
                if img is not None:
                    proc_img = preprocess_digit_image(img)
                    images.append(proc_img)
                    labels.append(digit)
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)

def generate_focused_augmentations(
    images: np.ndarray,
    labels: np.ndarray,
    target_classes: list = [1, 7],
    samples_per_class: dict = {1: 800, 7: 1000},
    max_shear_deg: float = 12.0
):
    """
    Generates synthetic tilted/sheared and rotated samples specifically
    for vulnerable classes (e.g. '1' and '7') to enforce structural invariance
    to camera slant and serifa orientation.
    """
    extra_images = []
    extra_labels = []
    rng = np.random.default_rng(config.SEED)

    for cls in target_classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        n_samples = samples_per_class.get(cls, 800)
        print(f"  Generating {n_samples} focused shear/tilt augmentations for digit '{cls}'...")
        chosen_indices = rng.choice(cls_indices, size=n_samples, replace=True)

        for idx in chosen_indices:
            orig = images[idx]
            h, w = orig.shape[:2]

            # Random shear angle between -max_shear_deg and +max_shear_deg
            shear_deg = rng.uniform(-max_shear_deg, max_shear_deg)
            shear = np.tan(np.radians(shear_deg))

            # Random slight rotation (-6 to +6 deg)
            rot_deg = rng.uniform(-6.0, 6.0)

            # Affine matrix for centered horizontal shear
            M_shear = np.array([
                [1.0, shear, -shear * (h / 2.0)],
                [0.0, 1.0, 0.0]
            ], dtype=np.float32)

            warped = cv2.warpAffine(orig, M_shear, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Apply rotation
            M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_deg, 1.0)
            warped = cv2.warpAffine(warped, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

            if len(warped.shape) == 2:
                warped = np.expand_dims(warped, axis=-1)

            extra_images.append(warped.astype(np.uint8))
            extra_labels.append(cls)

    if extra_images:
        extra_images = np.array(extra_images, dtype=np.uint8)
        extra_labels = np.array(extra_labels, dtype=np.int64)
        print(f"  --> Added {len(extra_images)} total focused augmented samples to training set.")
        combined_images = np.concatenate([images, extra_images], axis=0)
        combined_labels = np.concatenate([labels, extra_labels], axis=0)

        # Shuffle combined set
        p = rng.permutation(len(combined_images))
        return combined_images[p], combined_labels[p]

    return images, labels

def load_digit_dataset(dataset_path: Path = config.DATASET_DIR, augment_focused: bool = True):
    """
    Loads dataset from custom directory (organized as 'train/0..9', 'val/0..9', 'test/0..9'
    or flat class subfolders '0'..'9') or loads MNIST as a fallback dataset.
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    dataset_path = Path(dataset_path)
    
    # Case 1: Structured train/val subfolders exist (like dataset_mobilenet)
    train_dir = dataset_path / "train"
    val_dir   = dataset_path / "val"
    test_dir  = dataset_path / "test"
    
    if train_dir.exists() and any((train_dir / str(d)).exists() for d in range(10)):
        print(f"Loading split dataset from: {dataset_path}")
        x_train, y_train = load_images_from_folder(train_dir)
        print(f"  --> Train set (raw): {len(x_train)} images")
        
        if val_dir.exists() and any((val_dir / str(d)).exists() for d in range(10)):
            x_val, y_val = load_images_from_folder(val_dir)
            print(f"  --> Val set        : {len(x_val)} images")
        else:
            # Split train into train/val
            n = len(x_train)
            idx = np.random.permutation(n)
            n_tr = int(n * 0.85)
            x_val, y_val = x_train[idx[n_tr:]], y_train[idx[n_tr:]]
            x_train, y_train = x_train[idx[:n_tr]], y_train[idx[:n_tr]]
            
        if augment_focused:
            print("Applying focused shear/tilt data augmentation for classes 1 and 7...")
            x_train, y_train = generate_focused_augmentations(x_train, y_train)
            print(f"  --> Train set (augmented): {len(x_train)} images")

        if test_dir.exists() and any((test_dir / str(d)).exists() for d in range(10)):
            x_test, y_test = load_images_from_folder(test_dir)
            print(f"  --> Test set       : {len(x_test)} images")
        else:
            x_test, y_test = x_val, y_val
            
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # Case 2: Flat subfolders '0'..'9' exist
    subfolders_exist = dataset_path.exists() and all((dataset_path / str(digit)).exists() for digit in range(10))
    if subfolders_exist:
        print(f"Loading custom flat digit dataset from: {dataset_path}")
        images, labels = load_images_from_folder(dataset_path)
        print(f"Loaded {len(images)} custom images.")
        
        # Shuffle and split
        indices = np.arange(len(images))
        np.random.seed(config.SEED)
        np.random.shuffle(indices)
        images, labels = images[indices], labels[indices]
        
        n_total = len(images)
        n_train = int(n_total * 0.70)
        n_val   = int(n_total * 0.15)
        
        x_train, y_train = images[:n_train], labels[:n_train]
        x_val, y_val     = images[n_train:n_train+n_val], labels[n_train:n_train+n_val]
        x_test, y_test   = images[n_train+n_val:], labels[n_train+n_val:]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # Fallback to MNIST resized to 32x32x1
    print("Custom dataset directory not found or incomplete. Loading MNIST benchmark dataset...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    
    # Resize MNIST (28x28) to target shape (32x32x1)
    def resize_batch(imgs):
        resized_list = []
        for img in imgs:
            r = cv2.resize(img, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_AREA)
            resized_list.append(np.expand_dims(r, axis=-1))
        return np.array(resized_list, dtype=np.uint8)
    
    x_tr_32 = resize_batch(x_tr[:10000])  # Use subset for fast lightweight processing
    y_tr_10 = y_tr[:10000]
    x_te_32 = resize_batch(x_te[:2000])
    y_te_10 = y_te[:2000]
    
    n_train = int(len(x_tr_32) * 0.8)
    x_train, y_train = x_tr_32[:n_train], y_tr_10[:n_train]
    x_val, y_val     = x_tr_32[n_train:], y_tr_10[n_train:]
    x_test, y_test   = x_te_32, y_te_10
    
    print(f"Dataset split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_data_augmentation():
    """
    Data augmentation sequential layer for 32x32x1 inputs.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08, fill_mode="constant"),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
        tf.keras.layers.RandomZoom(0.08, fill_mode="constant"),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

def get_representative_dataset_generator(x_calibration: np.ndarray, num_samples=100):
    """
    Yields sample inputs for INT8 post-training quantization calibration.
    """
    def representative_dataset():
        indices = np.random.choice(len(x_calibration), min(num_samples, len(x_calibration)), replace=False)
        for idx in indices:
            # Expand batch dimension: (1, 32, 32, 1) float32 in range [0.0, 255.0] or [0.0, 1.0]
            sample = np.expand_dims(x_calibration[idx], axis=0).astype(np.float32)
            yield [sample]
    return representative_dataset

if __name__ == "__main__":
    (x_tr, y_tr), (x_v, y_v), (x_te, y_te) = load_digit_dataset()
    print("x_train shape:", x_tr.shape, "dtype:", x_tr.dtype)
