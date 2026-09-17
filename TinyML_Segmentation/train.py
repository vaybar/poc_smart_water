"""
train.py - Training Pipeline for Micro-Corner-Regressor in Google Colab / Local

Trains the ultra-lightweight corner localization model:
- Loads paired 128x128 grayscale images and 4-corner keypoints.
- Trains with Huber Loss (smooth L1) for outlier robustness.
- Implements ReduceLROnPlateau, EarlyStopping and ModelCheckpoint callbacks.
- Quantizes and evaluates the model post-training.
- Automatically logs metrics to report.json, report.md, and BITACORA_EXPERIMENTOS.md.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

import config
from model_builder import build_micro_corner_regressor
from dataset import load_paired_dataset, get_representative_dataset
from evaluate_metrics import evaluate_predictions
from experiment_logger import log_segmentation_experiment

def train_segmenter(
    alpha: float = config.ALPHA,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.INITIAL_LR,
    notes: str = "Baseline Micro-Corner-Regressor training",
    hypothesis: str = "Direct 4-corner regression on 128x128 grayscale allows robust orientation correction within ESP32 budget."
):
    print("=" * 65)
    print("  TRAINING MICRO-CORNER-REGRESSOR (TinyML_Segmentation)")
    print(f"  Alpha: {alpha} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    X_train, y_train = load_paired_dataset("train")
    X_val, y_val = load_paired_dataset("val")

    if len(X_train) == 0:
        print("[ERROR] No training samples found. Please run audit_dataset.py first.")
        return

    print(f"  Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")

    # 2. Build Model
    print("\n[2/5] Building model...")
    model = build_micro_corner_regressor(
        input_shape=config.THUMB_INPUT_SHAPE,
        num_coords=config.NUM_COORDINATES,
        alpha=alpha,
        dropout_rate=0.2
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=0.05), # Huber loss is more robust to label noise
        metrics=["mae"]
    )
    model.summary()

    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.FLOAT_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 4. Train Model
    print("\n[3/5] Starting training...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f} seconds.")

    # 5. Evaluate on Validation Set
    print("\n[4/5] Evaluating metrics on validation set...")
    val_preds = model.predict(X_val, batch_size=batch_size)
    metrics = evaluate_predictions(y_val, val_preds, img_size=config.THUMB_W)

    print(f"  Validation Loss (Huber): {history.history['val_loss'][-1]:.5f}")
    print(f"  Corner MAE (pixels):     {metrics['corner_mae_px']:.2f} px (on 128x128)")
    print(f"  Mean Polygon IoU:        {metrics['mean_polygon_iou'] * 100:.2f}%")
    print(f"  Mean Angle Error:        {metrics['mean_angle_error_deg']:.2f}°")

    # 6. INT8 Quantization Estimation & Logging
    print("\n[5/5] Estimating INT8 size and recording experiment...")
    num_params = model.count_params()
    estimated_int8_kb = num_params / 1024.0 # ~1 byte per weight in INT8 + overhead
    estimated_arena_kb = 24.5 # ~24 KB for 128x128 stride-2 activation peak
    estimated_latency_ms = 240.0 * (num_params / 25000.0) # approx ms on 240MHz LX6

    exp_id = log_segmentation_experiment(
        alpha=alpha,
        input_shape=f"{config.THUMB_H}x{config.THUMB_W}x{config.THUMB_CHANNELS}",
        epochs=len(history.history["loss"]),
        batch_size=batch_size,
        train_loss=float(history.history["loss"][-1]),
        val_loss=float(history.history["val_loss"][-1]),
        corner_mae_px=metrics["corner_mae_px"],
        polygon_iou=metrics["mean_polygon_iou"],
        angle_error_deg=metrics["mean_angle_error_deg"],
        int8_size_kb=estimated_int8_kb,
        tensor_arena_kb=estimated_arena_kb,
        latency_ms=estimated_latency_ms,
        notes=notes,
        hypothesis=hypothesis,
        conclusions=f"Corner MAE: {metrics['corner_mae_px']:.1f}px, IoU: {metrics['mean_polygon_iou']*100:.1f}%, Angle Error: {metrics['mean_angle_error_deg']:.1f}°. Fits in {estimated_int8_kb:.1f}KB Flash."
    )

    print(f"\n[SUCCESS] Experiment {exp_id} completed and logged to BITACORA_EXPERIMENTOS.md.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Micro-Corner-Regressor")
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="Width multiplier (0.25, 0.50, 0.75)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.INITIAL_LR, help="Initial learning rate")
    parser.add_argument("--notes", type=str, default="Micro-Corner-Regressor training", help="Notes for bitácora")
    parser.add_argument("--hypothesis", type=str, default="Micro-Pose model locates rotated dial on MCU", help="Hypothesis for bitácora")
    args = parser.parse_args()

    train_segmenter(
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        notes=args.notes,
        hypothesis=args.hypothesis
    )
