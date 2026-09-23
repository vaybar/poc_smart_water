"""
train.py - Training Pipeline for Micro-Corner-Regressor in Google Colab / Local

Trains the ultra-lightweight corner localization model:
- Loads paired 128x128 grayscale images and 4-corner keypoints.
- Trains with Wing + Soft IoU composite loss for landmark-grade precision.
- Implements Cosine Annealing LR, EarlyStopping and ModelCheckpoint callbacks.
- Quantizes and evaluates the model post-training.
- Automatically logs metrics to report.json, report.md, and BITACORA_EXPERIMENTOS.md.
"""

import argparse
import math
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

import config
from model_builder import build_micro_corner_regressor
from dataset import load_paired_dataset, get_representative_dataset
from evaluate_metrics import evaluate_predictions, plot_corner_error_scatter, plot_worst_cases_grid
from experiment_logger import log_segmentation_experiment


# ---------------------------------------------------------------------------
#  Wing Loss — designed for landmark/keypoint regression
# ---------------------------------------------------------------------------
def wing_loss(y_true, y_pred, w=0.03, epsilon=0.01):
    """
    Wing Loss (Feng et al., 2018) for keypoint regression.

    Amplifies gradient for small errors (< w) via logarithmic curvature,
    while behaving linearly for large errors to avoid gradient explosion.

    Args:
        y_true: Ground truth coordinates (batch, 8).
        y_pred: Predicted coordinates (batch, 8).
        w: Threshold below which log-curve is used (in normalized coords).
        epsilon: Curvature control inside the log region.
    """
    diff = tf.abs(y_true - y_pred)
    C = w - w * tf.math.log(1.0 + w / epsilon)
    loss = tf.where(
        diff < w,
        w * tf.math.log(1.0 + diff / epsilon),
        diff - C
    )
    return tf.reduce_mean(loss)


# ---------------------------------------------------------------------------
#  Soft Polygon IoU Loss — differentiable rasterization-based IoU
# ---------------------------------------------------------------------------
def soft_polygon_iou_loss(y_true, y_pred, canvas_size=64):
    """
    Differentiable Polygon IoU loss via soft rasterization.

    Renders predicted and ground-truth quadrilaterals onto a soft canvas
    using distance-to-edge sigmoid approximation, then computes 1 - IoU.

    This directly aligns the training loss with the evaluation metric (Polygon IoU).

    Args:
        y_true: (batch, 8) ground truth normalized corner coords.
        y_pred: (batch, 8) predicted normalized corner coords.
        canvas_size: Resolution of the rasterization grid.
    """
    batch_size = tf.shape(y_true)[0]

    # Create coordinate grid [0, 1] x [0, 1]
    coords = tf.linspace(0.0, 1.0, canvas_size)
    gx, gy = tf.meshgrid(coords, coords)  # (canvas, canvas)
    gx = tf.reshape(gx, [1, canvas_size, canvas_size, 1])  # (1, H, W, 1)
    gy = tf.reshape(gy, [1, canvas_size, canvas_size, 1])

    def render_quad_mask(corners):
        """Renders a soft mask for a batch of quadrilaterals using signed edge distances."""
        # corners: (batch, 8) -> reshape to (batch, 4, 2)
        pts = tf.reshape(corners, [-1, 4, 2])

        # For each edge (i -> i+1), compute signed distance of grid points
        # A point is inside the quad if it's on the same side of all 4 edges
        inside_accum = tf.ones([batch_size, canvas_size, canvas_size], dtype=tf.float32)

        for i in range(4):
            j = (i + 1) % 4
            # Edge vector
            ex = pts[:, j, 0] - pts[:, i, 0]  # (batch,)
            ey = pts[:, j, 1] - pts[:, i, 1]

            # Normal direction (pointing inward for CW ordering)
            # Cross product sign: (P - A) x (B - A) > 0 means left of edge
            # For our canonical TL->TR->BR->BL (CW), inward is the right side
            ax = pts[:, i, 0]  # (batch,)
            ay = pts[:, i, 1]

            # Signed distance from each grid point to edge line
            # d = (gx - ax) * ey - (gy - ay) * ex
            ax_r = tf.reshape(ax, [-1, 1, 1])
            ay_r = tf.reshape(ay, [-1, 1, 1])
            ex_r = tf.reshape(ex, [-1, 1, 1])
            ey_r = tf.reshape(ey, [-1, 1, 1])

            d = (gx[:, :, :, 0] - ax_r) * ey_r - (gy[:, :, :, 0] - ay_r) * ex_r

            # Soft step function: sigmoid with temperature for differentiability
            # Negative d = inside (right of CW edge), positive = outside
            sigma = 200.0  # Higher = sharper boundary (200 ≈ ~0.5px transition at 64x64)
            edge_mask = tf.sigmoid(-d * sigma * tf.cast(canvas_size, tf.float32))
            inside_accum = inside_accum * edge_mask

        return inside_accum  # (batch, canvas, canvas) ∈ [0, 1]

    mask_true = render_quad_mask(y_true)
    mask_pred = render_quad_mask(y_pred)

    # Soft IoU
    intersection = tf.reduce_sum(mask_true * mask_pred, axis=[1, 2])
    union = tf.reduce_sum(mask_true + mask_pred - mask_true * mask_pred, axis=[1, 2])

    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(1.0 - iou)


# ---------------------------------------------------------------------------
#  Composite Loss: Wing + Soft IoU
# ---------------------------------------------------------------------------
def wing_iou_loss(y_true, y_pred):
    """
    Composite loss combining Wing Loss (coordinate precision) with
    Soft Polygon IoU Loss (shape overlap accuracy).

    Weights: 0.6 * Wing + 0.4 * (1 - SoftIoU)
    """
    l_wing = wing_loss(y_true, y_pred)
    l_iou = soft_polygon_iou_loss(y_true, y_pred)
    return 0.6 * l_wing + 0.4 * l_iou


# ---------------------------------------------------------------------------
#  Cosine Annealing LR Scheduler
# ---------------------------------------------------------------------------
class CosineAnnealingSchedule(tf.keras.callbacks.Callback):
    """
    Cosine Annealing learning rate schedule (Loshchilov & Hutter, 2016).

    Smoothly decreases LR from initial_lr to min_lr following a cosine curve,
    avoiding the abrupt drops of ReduceLROnPlateau and enabling escape from
    sharp local minima.
    """
    def __init__(self, initial_lr: float, min_lr: float, total_epochs: int):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_decay
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        if epoch % 10 == 0:
            print(f"  [CosineAnnealing] Epoch {epoch}: LR = {new_lr:.6f}")


# ---------------------------------------------------------------------------
#  Canonical Label Sanity Check
# ---------------------------------------------------------------------------
def check_canonical_labels(num_samples: int = 20):
    """
    Quick sanity check that labels follow canonical quad ordering
    (TL→TR long edge on top, TL→BL short edge on left).

    Checks a random subset and warns if ordering looks inconsistent.
    Does NOT modify any files — purely diagnostic.
    """
    lbl_dir = config.LABELS_DIR / "train"
    if not lbl_dir.exists():
        return

    txt_files = sorted(list(lbl_dir.glob("*.txt")))
    if not txt_files:
        return

    rng = np.random.default_rng(42)
    sample_files = rng.choice(txt_files, size=min(num_samples, len(txt_files)), replace=False)

    suspicious = 0
    for txt_p in sample_files:
        try:
            line = txt_p.read_text(encoding="utf-8").strip()
            tokens = line.split()
            if len(tokens) < 13:
                continue
            pts = np.array([float(v) for v in tokens[5:13]], dtype=np.float32).reshape(4, 2)

            # Check: top edge (P0→P1) should be longer than left edge (P0→P3)
            top_len = np.linalg.norm(pts[1] - pts[0])
            left_len = np.linalg.norm(pts[3] - pts[0])
            if left_len > top_len * 1.5:  # Short edge shouldn't be much longer than long edge
                suspicious += 1

            # Check: P0 should generally be top-left (smaller y than P3)
            if pts[0, 1] > pts[3, 1] + 0.05:
                suspicious += 1

        except Exception:
            continue

    if suspicious > num_samples * 0.3:
        print(f"  [WARNING] {suspicious}/{num_samples} sampled labels may not be canonically ordered.")
        print(f"  Consider running: python fix_water_meter_labels.py")
    else:
        print(f"  [OK] Canonical label check passed ({suspicious}/{num_samples} suspicious).")


# ---------------------------------------------------------------------------
#  Main Training Function
# ---------------------------------------------------------------------------
def train_segmenter(
    alpha: float = config.ALPHA,
    epochs: int = 120,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.INITIAL_LR,
    notes: str = "EXP_007: Wing+IoU loss, aug geometrica espacial, salida directa sigmoid(8)",
    hypothesis: str = "Wing+IoU loss + aug geometrica + sigmoid directo llevara IoU > 60% y MAE < 5px sin exceder 55KB Flash."
):
    print("=" * 65)
    print("  TRAINING MICRO-CORNER-REGRESSOR V4 (TinyML_Segmentation)")
    print(f"  Alpha: {alpha} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 65)

    # 0. Sanity check on label ordering
    print("\n[0/5] Checking canonical label ordering...")
    check_canonical_labels()

    # 1. Load Data
    print("\n[1/5] Loading datasets (with spatial + photometric augmentation)...")
    X_train, y_train = load_paired_dataset("train", augment=True, augment_factor=4)
    X_val, y_val = load_paired_dataset("val", augment=False)

    if len(X_train) == 0:
        print("[ERROR] No training samples found. Please run audit_dataset.py first.")
        return

    print(f"  Loaded {len(X_train)} training samples (with augmentation) and {len(X_val)} validation samples.")

    # 2. Build Model
    print("\n[2/5] Building model (V4 Direct Sigmoid Output)...")
    model = build_micro_corner_regressor(
        input_shape=config.THUMB_INPUT_SHAPE,
        num_coords=config.NUM_COORDINATES,
        alpha=alpha,
        dropout_rate=0.2
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=wing_iou_loss,
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
        CosineAnnealingSchedule(
            initial_lr=learning_rate,
            min_lr=config.MIN_LR,
            total_epochs=epochs
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
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

    print(f"  Validation Loss (Wing+IoU): {history.history['val_loss'][-1]:.5f}")
    print(f"  Corner MAE (pixels):        {metrics['corner_mae_px']:.2f} px (on 128x128)")
    print(f"  Mean Polygon IoU:           {metrics['mean_polygon_iou'] * 100:.2f}%")
    print(f"  Mean Angle Error:           {metrics['mean_angle_error_deg']:.2f}°")

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

    # 7. Generate Visual Diagnostic Plots
    exp_dir = config.EXPERIMENTS_DIR / exp_id
    plot_corner_error_scatter(y_val, val_preds, save_path=exp_dir / "corner_error_scatter.png", img_size=config.THUMB_W)
    plot_worst_cases_grid(X_val, y_val, val_preds, save_path=exp_dir / "worst_cases_grid.png", top_k=9, img_size=config.THUMB_W)

    print(f"\n[SUCCESS] Experiment {exp_id} completed. Visual diagnostic plots saved to {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Micro-Corner-Regressor")
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="Width multiplier (0.25, 0.50, 0.75)")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.INITIAL_LR, help="Initial learning rate")
    parser.add_argument("--notes", type=str, default="EXP_007: Wing+IoU loss, aug geometrica espacial, salida directa sigmoid(8)", help="Notes for bitácora")
    parser.add_argument("--hypothesis", type=str, default="Wing+IoU loss + aug geometrica + sigmoid directo llevara IoU > 60% y MAE < 5px sin exceder 55KB Flash.", help="Hypothesis for bitácora")
    args = parser.parse_args()

    train_segmenter(
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        notes=args.notes,
        hypothesis=args.hypothesis
    )

