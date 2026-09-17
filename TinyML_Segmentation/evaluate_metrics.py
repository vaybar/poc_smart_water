"""
evaluate_metrics.py - Evaluation Metrics & Visual Diagnostic Plots for TinyML_Segmentation

Computes quantitative metrics:
1. Corner MAE (Mean Absolute Error) in normalized units and thumbnail pixels.
2. Quadrilateral Polygon IoU (Intersection over Union).
3. Angle estimation error (degrees difference of the counter orientation).

Generates visual diagnostic plots (analogs to Confusion Matrix):
1. plot_corner_error_scatter: 2D Residual Error Scatter plot for all 4 corners.
2. plot_worst_cases_grid: 3x3 Grid of worst IoU validation samples (Green=True, Red=Pred).
"""

import math
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_corner_mae(y_true: np.ndarray, y_pred: np.ndarray, img_size: int = 128) -> tuple[float, float]:
    """
    Computes Corner MAE (normalized and in pixels).
    y_true, y_pred: (N, 8) normalized coordinates.
    """
    diff = np.abs(y_true - y_pred)
    mae_norm = float(np.mean(diff))
    mae_px = mae_norm * img_size
    return mae_norm, mae_px

def compute_polygon_iou(quad_true: np.ndarray, quad_pred: np.ndarray, canvas_size: int = 256) -> float:
    """
    Computes Intersection over Union (IoU) between two 4-point polygons.
    quad_true, quad_pred: (4, 2) in normalized [0, 1] coordinates.
    """
    pts_true = (quad_true * canvas_size).astype(np.int32)
    pts_pred = (quad_pred * canvas_size).astype(np.int32)

    mask_true = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    mask_pred = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    cv2.fillPoly(mask_true, [pts_true], 1)
    cv2.fillPoly(mask_pred, [pts_pred], 1)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, img_size: int = 128) -> dict:
    """
    Runs full evaluation over a test/val batch.
    """
    mae_norm, mae_px = compute_corner_mae(y_true, y_pred, img_size)

    ious = []
    angle_errors = []

    for i in range(len(y_true)):
        q_true = y_true[i].reshape(4, 2)
        q_pred = y_pred[i].reshape(4, 2)

        iou = compute_polygon_iou(q_true, q_pred)
        ious.append(iou)

        # Angle of top edge
        ang_true = math.degrees(math.atan2(q_true[1, 1] - q_true[0, 1], q_true[1, 0] - q_true[0, 0]))
        ang_pred = math.degrees(math.atan2(q_pred[1, 1] - q_pred[0, 1], q_pred[1, 0] - q_pred[0, 0]))
        diff_ang = abs(ang_true - ang_pred)
        if diff_ang > 180:
            diff_ang = 360 - diff_ang
        angle_errors.append(diff_ang)

    return {
        "corner_mae_norm": float(mae_norm),
        "corner_mae_px": float(mae_px),
        "mean_polygon_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_angle_error_deg": float(np.mean(angle_errors)) if angle_errors else 0.0,
        "per_sample_ious": np.array(ious),
        "per_sample_angle_errors": np.array(angle_errors)
    }

def plot_corner_error_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | str, img_size: int = 128):
    """
    Generates a 2D Residual Error Scatter Plot for all 4 corners (TL, TR, BR, BL).
    Serves as the 2D spatial analog to a Confusion Matrix.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    # 4 corners: TL, TR, BR, BL
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
    labels = ["P0: Top-Left", "P1: Top-Right", "P2: Bottom-Right", "P3: Bottom-Left"]

    # Compute errors in pixels
    y_t_px = y_true * img_size
    y_p_px = y_pred * img_size

    for k in range(4):
        dx = y_p_px[:, k * 2] - y_t_px[:, k * 2]
        dy = y_p_px[:, k * 2 + 1] - y_t_px[:, k * 2 + 1]

        ax.scatter(dx, dy, alpha=0.6, color=colors[k], label=f"{labels[k]} (Bias: {dx.mean():.1f}, {dy.mean():.1f}px)", edgecolors="none", s=25)

    # Add reference circles for 2px, 5px, 10px error radii
    for r, linestyle, color_r in [(2.0, ":", "gray"), (5.0, "--", "orange"), (10.0, "-", "red")]:
        circle = plt.Circle((0, 0), r, fill=False, linestyle=linestyle, color=color_r, alpha=0.7, label=f"R = {r:.0f}px error limit")
        ax.add_patch(circle)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("2D Corner Residual Error Distribution (Spatial Confusion Matrix Analog)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("dx = x_pred - x_true (pixels)", fontsize=10)
    ax.set_ylabel("dy = y_pred - y_true (pixels)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate_metrics] Corner Error Scatter plot saved to {save_path}")

def plot_worst_cases_grid(X_val: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | str, top_k: int = 9, img_size: int = 128):
    """
    Generates a 3x3 Grid of the worst-performing validation samples (lowest IoU).
    Visual analog to misclassification error analysis.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_predictions(y_true, y_pred, img_size=img_size)
    ious = metrics["per_sample_ious"]

    # Sort indices by IoU (ascending order -> worst cases first)
    worst_indices = np.argsort(ious)[:top_k]

    rows = int(math.ceil(math.sqrt(top_k)))
    cols = int(math.ceil(top_k / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=120)
    axes = axes.flatten() if top_k > 1 else [axes]

    for idx, sample_idx in enumerate(worst_indices):
        ax = axes[idx]
        img = X_val[sample_idx]
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        ax.imshow(img, cmap="gray")

        # True quad (Green)
        q_true = (y_true[sample_idx].reshape(4, 2) * img_size)
        pts_t = np.vstack([q_true, q_true[0]])
        ax.plot(pts_t[:, 0], pts_t[:, 1], color="lime", linewidth=1.8, label="True Quad")

        # Pred quad (Red)
        q_pred = (y_pred[sample_idx].reshape(4, 2) * img_size)
        pts_p = np.vstack([q_pred, q_pred[0]])
        ax.plot(pts_p[:, 0], pts_p[:, 1], color="red", linewidth=1.8, linestyle="--", label="Pred Quad")

        iou_val = ious[sample_idx] * 100.0
        mae_val = np.mean(np.abs(q_true - q_pred))

        ax.set_title(f"Worst #{idx+1} | IoU: {iou_val:.1f}% | MAE: {mae_val:.1f}px", fontsize=9, color="darkred", fontweight="bold")
        ax.axis("off")

    plt.suptitle("Worst-Performing Validation Samples (Error Diagnosis)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate_metrics] Worst Cases Diagnostic Grid saved to {save_path}")
