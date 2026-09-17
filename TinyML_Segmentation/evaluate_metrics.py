"""
evaluate_metrics.py - Evaluation Metrics for Keypoint Localization & Polygon IoU

Computes quantitative metrics:
1. Corner MAE (Mean Absolute Error) in normalized units and thumbnail pixels.
2. Quadrilateral Polygon IoU (Intersection over Union).
3. Angle estimation error (degrees difference of the counter orientation).
"""

import math
import numpy as np
import cv2

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
        "mean_angle_error_deg": float(np.mean(angle_errors)) if angle_errors else 0.0
    }
