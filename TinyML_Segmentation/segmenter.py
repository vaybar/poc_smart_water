"""
segmenter.py - End-to-End Water Meter Segmentation Pipeline

Integrates:
1. Input frame downsampling to (128, 128, 1) grayscale.
2. Inference via Micro-Corner-Regressor (TFLite INT8 or Keras).
3. Bilinear Inverse Quadrilateral Mapping to horizontal strip (64 x [N*32]).
4. Mechanical wheel slot slicing into N individual (64, 32, 1) uint8 digit crops.
"""

from dataclasses import dataclass
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

import config
from warp_utils import rectify_counter_strip, slice_digits_from_strip

@dataclass
class SegmentationResult:
    digits: list[np.ndarray]            # N arrays of shape (64, 32, 1) uint8
    rectified_strip: np.ndarray         # Strip (64, N*32) uint8
    corners_original: np.ndarray        # 4 corners in original image pixels: (4, 2)
    execution_time_ms: float

class TinyMLSegmenter:
    """
    MCU-Compatible Segmentation Engine for Water Meter AMR.
    """
    def __init__(
        self,
        model_path: Path | str | None = None,
        num_digits: int = config.DEFAULT_NUM_DIGITS
    ):
        self.num_digits = num_digits
        self.digit_w = config.DIGIT_W
        self.digit_h = config.DIGIT_H
        self.target_strip_w = self.num_digits * self.digit_w
        self.target_strip_h = self.digit_h

        # Resolve model path (prefer INT8 TFLite if available, else Keras float)
        if model_path is None:
            if config.TFLITE_INT8_PATH.exists():
                model_path = config.TFLITE_INT8_PATH
            elif config.FLOAT_MODEL_PATH.exists():
                model_path = config.FLOAT_MODEL_PATH
            else:
                model_path = None

        self.model_path = model_path
        self.is_tflite = False
        self.interpreter = None
        self.keras_model = None

        if model_path is not None and Path(model_path).exists():
            if str(model_path).endswith(".tflite"):
                self.is_tflite = True
                self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
                self.interpreter.allocate_tensors()
                self.in_details = self.interpreter.get_input_details()
                self.out_details = self.interpreter.get_output_details()
                print(f"[TinyMLSegmenter] Loaded TFLite model from {model_path}")
            else:
                self.keras_model = tf.keras.models.load_model(str(model_path))
                print(f"[TinyMLSegmenter] Loaded Keras model from {model_path}")
        else:
            print("[TinyMLSegmenter] [WARN] No trained weights found. Instantiating in heuristic/fallback mode.")

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Executes end-to-end segmentation on an arbitrary water meter image.

        Args:
            image: numpy array (H, W, 3) BGR or (H, W) Grayscale.

        Returns:
            SegmentationResult with N digits of shape (64, 32, 1).
        """
        t0 = time.time()
        orig_h, orig_w = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Create 128x128 thumbnail
        thumb = cv2.resize(gray, (config.THUMB_W, config.THUMB_H), interpolation=cv2.INTER_AREA)

        # Step 2: Predict 4 corners
        if self.is_tflite and self.interpreter is not None:
            thumb_input = thumb.astype(np.float32)
            # Check quantization type
            if self.in_details[0]["dtype"] == np.int8:
                scale, zero_point = self.in_details[0]["quantization"]
                thumb_input = (thumb_input / scale + zero_point).astype(np.int8)
                thumb_input = np.expand_dims(thumb_input, axis=(0, -1))
            else:
                thumb_input = np.expand_dims(thumb_input, axis=(0, -1))

            self.interpreter.set_tensor(self.in_details[0]["index"], thumb_input)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.out_details[0]["index"])[0]
        elif self.keras_model is not None:
            thumb_input = np.expand_dims(thumb, axis=(0, -1)).astype(np.float32)
            preds = self.keras_model.predict(thumb_input, verbose=0)[0]
        else:
            # Fallback heuristic center box if model not yet trained
            preds = np.array([0.30, 0.40, 0.70, 0.40, 0.70, 0.60, 0.30, 0.60], dtype=np.float32)

        # Scale normalized coordinates back to original image dimensions
        norm_corners = preds.reshape(4, 2)
        orig_corners = norm_corners.copy()
        orig_corners[:, 0] *= orig_w
        orig_corners[:, 1] *= orig_h

        # Step 3: Bilinear Quadrilateral Rectification (64 x [N*32])
        rectified_strip = rectify_counter_strip(
            src_gray=gray,
            corners=orig_corners,
            target_h=self.target_strip_h,
            target_w=self.target_strip_w
        )

        # Step 4: Slicing into N individual (64, 32, 1) tiles
        digits = slice_digits_from_strip(
            strip=rectified_strip,
            num_digits=self.num_digits,
            digit_w=self.digit_w,
            digit_h=self.digit_h
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        return SegmentationResult(
            digits=digits,
            rectified_strip=rectified_strip,
            corners_original=orig_corners,
            execution_time_ms=elapsed_ms
        )
