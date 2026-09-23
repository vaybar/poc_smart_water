"""
config.py - TinyML Segmentation Configuration for ESP32 Classic

Defines memory budget, input thumbnail resolution, target output digit dimensions (64x32x1),
dataset paths, and training hyperparameters for water meter counter localization and segmentation.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

MCU_DIR = BASE_DIR / "mcu_c_code"
MCU_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Budget Constraints for ESP32 Classic (Xtensa LX6 @ 240 MHz, 520 KB SRAM, 4 MB PSRAM)
MAX_FLASH_BYTES = 80 * 1024         # Target < 80 KB Flash for segmenter model
MAX_TENSOR_ARENA_BYTES = 35 * 1024   # Target < 35 KB Internal SRAM Tensor Arena

# Model Input Specifications (Thumbnail for Localization)
THUMB_H = 128
THUMB_W = 128
THUMB_CHANNELS = 1                  # Grayscale to minimize RAM and capture time
THUMB_INPUT_SHAPE = (THUMB_H, THUMB_W, THUMB_CHANNELS)

# Target Digit Output Specifications (for TinyML_Digit_Classifier)
DIGIT_H = 64                        # Height = 64 px
DIGIT_W = 32                        # Width  = 32 px
DIGIT_CHANNELS = 1                  # Grayscale (64, 32, 1)
DEFAULT_NUM_DIGITS = 6              # Default number of mechanical wheels (5 or 6)

# Rectified Strip Dimensions (Height = 64, Width = NUM_DIGITS * 32)
STRIP_H = DIGIT_H
STRIP_W_6 = DEFAULT_NUM_DIGITS * DIGIT_W  # 192 px for 6 digits

# Model Architecture Parameters
ALPHA = 0.75                        # Width multiplier for Micro-Pose-CNN (0.25, 0.50, 0.75)
NUM_KEYPOINTS = 4                   # 4 corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
NUM_COORDINATES = NUM_KEYPOINTS * 2 # 8 coordinates (x1, y1, ..., x4, y4)
MAX_QUAD_OFFSET = 0.35              # Max corner offset from center (tanh * 0.35)

# Dataset Paths
DATASET_ROOT = ROOT_DIR / "water_meter"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 80
INITIAL_LR = 1e-3
MIN_LR = 1e-6
VAL_SPLIT = 0.15
SEED = 42


# Export Paths
FLOAT_MODEL_PATH = MODELS_DIR / "micro_corner_regressor.keras"
TFLITE_INT8_PATH = MODELS_DIR / "micro_corner_regressor_int8.tflite"
C_HEADER_PATH    = MCU_DIR / "corner_model_quantized.h"
