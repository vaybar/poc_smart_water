"""
config.py - TinyML Digit Classifier Configuration

Defines memory budget, architecture parameters, dataset paths,
and target deployment specifications for ESP32-S3 microcontroller.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Microcontroller Target Budget Constraints
MAX_FLASH_BYTES = 256 * 1024       # Target < 256 KB Flash (upper limit 500 KB)
MAX_TENSOR_ARENA_BYTES = 40 * 1024  # Target < 40 KB RAM Tensor Arena

# Model Architecture Parameters
IMG_H = 64                        # Height = 64 px (native aspect ratio 2:1)
IMG_W = 32                        # Width  = 32 px
CHANNELS = 1                      # Grayscale input (1 channel vs 3)
INPUT_SHAPE = (IMG_H, IMG_W, CHANNELS)
NUM_CLASSES = 10                  # Digits 0 through 9
ALPHA = 0.25                      # Width multiplier for Micro-MobileNet scaling

# Dataset Settings
# Points to dataset_mobilenet in parent directory if available, or local data folder
PARENT_DATASET = BASE_DIR.parent / "dataset_mobilenet"
DATASET_DIR = PARENT_DATASET if PARENT_DATASET.exists() else BASE_DIR / "data"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_LR = 1e-3
MIN_LR = 1e-6
VAL_SPLIT = 0.2
SEED = 42

# Export File Paths
FLOAT_MODEL_PATH = MODELS_DIR / "micro_mobilenet.keras"
TFLITE_INT8_PATH = MODELS_DIR / "micro_mobilenet_int8.tflite"
C_HEADER_PATH    = MODELS_DIR / "digit_model_quantized.h"
