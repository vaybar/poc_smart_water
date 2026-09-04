# TinyML Digit Classifier (< 256 KB - 500 KB RAM/Flash for Microcontrollers)

An ultra-lightweight Deep Learning digit classification pipeline designed for low-cost microcontrollers such as the **ESP32-S3**, **STM32**, and **ARM Cortex-M** series.

## Key Technical Specifications

- **Target Microcontroller:** ESP32-S3 (240 MHz dual-core Xtensa LX7, 512 KB SRAM)
- **Model Architecture:** Custom Micro-MobileNet with Depthwise Separable Convolutions & Width Scaling ($\alpha = 0.25$)
- **Input Dimensions:** $32 \times 32 \times 1$ (Grayscale)
- **Model Footprint (Flash):** ~30 KB – 110 KB (Full INT8 Quantized `.tflite` / `.h`)
- **RAM Tensor Arena:** ~28 KB – 35 KB SRAM
- **Inference Latency:** < 5–12 ms per frame on ESP32-S3 @ 240 MHz
- **Quantization:** Full INT8 (weights and activation tensors) with representative dataset calibration

---

## Memory Comparison: Standard MobileNet vs. Micro-MobileNet

| Feature | Standard MobileNetV3-Small | Micro-MobileNet ($\alpha = 0.25$) | Reduction Factor |
| :--- | :--- | :--- | :--- |
| **Input Shape** | $96 \times 96 \times 3$ (RGB) | $32 \times 32 \times 1$ (Grayscale) | **9x input data reduction** |
| **Flash Size (INT8)** | 1,295 KB (1.3 MB) | **~50 KB - 110 KB** | **> 12x smaller** |
| **Tensor Arena (RAM)** | ~350 KB | **~35 KB** | **10x RAM savings** |
| **Compatibility** | RPi / Jetson / PC | **ESP32-S3 / STM32 MCUs** | **Enables MCU deployment** |

---

## Project Structure

```
TinyML_Digit_Classifier/
├── config.py                 # Central config (dimensions, alpha, thresholds, paths)
├── model_builder.py          # Micro-MobileNet architecture builder
├── dataset.py                # Dataset loader, augmentation & INT8 calibration generator
├── train.py                  # Training pipeline with learning rate schedules
├── quantize_and_export.py    # INT8 post-training quantization & C header (.h) exporter
├── evaluate_metrics.py       # Metrics report (Accuracy, Precision/Recall, Confusion Matrix, Latency)
├── test_inference.py         # Python TFLite inference simulator & benchmark
├── FAQ.md                    # FAQ & technical architecture design decisions
├── README.md                 # Complete documentation & usage guide
└── esp32_s3_example/
    └── esp32_s3_digit_classifier.ino  # ESP32-S3 Arduino sketch for edge AI deployment
```

---

## Quick Start Guide

### 1. Installation Requirements
Ensure you have Python 3.9+ and TensorFlow 2.x installed:
```bash
pip install tensorflow numpy opencv-python scikit-learn
```

### 2. Train the Micro-MobileNet Model
Train the model with width multiplier $\alpha = 0.25$:
```bash
python train.py --alpha 0.25 --epochs 30
```
This saves the Float32 model to `models/micro_mobilenet.keras`.

### 3. INT8 Quantization & C Header Export
Perform post-training INT8 quantization and generate the C byte array header file:
```bash
python quantize_and_export.py
```
This produces:
- `models/micro_mobilenet_int8.tflite`
- `models/digit_model_quantized.h`

### 4. Benchmark & Test Python Inference
Simulate microcontroller inference on a single image or run full test set benchmark:
```bash
# Benchmark test set
python test_inference.py --benchmark

# Single image test
python test_inference.py --image path/to/digit_sample.png
```

### 5. Deploy to ESP32-S3 Microcontroller
1. Open `esp32_s3_example/esp32_s3_digit_classifier.ino` in Arduino IDE or PlatformIO.
2. Ensure `models/digit_model_quantized.h` is copied into the sketch folder.
3. Install the **TensorFlowLite_ESP32** or **tflite-micro** library.
4. Select Board: **ESP32S3 Dev Module**.
5. Compile and Flash to your ESP32-S3!
