# 🔬 TinyML Digit Classifier: Experimental Suite & Microcontroller Optimization

An ultra-lightweight Deep Learning pipeline designed for digit classification in smart water meters (AMR - Automatic Meter Reading), optimized to run on low-cost microcontrollers such as **ESP32-S3**, **STM32**, and **ARM Cortex-M** series.

The objective of this experimental campaign is to **empirically demonstrate that high accuracy (> 96.7%) can be achieved on embedded hardware using extremely compact convolutional networks (Micro-MobileNet)** consuming minimal memory resources.

---

## 📊 Dataset Origin & Citation

The training and validation images were extracted and cropped from the public dataset documented in:
* **Paper / Reference:** [Nature Scientific Data (2026) - s41597-026-06809-z](https://www.nature.com/articles/s41597-026-06809-z)
* **Preprocessing:** Grayscale conversion, normalization, class rebalancing, and aspect ratio adaptation to `64x32` (vertical orientation matching meter roller digits).

---

## 📈 Experimental Comparative Table (`EXP_001` - `EXP_010`)

The entire project evolution is reproducibly tracked in `experiments_log.csv` and `BITACORA_EXPERIMENTOS.md`:

| Exp ID | Date | Alpha ($\alpha$) | Input | Epochs | Float32 Acc | INT8 Acc | INT8 Loss | Macro F1 | Flash (KB) | RAM Arena (KB) | Key Milestone / Change |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `EXP_001` | 2026-09-04 | 0.25 | `32x32x1` | 30 | 87.69% | **87.25%** | 0.44% | - | **12.85 KB** | **13.86 KB** | Baseline model with $\alpha=0.25$ |
| `EXP_005` | 2026-09-08 | 0.25 | `32x32x1` | 30 | 87.13% | **87.79%** | -0.66% | 0.851 | **12.85 KB** | **13.86 KB** | Hyperparameter tuning |
| `EXP_006` | 2026-09-08 | 0.35 | `32x32x1` | 30 | 87.79% | **87.87%** | -0.08% | 0.854 | **14.24 KB** | **14.27 KB** | Initial capacity scaling ($\alpha=0.35$) |
| `EXP_007` | 2026-09-08 | 0.35 | `64x32x1` | 30 | 91.89% | **91.94%** | -0.05% | 0.899 | **14.24 KB** | **14.27 KB** | **Key Breakthrough:** Resolution shift to `64x32` (+4.07% Acc) |
| `EXP_008` | 2026-09-10 | 0.35 | `64x32x1` | 30 | 92.60% | **92.45%** | 0.15% | 0.923 | **14.24 KB** | **14.27 KB** | Rebalancing + Data Augmentation Shear (1/7) |
| `EXP_009` | 2026-09-12 | 0.50 | `64x32x1` | 30 | 94.36% | **94.45%** | -0.08% | 0.925 | **16.85 KB** | **15.06 KB** | Capacity scaling ($\alpha=0.50$) with cleaned dataset |
| `EXP_010` | 2026-09-12 | 0.75 | `64x32x1` | 35 | 96.70% | **96.73%** | -0.03% | **0.959** | **22.45 KB** | **16.73 KB** | **Optimal Model:** $\alpha=0.75$, 35 epochs & balanced *class weights* |

---

## 🎯 Pareto Frontier Analysis

The trade-off between classification accuracy and embedded hardware resource consumption is summarized in the **Pareto Frontier chart**:

![Pareto Frontier](experiments/pareto_tradeoff.png)

### 💡 Key Experimental Insights:
1. **Aspect Ratio Optimization (`64x32` vs `32x32`):** Transitioning from square `32x32` to vertical `64x32` eliminated vertical distortion on roller digits, yielding a +4.07% accuracy boost without increasing model parameters.
2. **Capacity Scaling ($\alpha=0.25 \to 0.75$):** Scaling the width multiplier $\alpha$ increased accuracy from **87.79%** to **96.73%**.
3. **Full INT8 Quantization Preservation:** Post-Training Quantization (PTQ) loss is virtually non-existent ($< 0.1\%$), confirming full suitability for microcontroller deployment without floating-point units (FPUs).
4. **Embedded Footprint (ESP32-S3):** Even the highest-performing model (`EXP_010`) requires only **22.45 KB Flash** (target $< 256\text{ KB}$) and **16.73 KB RAM Arena** (target $< 40\text{ KB}$), guaranteeing fast inference @ 240 MHz.

---

## 🛠️ Experimental Tooling & Scripts

* **[train.py](train.py):** Training pipeline with configurable width multiplier $\alpha$, epochs, batch size, random seeds (`SEED=42`), and class weighting.
* **[quantize_and_export.py](quantize_and_export.py):** Full INT8 quantization with representative dataset calibration and C byte array export (`digit_model_quantized.h`).
* **[experiment_logger.py](experiment_logger.py):** Automated logging to [BITACORA_EXPERIMENTOS.md](BITACORA_EXPERIMENTOS.md) and [experiments_log.csv](experiments_log.csv).
* **[plot_pareto_front.py](plot_pareto_front.py):** Pareto Frontier graph generator for publications and reports.
* **[inspect_error_pair.py](inspect_error_pair.py):** Visual inspection tool for specific error pairs (True vs. Predicted).
* **[analyze_misclassifications.py](analyze_misclassifications.py):** Confusion matrix generator and misclassified sample analyzer.

---

## 🚀 Quick Start Guide

### 1. Train Model (e.g., $\alpha = 0.75$)
```bash
python train.py
```

### 2. Quantize & Log Experiment
```bash
python quantize_and_export.py
```

### 3. Generate Pareto Frontier Chart
```bash
python plot_pareto_front.py
```

### 4. Inspect Misclassification Pairs (e.g., True=0, Predicted=6)
```bash
python inspect_error_pair.py --true 0 --pred 6
```
