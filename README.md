# 🚰 PoC Smart Water: Automated Water Meter Reading with Edge AI & TinyML

This repository contains the Proof of Concept (PoC) for intelligent, automated water meter reading (AMR - *Automatic Meter Reading*) utilizing computer vision and ultra-lightweight deep neural networks designed for low-cost microcontrollers.

---

## 📁 Repository Structure

* **[TinyML_Digit_Classifier/](TinyML_Digit_Classifier/README.md):** Digit classification experimental suite using **Micro-MobileNet** (INT8 quantized, $< 23\text{ KB}$ Flash and $< 17\text{ KB}$ RAM Arena), thoroughly tested from `EXP_001` through `EXP_010` to reach **96.73% accuracy**.
* **`water_meter_pipeline.py` & `mobilenet_pipeline.py`:** Water meter detection and digit segmentation pipelines.

---

## 🔬 TinyML Experimental Suite & Pareto Frontier

For detailed technical documentation, the full experimental comparative table (`EXP_001` - `EXP_010`), dataset paper citation ([Nature Scientific Data 2026](https://www.nature.com/articles/s41597-026-06809-z)), and the **Pareto Frontier chart**, please refer to the dedicated module README:

👉 **[Read Full Experimental Suite Documentation in TinyML_Digit_Classifier/README.md](TinyML_Digit_Classifier/README.md)**
