# Reporte de Experimento: EXP_001

- **Fecha:** 2026-09-18 18:51
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.5$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 26 épocas, Batch Size 32

## 1. Hipótesis
Micro-Pose model locates rotated dial on MCU

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.00492`
- **Error Medio de Esquina (MAE):** `13.53 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `16.65%`
- **Error Angular de Orientación:** `13.19°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `3.89 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `38.208 ms`

## 4. Notas y Conclusiones
- **Notas:** Micro-Corner-Regressor training
- **Conclusiones:** Corner MAE: 13.5px, IoU: 16.7%, Angle Error: 13.2°. Fits in 3.9KB Flash.
