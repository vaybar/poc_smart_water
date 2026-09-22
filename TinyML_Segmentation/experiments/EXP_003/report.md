# Reporte de Experimento: EXP_003

- **Fecha:** 2026-09-22 12:52
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 19 épocas, Batch Size 16

## 1. Hipótesis
La aumentacion afin de esquinas y traslacion eliminara el overfitting del EXP_002 y mejorara IoU > 70%

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.01372`
- **Error Medio de Esquina (MAE):** `10.30 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `26.86%`
- **Error Angular de Orientación:** `15.52°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `38.90 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `382.368 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_003: Aumentacion afin geometrica (2000 muestras/epoca) + cabeza 2D
- **Conclusiones:** Corner MAE: 10.3px, IoU: 26.9%, Angle Error: 15.5°. Fits in 38.9KB Flash.
