# Reporte de Experimento: EXP_008

- **Fecha:** 2026-09-23 19:02
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 86 épocas, Batch Size 16

## 1. Hipótesis
CoordConv preserva info espacial, centro+offsets acopla esquinas, curriculum loss evita dead gradients -> IoU > 55% y MAE < 6px

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.21792`
- **Error Medio de Esquina (MAE):** `7.87 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `35.77%`
- **Error Angular de Orientación:** `21.01°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `39.15 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `384.8448 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_008: Implement V5 head (CoordConv + Center-Offsets), CutOut augmentation and Curriculum Loss
- **Conclusiones:** Corner MAE: 7.9px, IoU: 35.8%, Angle Error: 21.0°. Fits in 39.1KB Flash.
