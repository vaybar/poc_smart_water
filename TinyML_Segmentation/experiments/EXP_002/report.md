# Reporte de Experimento: EXP_002

- **Fecha:** 2026-09-18 19:36
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 34 épocas, Batch Size 16

## 1. Hipótesis
Preservar mapa 4x4 espacial y usar MSE impedirá el colapso a la media y elevará el IoU.

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.01461`
- **Error Medio de Esquina (MAE):** `10.48 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `27.17%`
- **Error Angular de Orientación:** `18.18°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `38.90 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `382.368 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_002: Cabeza espacial 2D sin GAP + loss MSE + Data Augmentation
- **Conclusiones:** Corner MAE: 10.5px, IoU: 27.2%, Angle Error: 18.2°. Fits in 38.9KB Flash.
