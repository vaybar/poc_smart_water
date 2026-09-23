# Reporte de Experimento: EXP_007

- **Fecha:** 2026-09-23 17:04
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 120 épocas, Batch Size 32

## 1. Hipótesis
Wing+IoU loss + aug geometrica + sigmoid directo llevara IoU > 60% y MAE < 5px sin exceder 55KB Flash.

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.28552`
- **Error Medio de Esquina (MAE):** `8.69 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `41.58%`
- **Error Angular de Orientación:** `23.75°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `54.76 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `538.2719999999999 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_007 - propuesta Claude Opus: Augmentacion Geometrica - Regression con Sigmoid
- **Conclusiones:** Corner MAE: 8.7px, IoU: 41.6%, Angle Error: 23.7°. Fits in 54.8KB Flash.
