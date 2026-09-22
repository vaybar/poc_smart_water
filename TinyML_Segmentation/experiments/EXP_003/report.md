# Reporte de Experimento: EXP_003

- **Fecha:** 2026-09-22 18:44
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 20 épocas, Batch Size 32

## 1. Hipótesis
Entrenar con miles de muestras reales eliminara el overfitting del EXP_002 sin perder precision espacial

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.02759`
- **Error Medio de Esquina (MAE):** `14.92 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `15.82%`
- **Error Angular de Orientación:** `28.66°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `38.90 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `382.368 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_004: Modelo EXP_002 con dataset masivo (imagenes reales convertidas)
- **Conclusiones:** Corner MAE: 14.9px, IoU: 15.8%, Angle Error: 28.7°. Fits in 38.9KB Flash.
