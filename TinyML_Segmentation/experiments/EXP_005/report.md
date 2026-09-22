# Reporte de Experimento: EXP_005

- **Fecha:** 2026-09-22 22:30
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 80 épocas, Batch Size 32

## 1. Hipótesis
Alinear canonicamente el borde largo (TL->TR) eliminara la contradiccion de 90 grados y permitira convergencia con MAE < 5px y IoU > 60%

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.00120`
- **Error Medio de Esquina (MAE):** `8.82 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `36.85%`
- **Error Angular de Orientación:** `23.55°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `54.76 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `538.2719999999999 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_005: MicroCornerRegressor V2 con dataset canonico alineado por eje mayor
- **Conclusiones:** Corner MAE: 8.8px, IoU: 36.9%, Angle Error: 23.6°. Fits in 54.8KB Flash.
