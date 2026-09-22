# Reporte de Experimento: EXP_004

- **Fecha:** 2026-09-22 21:36
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 48 épocas, Batch Size 32

## 1. Hipótesis
Preservar resolucion 8x8 y fusionar bordes multiescala reducira el MAE < 5px y elevara el IoU > 60%

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.00188`
- **Error Medio de Esquina (MAE):** `13.12 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `28.09%`
- **Error Angular de Orientación:** `35.09°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `54.76 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `538.2719999999999 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_004: MicroCornerRegressor V2 con resolucion 8x8 y skip connection multiescala
- **Conclusiones:** Corner MAE: 13.1px, IoU: 28.1%, Angle Error: 35.1°. Fits in 54.8KB Flash.
