# Reporte de Experimento: EXP_006

- **Fecha:** 2026-09-23 12:58
- **Arquitectura:** Micro-Corner-Regressor ($lpha=0.75$)
- **Resolución Thumbnail:** 128x128x1
- **Hiperparámetros:** 80 épocas, Batch Size 32

## 1. Hipótesis
Predecir centroide y vectores directores (u, v) garantizara paralelismo absoluto, llevando el IoU > 60% y MAE < 4.5px.

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `0.00171`
- **Error Medio de Esquina (MAE):** `12.11 px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `20.44%`
- **Error Angular de Orientación:** `30.20°`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `54.63 KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `24.50 KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `537.024 ms`

## 4. Notas y Conclusiones
- **Notas:** EXP_006: Calculo con centroide y rectas hacia arriba y costado (abarcando el rectangulo)
- **Conclusiones:** Corner MAE: 12.1px, IoU: 20.4%, Angle Error: 30.2°. Fits in 54.6KB Flash.
