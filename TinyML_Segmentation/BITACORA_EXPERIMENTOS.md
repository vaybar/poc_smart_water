# Bitácora de Experimentos: TinyML_Segmentation (ESP32 Clásico)

Este documento registra sistemáticamente cada experimento realizado para la localización de la ventanilla del medidor de agua y su posterior rectificación geométrica en recortes de **64x32x1 en escala de grises**.

### Presupuesto de Hardware (ESP32 Clásico LX6):
- **Flash ROM Máximo:** `< 80 KB`
- **Tensor Arena Máximo:** `< 35 KB`
- **Formato de Salida Obligatorio:** $N$ recortes de `64x32x1` (uint8) para `TinyML_Digit_Classifier`.

---

## Tabla Histórica de Experimentos

| Exp ID | Fecha | Alpha | Input Shape | Val Loss | Corner MAE (px) | Polygon IoU | Error Angular (°) | Flash INT8 (KB) | Tensor Arena (KB) | Notas / Hallazgos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP_001** | 2026-09-18 18:51 | 0.5 | 128x128x1 | 0.0049 | 13.5 px | 16.7% | 13.2° | 3.9 KB | 24.5 KB | Micro-Corner-Regressor training |
| **EXP_002** | 2026-09-18 19:36 | 0.75 | 128x128x1 | 0.0146 | 10.5 px | 27.2% | 18.2° | 38.9 KB | 24.5 KB | EXP_002: Cabeza espacial 2D sin GAP + loss MSE + Data Augmentation |
| **EXP_003** | 2026-09-22 18:44 | 0.75 | 128x128x1 | 0.0276 | 14.9 px | 15.8% | 28.7° | 38.9 KB | 24.5 KB | EXP_004: Modelo EXP_002 con dataset masivo (imagenes reales convertidas) |
| **EXP_004** | 2026-09-22 21:36 | 0.75 | 128x128x1 | 0.0019 | 13.1 px | 28.1% | 35.1° | 54.8 KB | 24.5 KB | EXP_004: MicroCornerRegressor V2 con resolucion 8x8 y skip connection multiescala |
| **EXP_005** | 2026-09-22 22:30 | 0.75 | 128x128x1 | 0.0012 | 8.8 px | 36.9% | 23.6° | 54.8 KB | 24.5 KB | EXP_005: MicroCornerRegressor V2 con dataset canonico alineado por eje mayor |
| **EXP_006** | 2026-09-23 12:58 | 0.75 | 128x128x1 | 0.0017 | 12.1 px | 20.4% | 30.2° | 54.6 KB | 24.5 KB | EXP_006: Calculo con centroide y rectas hacia arriba y costado (abarcando el rectangulo) |
| **EXP_007** | 2026-09-23 17:04 | 0.75 | 128x128x1 | 0.2855 | 8.7 px | 41.6% | 23.8° | 54.8 KB | 24.5 KB | EXP_007 - propuesta Claude Opus: Augmentacion Geometrica - Regression con Sigmoid |

---

## Resumen de Decisiones de Arquitectura
1. **Regresión Directa de 4 Esquinas:** Supera a YOLOv8n (3.3 MB vs <50 KB).
2. **Bilinear Quadrilateral Mapping:** Permite corregir ángulos de inclinación arbitrarios sin OpenCV.
3. **Corte Determinista por Ranuras (*Slots*):** Los tambores mecánicos tienen espaciado físico idéntico, eliminando la vulnerabilidad a barro y suciedad en el dial.
