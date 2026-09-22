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
| **EXP_003** | 2026-09-22 12:52 | 0.75 | 128x128x1 | 0.0137 | 10.3 px | 26.9% | 15.5° | 38.9 KB | 24.5 KB | EXP_003: Aumentacion afin geometrica (2000 muestras/epoca) + cabeza 2D |

---

## Resumen de Decisiones de Arquitectura
1. **Regresión Directa de 4 Esquinas:** Supera a YOLOv8n (3.3 MB vs <50 KB).
2. **Bilinear Quadrilateral Mapping:** Permite corregir ángulos de inclinación arbitrarios sin OpenCV.
3. **Corte Determinista por Ranuras (*Slots*):** Los tambores mecánicos tienen espaciado físico idéntico, eliminando la vulnerabilidad a barro y suciedad en el dial.
