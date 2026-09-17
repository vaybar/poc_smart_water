# Bitácora de Experimentos: TinyML_Segmentation (ESP32 Clásico)

Este documento registra sistemáticamente cada experimento realizado para la localización de la ventanilla del medidor de agua y su posterior rectificación geométrica en recortes de **64x32x1 en escala de grises**.

### Presupuesto de Hardware (ESP32 Clásico LX6):
- **Flash ROM Máximo:** `< 80 KB`
- **Tensor Arena Máximo:** `< 35 KB`
- **Formato de Salida Obligatorio:** $N$ recortes de `64x32x1` (uint8) para `TinyML_Digit_Classifier`.

---

## Tabla Histórica de Experimentos

| Exp ID | Fecha | Alpha | Input Shape | Val Loss | Corner MAE (px) | Polygon IoU | Flash INT8 (KB) | Tensor Arena (KB) | Notas / Hallazgos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Pendiente* | - | 0.50 | 128x128x1 | - | - | - | ~38 KB | ~24 KB | Línea base: Micro-Corner-Regressor con ReLU6 y Bilinear Warp |

---

## Metodología de Evaluación y Métricas

1. **Corner MAE (Error Medio Absoluto en Píxeles):**
   Mide la distancia euclidiana media en píxeles (sobre el lienzo de $128 \times 128$) entre las 4 esquinas reales anotadas y las 4 esquinas estimadas por la red.
2. **Polygon IoU (Intersección sobre Unión del Cuadrilátero):**
   Evalúa el solapamiento de la ventana rotada. Un $\text{IoU} > 0.85$ garantiza una rectificación visualmente limpia de los dígitos.
3. **Invarianza a Rotación:**
   Se valida que la tira rectificada y los recortes mantengan a los números orientados verticalmente incluso si el medidor se encuentra a $30^\circ$, $45^\circ$ o $90^\circ$.
4. **Resistencia a Barro y Suciedad:**
   Al segmentar por división fija de ranuras (*slots*) sobre la ventanilla rectificada, se evita la pérdida de cortes que sufría la proyección vertical con manchas de tierra.
