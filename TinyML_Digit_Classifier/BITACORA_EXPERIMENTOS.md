# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra automáticamente los resultados de cada experimento, cambios de hiperparámetros y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 📊 Tabla Comparativa de Experimentos

| Exp ID | Fecha | Alpha ($\alpha$) | Entrada | Épocas | Batch | Float32 Acc | INT8 Acc | Pérdida INT8 | Macro F1 | Weighted F1 | Latencia (ms) | Flash (KB) | RAM Arena (KB) | Reporte Detallado | Notas / Cambios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_001` | 2026-09-04 22:12 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_002` | 2026-09-04 22:21 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_003` | 2026-09-07 14:43 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | 0.843 | 0.875 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_003/report.md) | Evaluación de prueba previa recuperada (retroactiva) |

---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
