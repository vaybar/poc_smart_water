# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra automáticamente los resultados de cada experimento, cambios de hiperparámetros y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 📊 Tabla Comparativa de Experimentos

| Exp ID | Fecha | Alpha ($\alpha$) | Entrada | Épocas | Batch | Float32 Acc | INT8 Acc | Pérdida INT8 | Macro F1 | Weighted F1 | Latencia (ms) | Flash (KB) | RAM Arena (KB) | Reporte Detallado | Notas / Cambios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_001` | 2026-09-04 22:12 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_002` | 2026-09-04 22:21 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_003` | 2026-09-07 14:43 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | 0.843 | 0.875 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_003/report.md) | Evaluación de prueba previa recuperada (retroactiva) |
| `EXP_004` | 2026-09-07 21:29 | 0.25 | `32x32x1` | 30 | 32 | 86.61% | **86.88%** | -0.27% | 0.835 | 0.872 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_004/report.md) | Entrenamiento con alpha=0.25 |
| `EXP_005` | 2026-09-08 18:59 | 0.25 | `32x32x1` | 30 | 32 | 87.13% | **87.79%** | -0.66% | 0.851 | 0.881 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_005/report.md) | Entrenamiento con alpha=0.25 |
| `EXP_006` | 2026-09-08 20:20 | 0.35 | `32x32x1` | 30 | 32 | 87.79% | **87.87%** | -0.08% | 0.854 | 0.882 | 0.024 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_006/report.md) | Entrenamiento con alpha=0.35 |
| `EXP_007` | 2026-09-08 21:26 | 0.35 | `64x32x1` | 30 | 32 | 91.89% | **91.94%** | -0.05% | 0.899 | 0.921 | 0.041 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_007/report.md) | Entrenamiento con alpha=0.35 |
| `EXP_008` | 2026-09-10 22:43 | 0.35 | `64x32x1` | 30 | 32 | 92.6% | **92.45%** | 0.15% | 0.923 | 0.924 | 0.043 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_008/report.md) | Entrenamiento con alpha=0.35 |

---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
