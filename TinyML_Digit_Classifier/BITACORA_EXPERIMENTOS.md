# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra las hipótesis de desarrollo, cambios de hiperparámetros, resultados de cada experimento y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 🎯 Diario de Iteraciones, Hipótesis y Conclusiones

### 🔹 Fase 1: Baseline Inicial (`EXP_001` - `EXP_003`)
- **Hipótesis:** Comprobar la viabilidad de la arquitectura Micro-MobileNet ($\alpha=0.25$) para clasificar dígitos en menos de 256 KB Flash y 40 KB RAM Arena.
- **Resultado:** **87.25% Acc INT8**, 12.85 KB Flash, 13.86 KB RAM Arena.

---

### 🔹 Fase 2: Limpieza v1.1 y Modificación de Capacidad (`EXP_004` - `EXP_006`)
- **Hipótesis:** Filtrar 60 imágenes ruidosas/planas ($std < 10.0$) e incrementar $\alpha = 0.35$.
- **Resultado:** **87.87% Acc INT8**.
- **Diagnóstico:** El análisis reveló que la entrada $32 \times 32$ aplastaba al 50% la relación de aspecto nativa de $64 \times 32$, cerrando la curva del dígito '6' y confundiéndolo con el '0'.

---

### 🔹 Fase 3: Resolución Nativa $64 \times 32 \times 1$ + $\alpha=0.35$ (`EXP_007` - ÉXITO 🎉)
- **Hipótesis:** Cambiar la dimensión de entrada a la relación de aspecto nativa de $64 \times 32 \times 1$ (con $\alpha = 0.35$) evitará la distorsión del reescalado, restaurará la geometría curva del '6' y romperá la barrera del 90.0% de exactitud INT8.
- **Resultado:** **91.94% Acc INT8 (+4.07% de mejora)**.
  - **F1-Score del '6':** Saltó de `0.823` a **`0.910` (91.0%)**.
- **Diagnóstico de Falla Remanente:** Se identificó que 47 muestras del dígito '1' con serifa fueron clasificadas erróneamente como '7', degradando la precisión del '7' a 0.734.
- **Reporte:** [📄 Ver Reporte EXP_007](experiments/EXP_007/report.md)

---

### 🔹 Fase 4: Triple Enfoque de Rebalanceo, Augmentation Lateral y Loss Weighting (`EXP_008` - ÉXITO HISTÓRICO 🚀)
- **Hipótesis:** Rebalancear `train/0` a 2,000 imágenes, aplicar *Data Augmentation* de cizallamiento e inclinación lateral (*horizontal shear* $\pm 12^\circ$) para '1' y '7', e incrementar el peso de pérdida en '1' ($1.3\times$) y '7' ($1.4\times$) eliminará las confusiones por inclinación de cámara y serifa.
- **Resultado:** **92.45% Acc INT8**.
  - **Confusión '1' $\rightarrow$ '7':** Reducida de 47 muestras a **0 muestras**.
  - **F1-Score del '1':** Subió de **87.2% a 98.3%**.
  - **Precisión del '7':** Subió de **73.4% a 94.6%**.
  - **Consumo Hardware:** **14.24 KB Flash** y **14.27 KB RAM Arena**.
- **Reporte:** [📄 Ver Reporte EXP_008](experiments/EXP_008/report.md)

---

### 🔹 Fase 5: Escalado de Capacidad Convolucional a $\alpha=0.50$ (`EXP_009` - ÉXITO CONFIRMADO 🏆)
- **Hipótesis:** Dado que $\alpha=0.35$ consumía apenas 14.24 KB Flash (5.5% del límite), duplicar la capacidad convolucional a **$\alpha = 0.50$** otorgará la capacidad de abstracción de características para discriminar sutilezas de curvatura entre '7' vs '3' y '9' vs '8' en el dataset limpio y rebalanceado.
- **Resultado:** **94.45% Acc INT8 (+2.00% de mejora sobre EXP_008)**.
  - **Pérdida por Cuantización:** Prácticamente nula (**-0.08%**, INT8 supera al Float32 de 94.36%).
  - **Consumo Hardware:** **16.85 KB Flash** (solo 6.5% del límite de 256 KB) y **15.06 KB RAM Arena** (37.6% del límite de 40 KB).
  - **Latencia:** **0.042 ms / imagen** (más de 23,000 FPS en simulación Python).
- **Reporte:** [📄 Ver Reporte EXP_009](experiments/EXP_009/report.md)

---

## 📊 Tabla Comparativa de Experimentos

| Exp ID | Fecha | Alpha ($\alpha$) | Entrada | Épocas | Batch | Float32 Acc | INT8 Acc | Pérdida INT8 | Macro F1 | Weighted F1 | Latencia (ms) | Flash (KB) | RAM Arena (KB) | Reporte Detallado | Notas / Cambios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_001` | 2026-09-04 22:12 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Baseline Micro-MobileNet alpha=0.25 |
| `EXP_002` | 2026-09-04 22:21 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Réplica de verificación inicial |
| `EXP_003` | 2026-09-07 14:43 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | 0.843 | 0.875 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_003/report.md) | Evaluación retroactiva de test set |
| `EXP_004` | 2026-09-07 21:29 | 0.25 | `32x32x1` | 30 | 32 | 86.61% | **86.88%** | -0.27% | 0.835 | 0.872 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_004/report.md) | Limpieza de imágenes planas (std < 10) |
| `EXP_005` | 2026-09-08 18:59 | 0.25 | `32x32x1` | 30 | 32 | 87.13% | **87.79%** | -0.66% | 0.851 | 0.881 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_005/report.md) | Optimización de filtros de limpieza |
| `EXP_006` | 2026-09-08 20:20 | 0.35 | `32x32x1` | 30 | 32 | 87.79% | **87.87%** | -0.08% | 0.854 | 0.882 | 0.024 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_006/report.md) | Escalado a alpha=0.35 en 32x32 |
| `EXP_007` | 2026-09-08 21:26 | 0.35 | `64x32x1` | 30 | 32 | 91.89% | **91.94%** | -0.05% | 0.899 | 0.921 | 0.041 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_007/report.md) | Entrada nativa 64x32x1 con alpha=0.35 |
| `EXP_008` | 2026-09-10 22:43 | 0.35 | `64x32x1` | 30 | 32 | 92.6% | **92.45%** | 0.15% | 0.923 | 0.924 | 0.043 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_008/report.md) | Rebalanceo + Augmentation Shear 1/7 + Class Weighting |
| `EXP_009` | 2026-09-12 18:34 | 0.50 | `64x32x1` | 30 | 32 | 94.36% | **94.45%** | -0.08% | 0.926 | 0.945 | 0.042 | **16.85 KB** | **15.06 KB** | [📄 Ver Reporte](experiments/EXP_009/report.md) | Escalado de capacidad a alpha=0.50 con dataset limpio y rebalanceado |

---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
