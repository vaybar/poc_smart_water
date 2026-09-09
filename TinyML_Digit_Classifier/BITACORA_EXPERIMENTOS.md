# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra los experimentos, hipótesis de desarrollo, cambios de hiperparámetros y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 🎯 Diario de Iteraciones e Hipótesis

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
- **Hipótesis:** Cambiar a la relación de aspecto nativa de $64 \times 32 \times 1$ evitará la distorsión del '6' y romperá la barrera del 90%.
- **Resultado:** **91.94% Acc INT8 (+4.07% de mejora)**.
  - **F1-Score del '6':** Saltó de `0.823` a **`0.910` (91.0%)**.
  - **Hardware ESP32-S3:** **14.24 KB Flash** (5.5% del límite) y **14.27 KB RAM Arena** (35.6% del límite).
- **Reporte:** [📄 Ver Reporte EXP_007](experiments/EXP_007/report.md)

---

### 🔹 Fase 4: Optimización de Frontera '1' vs '7' (`EXP_008` - Próximo Paso)
- **Objetivo:** Resolver las 47 confusiones entre el dígito '1' con serifa y el '7' mediante pesaje de pérdida (*class weighting*) o data augmentation de inclinación lateral, buscando alcanzar **> 93.0% Acc**.

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
| `EXP_007` | 2026-09-08 21:26 | 0.35 | `64x32x1` | 30 | 32 | 91.89% | **91.94%** | -0.05% | 0.899 | 0.921 | 0.041 | **14.24 KB** | **14.27 KB** | [📄 Ver Reporte](experiments/EXP_007/report.md) | Entrada nativa 64x32x1 con alpha=0.35 |

---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
