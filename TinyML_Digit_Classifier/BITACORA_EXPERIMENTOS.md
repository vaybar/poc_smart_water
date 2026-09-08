# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra los experimentos, hipótesis de desarrollo, cambios de hiperparámetros y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 🎯 Diario de Iteraciones e Hipótesis

### 🔹 Fase 1: Baseline Inicial (`EXP_001` - `EXP_003`)
- **Hipótesis:** Comprobar la viabilidad de la arquitectura Micro-MobileNet ($\alpha=0.25$) para clasificar dígitos en menos de 256 KB Flash y 40 KB RAM Arena.
- **Resultado:** **87.25% Acc INT8**, 12.85 KB Flash, 13.86 KB RAM Arena.

---

### 🔹 Fase 2: Limpieza de Dataset v1.1 (`EXP_004`)
- **Hipótesis:** Filtrar 60 imágenes ruidosas/planas ($std < 10.0$) mejorará la estabilidad del modelo manteniendo $\alpha=0.25$ y 30 épocas.
- **Resultado:** **86.88% Acc INT8**.
- **Diagnóstico Ejecutado (`analyze_misclassifications.py`):**
  - Se descartó *Label Noise* severo (confianza máxima en fallos de solo 23%).
  - Confusiones principales: `0 -> 9` (64 fallos) y `1 -> 7` (36 fallos).
  - La restricción de rendimiento no es ruido de etiquetas, sino **subcapacidad de expresión del multiplicador $\alpha=0.25$**.

---

### 🔹 Fase 3: Escalado de Capacidad del Modelo (`EXP_005` - Próxima Ejecución)
- **Hipótesis:** Aumentar el multiplicador de ancho a **$\alpha = 0.35$** (manteniendo 30 épocas y el dataset limpio v1.1) dará los canales necesarios para resolver las fronteras de decisión entre `0-9` y `1-7`, superando el **90.0% Acc** sin exceder los límites de la ESP32-S3 (~25 KB Flash vs 256 KB límite).

---

## 📊 Tabla Comparativa de Experimentos

| Exp ID | Fecha | Alpha ($\alpha$) | Entrada | Épocas | Batch | Float32 Acc | INT8 Acc | Pérdida INT8 | Macro F1 | Weighted F1 | Latencia (ms) | Flash (KB) | RAM Arena (KB) | Reporte Detallado | Notas / Cambios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_001` | 2026-09-04 22:12 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_002` | 2026-09-04 22:21 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | - | - | - | **12.85 KB** | **13.86 KB** | - | Entrenamiento con alpha=0.25 |
| `EXP_003` | 2026-09-07 14:43 | 0.25 | `32x32x1` | 30 | 32 | 87.69% | **87.25%** | 0.44% | 0.843 | 0.875 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_003/report.md) | Evaluación de prueba previa recuperada (retroactiva) |
| `EXP_004` | 2026-09-07 21:29 | 0.25 | `32x32x1` | 30 | 32 | 86.61% | **86.88%** | -0.27% | 0.835 | 0.872 | 0.023 | **12.85 KB** | **13.86 KB** | [📄 Ver Reporte](experiments/EXP_004/report.md) | Entrenamiento con alpha=0.25 |

---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
