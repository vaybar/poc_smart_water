# 🗃️ Registro de Cambios y Auditoría del Dataset (Data Changelog)

Este documento registra cronológicamente las modificaciones, limpiezas, filtrados y rebalanceos aplicados al conjunto de datos de dígitos (`dataset_mobilenet`).

---

## 📌 Historial de Versiones del Dataset

### 🔹 Versión 1.2 (Diagnóstico de Errores & Label Noise Auditado) - [Fecha: 2026-09-08]
- **Ubicación Dataset Activo:** `dataset_mobilenet/`
- **Diagnóstico Ejecutado:** `analyze_misclassifications.py` sobre `EXP_004`.
- **Hallazgo Clave:**
  - Se analizaron los 460 errores (12.65% del test set).
  - La confianza máxima de la predicción errónea fue de apenas **23.0%**, descartando la presencia de *Label Noise* severo (etiquetas cruzadas en el dataset original).
  - **Causa Principal:** La degradación de exactitud proviene de la baja capacidad de filtros ($\alpha = 0.25$) para resolver fronteras entre dígitos similares (`0 -> 9` con 64 fallos, `1 -> 7` con 36 fallos).
- **Experimentos Asociados:** `EXP_005` (donde se aislará la variable incrementando $\alpha = 0.35$).
- **Estado:** Dataset Limpio v1.1 Confirmado.

---

### 🔹 Versión 1.1 (Limpio & Aislado) - [Fecha: 2026-09-07]
- **Ubicación Dataset Activo:** `dataset_mobilenet/`
- **Ubicación Imágenes Descartadas:** `dataset_mobilenet/discarded_images/`
- **Volumen Retenido:** **23.644 imágenes** (Train: 20.007, Val: 3.637).
- **Acción Realizada:** Filtrado automático ejecutando `clean_dataset.py` (Criterio: Desviación estándar de intensidad de píxeles $std < 10.0$ o archivos corruptos).
- **Desglose de Imágenes Descartadas y Aisladas (Total: 60):**
  - Dígito 0: 34 imágenes movidas a `discarded_images/`
  - Dígito 1: 6 imágenes movidas
  - Dígito 7: 5 imágenes movidas
  - Dígito 9: 6 imágenes movidas
  - Dígito 2: 3 imágenes movidas
  - Dígito 6: 2 imágenes movidas
  - Dígito 8: 2 imágenes movidas
  - Dígito 3: 1 imagen movida
  - Dígito 5: 1 imagen movida
- **Experimentos Asociados:** `EXP_004`.
- **Estado:** Histórico v1.1.

---

### 🔹 Versión 1.0 (Original / Base Auditado) - [Fecha: 2026-09-07]
- **Ubicación:** `dataset_mobilenet/`
- **Volumen Total:** 23.704 imágenes (Train: 20.058, Val: 3.646).
- **Experimentos Asociados:** `EXP_001`, `EXP_002`, `EXP_003`.
- **Estado:** Archivo Histórico.
