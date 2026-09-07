# 🗃️ Registro de Cambios y Auditoría del Dataset (Data Changelog)

Este documento registra cronológicamente las modificaciones, limpiezas, filtrados y rebalanceos aplicados al conjunto de datos de dígitos (`dataset_mobilenet`).

---

## 📌 Historial de Versiones del Dataset

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
- **Experimentos Asociados:** `EXP_004` en adelante.
- **Estado:** Dataset Activo para Entrenamiento.

---

### 🔹 Versión 1.0 (Original / Base Auditado) - [Fecha: 2026-09-07]
- **Ubicación:** `dataset_mobilenet/`
- **Volumen Total:** 23.704 imágenes (Train: 20.058, Val: 3.646).
- **Desglose por Clase (Train Set):**
  - Dígito 0: **8.684 imágenes (43.3%)**
  - Dígito 1: 2.088 imágenes (10.4%)
  - Dígito 2: 1.508 imágenes (7.5%)
  - Dígito 3: 1.443 imágenes (7.2%)
  - Dígito 4: 1.162 imágenes (5.8%)
  - Dígito 5: 1.119 imágenes (5.6%)
  - Dígito 6: 1.079 imágenes (5.4%)
  - Dígito 8: 1.039 imágenes (5.2%)
  - Dígito 7: 975 imágenes (4.9%)
  - Dígito 9: **961 imágenes (4.8%)**
- **Experimentos Asociados:** `EXP_001`, `EXP_002`, `EXP_003`.
- **Estado:** Archivo Histórico.
