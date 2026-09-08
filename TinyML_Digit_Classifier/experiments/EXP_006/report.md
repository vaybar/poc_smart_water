# 🔬 Reporte Detallado de Experimento - `EXP_006`

- **Fecha / Hora:** 2026-09-08 20:20
- **Notas / Cambios:** Entrenamiento con alpha=0.35 (Dataset Limpio v1.1)

---

## 🎯 Hipótesis y Contexto del Experimento
Este experimento evalúa el impacto de **incrementar la capacidad del modelo a $\alpha = 0.35$** (manteniendo 30 épocas y el dataset limpio v1.1) para determinar si la adición de filtros (de 8-16 a 11-22 filtros) resuelve la ambigüedad visual en dígitos complejos.

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\alpha$)** | `0.35` |
| **Dimensión de Entrada** | `32x32x1` |
| **Épocas de Entrenamiento** | `30` |
| **Tamaño de Batch** | `32` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `87.79%` | N/A |
| **Exactitud INT8** | **`87.87%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.08%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0238 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.8536` | Max |
| **Weighted F1-Score** | `0.8824` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`14.24 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`14.27 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.995 | 0.869 | **0.928** | 1556.0 |
| Dígito '1' | 0.900 | 0.799 | **0.846** | 393.0 |
| Dígito '2' | 0.836 | 0.956 | **0.892** | 272.0 |
| Dígito '3' | 0.752 | 0.926 | **0.830** | 242.0 |
| Dígito '4' | 0.935 | 0.873 | **0.903** | 213.0 |
| Dígito '5' | 0.794 | 0.914 | **0.850** | 220.0 |
| Dígito '6' | 0.753 | 0.908 | **0.824** | 185.0 |
| Dígito '7' | 0.615 | 0.894 | **0.729** | 179.0 |
| Dígito '8' | 0.918 | 0.873 | **0.894** | 204.0 |
| Dígito '9' | 0.801 | 0.884 | **0.841** | 173.0 |
| **macro avg** | 0.830 | 0.890 | **0.854** | 3637.0 |
| **weighted avg** | 0.896 | 0.879 | **0.882** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1352   21   13   41    0   18   50   27    2   32
 Digit '1'  |    4  314    3    5    2    0    0   64    1    0
 Digit '2'  |    1    0  260    5    3    1    0    2    0    0
 Digit '3'  |    1    5    2  224    4    4    0    0    1    1
 Digit '4'  |    0    3    8    7  186    1    0    6    1    1
 Digit '5'  |    0    0   10    7    0  201    0    1    1    0
 Digit '6'  |    0    0    2    0    0   10  168    0    4    1
 Digit '7'  |    0    5    8    5    1    0    0  160    0    0
 Digit '8'  |    0    1    5    2    3    9    3    0  178    3
 Digit '9'  |    1    0    0    2    0    9    2    0    6  153
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
