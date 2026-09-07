# 🔬 Reporte Detallado de Experimento - `EXP_004`

- **Fecha / Hora:** 2026-09-07 21:29
- **Notas / Cambios:** Entrenamiento con alpha=0.25

---

## 🎯 Hipótesis y Contexto del Experimento
_No se especificó hipótesis formal para este experimento._

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\alpha$)** | `0.25` |
| **Dimensión de Entrada** | `32x32x1` |
| **Épocas de Entrenamiento** | `30` |
| **Tamaño de Batch** | `32` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `86.61%` | N/A |
| **Exactitud INT8** | **`86.88%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.27%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0228 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.8345` | Max |
| **Weighted F1-Score** | `0.8716` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`12.85 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`13.86 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.977 | 0.875 | **0.923** | 1556.0 |
| Dígito '1' | 0.858 | 0.906 | **0.881** | 393.0 |
| Dígito '2' | 0.898 | 0.875 | **0.886** | 272.0 |
| Dígito '3' | 0.768 | 0.905 | **0.831** | 242.0 |
| Dígito '4' | 0.880 | 0.897 | **0.888** | 213.0 |
| Dígito '5' | 0.727 | 0.882 | **0.797** | 220.0 |
| Dígito '6' | 0.668 | 0.816 | **0.735** | 185.0 |
| Dígito '7' | 0.860 | 0.855 | **0.857** | 179.0 |
| Dígito '8' | 0.751 | 0.770 | **0.760** | 204.0 |
| Dígito '9' | 0.768 | 0.803 | **0.785** | 173.0 |
| **macro avg** | 0.816 | 0.858 | **0.834** | 3637.0 |
| **weighted avg** | 0.879 | 0.869 | **0.872** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1362   41    9   20   11   18   50    6   14   25
 Digit '1'  |    8  356    1   10    4    2    0   11    1    0
 Digit '2'  |    2    1  238    6   10    5    0    8    2    0
 Digit '3'  |    1    2    3  219    1   11    1    0    4    0
 Digit '4'  |    4    2    3    9  191    2    0    0    1    1
 Digit '5'  |    2    0    3    6    0  194    6    0    6    3
 Digit '6'  |    5    0    0    0    0    8  151    0   18    3
 Digit '7'  |    3   13    8    1    0    0    0  153    0    1
 Digit '8'  |    2    0    0    6    0   18   12    0  157    9
 Digit '9'  |    5    0    0    8    0    9    6    0    6  139
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
