# 🔬 Reporte Detallado de Experimento - `EXP_008`

- **Fecha / Hora:** 2026-09-10 22:43
- **Notas / Cambios:** Entrenamiento con alpha=0.35

---

## 🎯 Hipótesis y Contexto del Experimento
_No se especificó hipótesis formal para este experimento._

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\alpha$)** | `0.35` |
| **Dimensión de Entrada** | `64x32x1` |
| **Épocas de Entrenamiento** | `30` |
| **Tamaño de Batch** | `32` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `92.6%` | N/A |
| **Exactitud INT8** | **`92.45%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `0.15%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0431 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.9234` | Max |
| **Weighted F1-Score** | `0.9243` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`14.24 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`14.27 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.919 | 0.977 | **0.947** | 175.0 |
| Dígito '1' | 0.979 | 0.987 | **0.983** | 234.0 |
| Dígito '2' | 0.912 | 0.904 | **0.908** | 219.0 |
| Dígito '3' | 0.908 | 0.908 | **0.908** | 207.0 |
| Dígito '4' | 0.950 | 0.959 | **0.954** | 217.0 |
| Dígito '5' | 0.933 | 0.933 | **0.933** | 179.0 |
| Dígito '6' | 0.884 | 0.944 | **0.913** | 178.0 |
| Dígito '7' | 0.946 | 0.854 | **0.897** | 205.0 |
| Dígito '8' | 0.865 | 0.901 | **0.883** | 192.0 |
| Dígito '9' | 0.939 | 0.876 | **0.907** | 194.0 |
| **macro avg** | 0.924 | 0.924 | **0.923** | 2000.0 |
| **weighted avg** | 0.925 | 0.924 | **0.924** | 2000.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  |  171    0    0    0    0    0    3    0    1    0
 Digit '1'  |    0  231    0    0    0    1    2    0    0    0
 Digit '2'  |    6    0  198    4    0    0    5    0    6    0
 Digit '3'  |    0    0    7  188    0    5    0    5    1    1
 Digit '4'  |    0    0    0    0  208    0    7    1    0    1
 Digit '5'  |    0    0    1    2    1  167    0    2    5    1
 Digit '6'  |    3    0    0    0    0    1  168    0    6    0
 Digit '7'  |    0    4    5   13    2    0    0  175    0    6
 Digit '8'  |    5    0    6    0    0    1    5    0  173    2
 Digit '9'  |    1    1    0    0    8    4    0    2    8  170
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
