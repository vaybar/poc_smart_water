# 🔬 Reporte Detallado de Experimento - `EXP_005`

- **Fecha / Hora:** 2026-09-08 18:59
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
| **Exactitud Float32** | `87.13%` | N/A |
| **Exactitud INT8** | **`87.79%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.66%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0233 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.8505` | Max |
| **Weighted F1-Score** | `0.8813` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`12.85 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`13.86 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.981 | 0.877 | **0.926** | 1556.0 |
| Dígito '1' | 0.844 | 0.865 | **0.854** | 393.0 |
| Dígito '2' | 0.841 | 0.915 | **0.877** | 272.0 |
| Dígito '3' | 0.844 | 0.897 | **0.870** | 242.0 |
| Dígito '4' | 0.968 | 0.854 | **0.908** | 213.0 |
| Dígito '5' | 0.836 | 0.950 | **0.889** | 220.0 |
| Dígito '6' | 0.637 | 0.930 | **0.756** | 185.0 |
| Dígito '7' | 0.680 | 0.866 | **0.762** | 179.0 |
| Dígito '8' | 0.903 | 0.819 | **0.859** | 204.0 |
| Dígito '9' | 0.812 | 0.798 | **0.805** | 173.0 |
| **macro avg** | 0.835 | 0.877 | **0.850** | 3637.0 |
| **weighted avg** | 0.892 | 0.878 | **0.881** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1364   36   19   16    0    8   64   28    5   16
 Digit '1'  |   10  340    3    4    3    0    0   33    0    0
 Digit '2'  |    1    2  249    4    1    6    0    8    1    0
 Digit '3'  |    2    5    1  217    1   10    1    0    1    4
 Digit '4'  |    0    7    9    8  182    1    0    2    4    0
 Digit '5'  |    1    1    3    3    1  209    2    0    0    0
 Digit '6'  |    2    0    0    0    0    7  172    0    2    2
 Digit '7'  |    6    8   10    0    0    0    0  155    0    0
 Digit '8'  |    0    2    2    1    0    4   18    0  167   10
 Digit '9'  |    4    2    0    4    0    5   13    2    5  138
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
