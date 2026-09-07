# 🔬 Reporte Detallado de Experimento - `EXP_003`

- **Fecha / Hora:** 2026-09-07 14:43
- **Notas / Cambios:** Evaluación de prueba previa recuperada (retroactiva)

---

## 🎯 Hipótesis y Contexto del Experimento
Este experimento establece la **línea base (baseline)** del clasificador de dígitos. Utiliza una arquitectura ultraligera **Micro-MobileNet** basada en convoluciones separables en profundidad (*Depthwise Separable Convolutions*) con multiplicador de ancho $\alpha = 0.25$ e hiperparámetros iniciales (30 épocas, batch size de 32, entrada en escala de grises de $32 \times 32 \times 1$).

**Objetivo de la prueba:** Evaluar la exactitud del modelo cuantizado a **INT8** y su factibilidad de despliegue en microcontroladores **ESP32-S3** bajo un presupuesto estricto de memoria (< 256 KB Flash, < 40 KB RAM).

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
| **Exactitud Float32** | `87.69%` | N/A |
| **Exactitud INT8** | **`87.25%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `0.44%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.023 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.843` | Max |
| **Weighted F1-Score** | `0.875` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`12.85 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`13.86 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.980 | 0.869 | **0.921** | 1561 |
| Dígito '1' | 0.883 | 0.863 | **0.873** | 395 |
| Dígito '2' | 0.836 | 0.919 | **0.876** | 272 |
| Dígito '3' | 0.828 | 0.855 | **0.841** | 242 |
| Dígito '4' | 0.884 | 0.930 | **0.906** | 213 |
| Dígito '5' | 0.740 | 0.918 | **0.819** | 220 |
| Dígito '6' | 0.771 | 0.876 | **0.820** | 185 |
| Dígito '7' | 0.754 | 0.833 | **0.792** | 180 |
| Dígito '8' | 0.880 | 0.828 | **0.854** | 204 |
| Dígito '9' | 0.638 | 0.839 | **0.725** | 174 |
| **macro avg** | 0.819 | 0.873 | **0.843** | 3646 |
| **weighted avg** | 0.885 | 0.872 | **0.875** | 3646 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1356   31   17   20    3   23   31    9    7   64
 Digit '1'  |    9  341    1    2    4    0    0   37    1    0
 Digit '2'  |    2    0  250    6    2    6    0    2    1    3
 Digit '3'  |    5    1    5  207    4   15    2    0    1    2
 Digit '4'  |    0    2    4    5  198    0    0    0    2    2
 Digit '5'  |    1    0    8    5    0  202    1    0    2    1
 Digit '6'  |    2    0    0    0    0   10  162    1    5    5
 Digit '7'  |    2   11   10    1    4    2    0  150    0    0
 Digit '8'  |    1    0    1    2    6    8   11    0  169    6
 Digit '9'  |    6    0    3    2    3    7    3    0    4  146
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
