# 🔬 Reporte Detallado de Experimento - `EXP_010`

- **Fecha / Hora:** 2026-09-12 19:41
- **Notas / Cambios:** EXP_010: Escalado alpha=0.75, 35 epocas y class_weights balanceados

---

## 🎯 Hipótesis y Contexto del Experimento
Escalado de capacidad convolucional a alpha=0.75 con entrada 64x32x1

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\alpha$)** | `0.75` |
| **Dimensión de Entrada** | `64x32x1` |
| **Épocas de Entrenamiento** | `35` |
| **Tamaño de Batch** | `32` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `96.7%` | N/A |
| **Exactitud INT8** | **`96.73%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.03%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0498 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.959` | Max |
| **Weighted F1-Score** | `0.9676` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`22.45 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`16.73 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.993 | 0.963 | **0.978** | 1556.0 |
| Dígito '1' | 0.958 | 0.975 | **0.966** | 393.0 |
| Dígito '2' | 0.989 | 0.974 | **0.981** | 272.0 |
| Dígito '3' | 0.944 | 0.971 | **0.957** | 242.0 |
| Dígito '4' | 0.995 | 0.986 | **0.991** | 213.0 |
| Dígito '5' | 0.960 | 0.982 | **0.971** | 220.0 |
| Dígito '6' | 0.913 | 0.968 | **0.940** | 185.0 |
| Dígito '7' | 0.961 | 0.966 | **0.964** | 179.0 |
| Dígito '8' | 0.948 | 0.975 | **0.961** | 204.0 |
| Dígito '9' | 0.842 | 0.925 | **0.882** | 173.0 |
| **macro avg** | 0.950 | 0.968 | **0.959** | 3637.0 |
| **weighted avg** | 0.969 | 0.967 | **0.968** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1498   11    0    4    0    1    8    3    4   27
 Digit '1'  |    3  383    0    3    0    1    0    3    0    0
 Digit '2'  |    0    0  265    2    1    1    0    1    2    0
 Digit '3'  |    0    2    1  235    0    2    1    0    0    1
 Digit '4'  |    1    1    0    0  210    0    0    0    0    1
 Digit '5'  |    0    1    1    1    0  216    1    0    0    0
 Digit '6'  |    2    0    0    0    0    0  179    0    3    1
 Digit '7'  |    1    2    1    1    0    1    0  173    0    0
 Digit '8'  |    0    0    0    1    0    1    3    0  199    0
 Digit '9'  |    3    0    0    2    0    2    4    0    2  160
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
