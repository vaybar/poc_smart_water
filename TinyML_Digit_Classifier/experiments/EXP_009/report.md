# 🔬 Reporte Detallado de Experimento - `EXP_009`

- **Fecha / Hora:** 2026-09-12 18:34
- **Notas / Cambios:** EXP_009 (Re-ejecución determinista): alpha=0.50 con dataset limpio y rebalanceado

---

## 🎯 Hipótesis y Contexto del Experimento
Escalado de capacidad convolucional a alpha=0.5 con entrada 64x32x1

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\alpha$)** | `0.5` |
| **Dimensión de Entrada** | `64x32x1` |
| **Épocas de Entrenamiento** | `30` |
| **Tamaño de Batch** | `32` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `94.36%` | N/A |
| **Exactitud INT8** | **`94.45%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.08%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0418 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.9255` | Max |
| **Weighted F1-Score** | `0.9453` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`16.85 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`15.06 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.993 | 0.952 | **0.972** | 1556.0 |
| Dígito '1' | 0.941 | 0.926 | **0.933** | 393.0 |
| Dígito '2' | 0.966 | 0.952 | **0.959** | 272.0 |
| Dígito '3' | 0.946 | 0.938 | **0.942** | 242.0 |
| Dígito '4' | 0.945 | 0.972 | **0.958** | 213.0 |
| Dígito '5' | 0.906 | 0.968 | **0.936** | 220.0 |
| Dígito '6' | 0.865 | 0.935 | **0.899** | 185.0 |
| Dígito '7' | 0.833 | 0.922 | **0.875** | 179.0 |
| Dígito '8' | 0.935 | 0.912 | **0.923** | 204.0 |
| Dígito '9' | 0.803 | 0.919 | **0.857** | 173.0 |
| **macro avg** | 0.913 | 0.940 | **0.926** | 3637.0 |
| **weighted avg** | 0.947 | 0.944 | **0.945** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1482   11    1    2    8    3   11    9    5   24
 Digit '1'  |    4  364    2    3    1    0    0   17    2    0
 Digit '2'  |    0    0  259    3    0    7    0    2    0    1
 Digit '3'  |    0    4    0  227    1    2    1    3    0    4
 Digit '4'  |    0    0    3    0  207    0    0    2    0    1
 Digit '5'  |    1    0    0    1    0  213    5    0    0    0
 Digit '6'  |    3    0    0    0    0    2  173    0    4    3
 Digit '7'  |    0    7    2    2    1    1    0  165    0    1
 Digit '8'  |    1    1    1    0    1    0    9    0  186    5
 Digit '9'  |    2    0    0    2    0    7    1    0    2  159
```

---

## 💡 Conclusiones y Aprendizajes

_No se registraron conclusiones explícitas para este experimento._
