# 🔬 Reporte Detallado de Experimento - `EXP_007`

- **Fecha / Hora:** 2026-09-08 21:26
- **Notas / Cambios:** Entrenamiento con alpha=0.35

---

## 🎯 Hipótesis y Contexto del Experimento

- **Contexto:** En los experimentos previos con entrada $32 \times 32$ (`EXP_006`), aplastar la imagen distorsionaba la relación de aspecto nativa del dial de $64 \times 32$ (2:1). Esto cerraba la curva del dígito '6', provocando frecuentes confusiones con el '0' y estancando el F1-Score del '6' en `0.823` con una exactitud INT8 de 87.87%.
- **Hipótesis:** Cambiar la dimensión de entrada a la relación de aspecto nativa de **$64 \times 32 \times 1$** (con $\alpha = 0.35$) evitará la distorsión del reescalado, restaurará la geometría curva real del dígito '6' y romperá la barrera del 90.0% de exactitud INT8 sin exceder las restricciones de memoria de la ESP32-S3 (< 256 KB Flash, < 40 KB RAM Arena).

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
| **Exactitud Float32** | `91.89%` | N/A |
| **Exactitud INT8** | **`91.94%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `-0.05%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `0.0409 ms` | **< 5.0 ms** |
| **Macro F1-Score** | `0.899` | Max |
| **Weighted F1-Score** | `0.9206` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`14.24 KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`14.27 KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

| Clase | Precision | Recall | F1-Score | Muestras (Support) |
| :--- | :--- | :--- | :--- | :--- |
| Dígito '0' | 0.987 | 0.929 | **0.957** | 1556.0 |
| Dígito '1' | 0.885 | 0.860 | **0.872** | 393.0 |
| Dígito '2' | 0.881 | 0.949 | **0.913** | 272.0 |
| Dígito '3' | 0.870 | 0.938 | **0.903** | 242.0 |
| Dígito '4' | 0.956 | 0.925 | **0.940** | 213.0 |
| Dígito '5' | 0.873 | 0.941 | **0.906** | 220.0 |
| Dígito '6' | 0.891 | 0.930 | **0.910** | 185.0 |
| Dígito '7' | 0.734 | 0.911 | **0.813** | 179.0 |
| Dígito '8' | 0.936 | 0.936 | **0.936** | 204.0 |
| Dígito '9' | 0.834 | 0.844 | **0.839** | 173.0 |
| **macro avg** | 0.885 | 0.916 | **0.899** | 3637.0 |
| **weighted avg** | 0.924 | 0.919 | **0.921** | 3637.0 |

---

## 🧩 Matriz de Confusión

```
True \ Pred |    0    1    2    3    4    5    6    7    8    9
----------------------------------------------------------------
 Digit '0'  | 1445   21   16   18    7   11   10    2    3   23
 Digit '1'  |    4  338    3    0    1    0    0   47    0    0
 Digit '2'  |    1    2  258    3    1    1    0    5    1    0
 Digit '3'  |    2    4    5  227    0    2    1    1    0    0
 Digit '4'  |    1    5    6    1  197    0    1    2    0    0
 Digit '5'  |    0    0    0    8    0  207    2    2    0    1
 Digit '6'  |    1    0    0    0    0    6  172    0    5    1
 Digit '7'  |    3   11    2    0    0    0    0  163    0    0
 Digit '8'  |    1    0    0    1    0    2    5    0  191    4
 Digit '9'  |    6    1    3    3    0    8    2    0    4  146
```

---

## 💡 Conclusiones y Aprendizajes

1. **Confirmación de Hipótesis:** La entrada nativa $64 \times 32 \times 1$ elevó la exactitud INT8 de **87.87% a 91.94% (+4.07%)**, superando la meta del 90.0%.
2. **Rescate del dígito '6':** El F1-score del dígito '6' aumentó de `0.823` a **`0.910` (91.0%)**, demostrando que la preservación de la relación de aspecto 2:1 era fundamental para distinguir '6' de '0'.
3. **Nuevo cuello de botella identificado ('1' vs '7'):** La matriz de confusión reveló que **47 muestras reales del dígito '1'** fueron clasificadas erróneamente como '7', degradando la precisión de la clase '7' a un mínimo de **0.734 (73.4%)**. Esto motivó el diseño de `EXP_008`.
