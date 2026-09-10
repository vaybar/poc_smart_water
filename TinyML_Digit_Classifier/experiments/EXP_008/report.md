# 🔬 Reporte Detallado de Experimento - `EXP_008`

- **Fecha / Hora:** 2026-09-10 22:43
- **Notas / Cambios:** Entrenamiento con alpha=0.35

---

## 🎯 Hipótesis y Contexto del Experimento

- **Contexto:** En `EXP_007` la matriz de confusión evidenció un cuello de botella crítico: **47 imágenes del dígito '1' con serifa** se clasificaron como '7', dejando la precisión del '7' en solo **73.4%**. Además, la clase '0' acumulaba un desbalance extremo de 8,655 imágenes en `train/`, dominando los gradientes.
- **Hipótesis:** Implementar un triple enfoque:
  1. **Rebalanceo de '0':** Limitar `train/0` a 2,000 imágenes representativas (trasladando el exceso a `discarded_images`).
  2. **Data Augmentation Enfocado:** Generar variaciones sintéticas de inclinación y cizallamiento lateral (*horizontal shear* $\pm 12^\circ$) para los dígitos '1' y '7'.
  3. **Class Weighting Focalizado:** Asignar mayor peso de pérdida a las clases '1' ($1.3\times$) y '7' ($1.4\times$).
  Esto eliminará las confusiones por inclinación de cámara/serifa, elevará la precisión del '7' por encima del 90.0% y alcanzará la mayor exactitud INT8 histórica del proyecto (> 92.0%).

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

1. **Eliminación Total de la Confusión '1' $\rightarrow$ '7':** Las confusiones del dígito '1' predicho erróneamente como '7' pasaron de **47 muestras en EXP_007 a 0 muestras en EXP_008**.
2. **Salto de F1-Score en el '1':** El F1-score del dígito '1' se incrementó drásticamente de **`0.872` (87.2%) a `0.983` (98.3%)**, convirtiéndose en la clase más precisa del modelo.
3. **Recuperación de la Precisión del '7':** La precisión del dígito '7' mejoró de **`0.734` (73.4%) a `0.946` (94.6%)**.
4. **Máximo Histórico de Exactitud INT8:** La exactitud cuantizada alcanzó un nuevo récord de **`92.45%`** (con solo 0.15% de pérdida respecto al Float32), manteniendo el presupuesto de hardware intacto (**14.24 KB Flash** y **14.27 KB RAM Arena** en ESP32-S3).
