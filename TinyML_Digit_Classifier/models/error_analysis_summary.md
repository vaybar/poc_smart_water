# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** 3637
- **Exactitud Global:** 91.94%
- **Total Errores:** 293 (8.06%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **0** | **0.0%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **293** | **100.0%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **0** | **0.0%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
| Dígito '1' | Dígito '7' | **47** |
| Dígito '0' | Dígito '9' | **23** |
| Dígito '0' | Dígito '1' | **21** |
| Dígito '0' | Dígito '3' | **18** |
| Dígito '0' | Dígito '2' | **16** |
| Dígito '0' | Dígito '5' | **11** |
| Dígito '7' | Dígito '1' | **11** |
| Dígito '0' | Dígito '6' | **10** |

---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2266` | Dígito '3' | **Dígito '2'** | `23.1%` | `1705.8` | `CROPPED/OFF_CENTER` |
| `1666` | Dígito '1' | **Dígito '7'** | `22.8%` | `409.07` | `CROPPED/OFF_CENTER` |
| `1874` | Dígito '1' | **Dígito '0'** | `22.7%` | `573.5` | `CROPPED/OFF_CENTER` |
| `1764` | Dígito '1' | **Dígito '7'** | `22.6%` | `262.72` | `CROPPED/OFF_CENTER` |
| `3592` | Dígito '9' | **Dígito '0'** | `22.6%` | `654.74` | `CROPPED/OFF_CENTER` |
| `1647` | Dígito '1' | **Dígito '7'** | `22.4%` | `263.24` | `CROPPED/OFF_CENTER` |
| `1661` | Dígito '1' | **Dígito '7'** | `22.4%` | `85.81` | `CROPPED/OFF_CENTER` |
| `1728` | Dígito '1' | **Dígito '7'** | `22.3%` | `351.69` | `CROPPED/OFF_CENTER` |
| `3579` | Dígito '9' | **Dígito '2'** | `22.1%` | `638.4` | `CROPPED/OFF_CENTER` |
| `1562` | Dígito '1' | **Dígito '7'** | `22.0%` | `197.04` | `CROPPED/OFF_CENTER` |

---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
