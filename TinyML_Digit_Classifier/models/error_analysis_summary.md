# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** 3637
- **Exactitud Global:** 87.35%
- **Total Errores:** 460 (12.65%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **0** | **0.0%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **0** | **0.0%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **460** | **100.0%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
| Dígito '0' | Dígito '9' | **64** |
| Dígito '1' | Dígito '7' | **36** |
| Dígito '0' | Dígito '6' | **31** |
| Dígito '0' | Dígito '1' | **28** |
| Dígito '0' | Dígito '5' | **23** |
| Dígito '0' | Dígito '3' | **20** |
| Dígito '0' | Dígito '2' | **17** |
| Dígito '3' | Dígito '5' | **15** |

---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `495` | Dígito '0' | **Dígito '1'** | `23.0%` | `401.52` | `CONFUSION` |
| `2261` | Dígito '3' | **Dígito '4'** | `22.9%` | `1573.79` | `CONFUSION` |
| `2192` | Dígito '2' | **Dígito '5'** | `22.7%` | `1005.61` | `CONFUSION` |
| `3309` | Dígito '8' | **Dígito '4'** | `22.4%` | `1032.59` | `CONFUSION` |
| `1933` | Dígito '1' | **Dígito '7'** | `22.4%` | `206.42` | `CONFUSION` |
| `1721` | Dígito '1' | **Dígito '4'** | `22.4%` | `849.97` | `CONFUSION` |
| `466` | Dígito '0' | **Dígito '2'** | `22.2%` | `1151.85` | `CONFUSION` |
| `303` | Dígito '0' | **Dígito '2'** | `21.8%` | `1215.45` | `CONFUSION` |
| `1290` | Dígito '0' | **Dígito '9'** | `21.7%` | `665.26` | `CONFUSION` |
| `637` | Dígito '0' | **Dígito '3'** | `21.7%` | `1413.75` | `CONFUSION` |

---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
