# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** 3637
- **Exactitud Global:** 87.87%
- **Total Errores:** 441 (12.13%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **0** | **0.0%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **0** | **0.0%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **441** | **100.0%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
| Dígito '1' | Dígito '7' | **64** |
| Dígito '0' | Dígito '6' | **50** |
| Dígito '0' | Dígito '3' | **41** |
| Dígito '0' | Dígito '9' | **32** |
| Dígito '0' | Dígito '7' | **27** |
| Dígito '0' | Dígito '1' | **21** |
| Dígito '0' | Dígito '5' | **18** |
| Dígito '0' | Dígito '2' | **13** |

---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1811` | Dígito '1' | **Dígito '7'** | `23.1%` | `711.59` | `CONFUSION` |
| `2133` | Dígito '2' | **Dígito '4'** | `23.1%` | `566.9` | `CONFUSION` |
| `1699` | Dígito '1' | **Dígito '7'** | `23.0%` | `1490.71` | `CONFUSION` |
| `795` | Dígito '0' | **Dígito '6'** | `23.0%` | `592.76` | `CONFUSION` |
| `1658` | Dígito '1' | **Dígito '7'** | `23.0%` | `327.89` | `CONFUSION` |
| `2023` | Dígito '2' | **Dígito '3'** | `23.0%` | `281.48` | `CONFUSION` |
| `137` | Dígito '0' | **Dígito '6'** | `22.9%` | `543.35` | `CONFUSION` |
| `2942` | Dígito '6' | **Dígito '5'** | `22.8%` | `205.67` | `CONFUSION` |
| `343` | Dígito '0' | **Dígito '7'** | `22.8%` | `392.33` | `CONFUSION` |
| `2206` | Dígito '2' | **Dígito '3'** | `22.7%` | `705.61` | `CONFUSION` |

---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
