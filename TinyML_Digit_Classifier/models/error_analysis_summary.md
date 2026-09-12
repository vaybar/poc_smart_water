# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** 3637
- **Exactitud Global:** 27.17%
- **Total Errores:** 2649 (72.83%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **0** | **0.0%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **0** | **0.0%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **2649** | **100.0%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
| Dígito '0' | Dígito '9' | **610** |
| Dígito '1' | Dígito '0' | **290** |
| Dígito '2' | Dígito '9' | **164** |
| Dígito '3' | Dígito '9' | **156** |
| Dígito '4' | Dígito '0' | **149** |
| Dígito '5' | Dígito '9' | **146** |
| Dígito '7' | Dígito '0' | **123** |
| Dígito '8' | Dígito '9' | **123** |

---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1767` | Dígito '1' | **Dígito '0'** | `23.1%` | `299.57` | `CONFUSION` |
| `1706` | Dígito '1' | **Dígito '0'** | `23.1%` | `528.36` | `CONFUSION` |
| `1682` | Dígito '1' | **Dígito '0'** | `23.1%` | `813.37` | `CONFUSION` |
| `2573` | Dígito '4' | **Dígito '0'** | `23.1%` | `278.92` | `CONFUSION` |
| `1612` | Dígito '1' | **Dígito '0'** | `23.1%` | `815.15` | `CONFUSION` |
| `2566` | Dígito '4' | **Dígito '0'** | `23.0%` | `2061.09` | `CONFUSION` |
| `1831` | Dígito '1' | **Dígito '0'** | `23.0%` | `1377.27` | `CONFUSION` |
| `2538` | Dígito '4' | **Dígito '0'** | `23.0%` | `1334.71` | `CONFUSION` |
| `1815` | Dígito '1' | **Dígito '0'** | `23.0%` | `319.74` | `CONFUSION` |
| `2527` | Dígito '4' | **Dígito '0'** | `23.0%` | `579.76` | `CONFUSION` |

---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
