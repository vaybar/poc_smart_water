# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** 3637
- **Exactitud Global:** 96.73%
- **Total Errores:** 119 (3.27%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **0** | **0.0%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **0** | **0.0%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **119** | **100.0%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
| Dígito '0' | Dígito '9' | **27** |
| Dígito '0' | Dígito '1' | **11** |
| Dígito '0' | Dígito '6' | **8** |
| Dígito '0' | Dígito '3' | **4** |
| Dígito '0' | Dígito '8' | **4** |
| Dígito '9' | Dígito '6' | **4** |
| Dígito '0' | Dígito '7' | **3** |
| Dígito '1' | Dígito '3' | **3** |

---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `238` | Dígito '0' | **Dígito '1'** | `23.0%` | `346.32` | `CONFUSION` |
| `3434` | Dígito '8' | **Dígito '6'** | `23.0%` | `590.55` | `CONFUSION` |
| `1496` | Dígito '0' | **Dígito '9'** | `22.7%` | `557.1` | `CONFUSION` |
| `1131` | Dígito '0' | **Dígito '7'** | `22.6%` | `1240.02` | `CONFUSION` |
| `1300` | Dígito '0' | **Dígito '9'** | `22.5%` | `699.94` | `CONFUSION` |
| `495` | Dígito '0' | **Dígito '1'** | `22.4%` | `234.98` | `CONFUSION` |
| `1302` | Dígito '0' | **Dígito '9'** | `22.4%` | `528.06` | `CONFUSION` |
| `2735` | Dígito '5' | **Dígito '3'** | `21.9%` | `328.92` | `CONFUSION` |
| `2342` | Dígito '3' | **Dígito '2'** | `21.8%` | `495.04` | `CONFUSION` |
| `174` | Dígito '0' | **Dígito '9'** | `21.8%` | `107.27` | `CONFUSION` |

---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
