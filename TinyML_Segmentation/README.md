# TinyML_Segmentation: Localizador y Segmentador de Medidor de Agua para ESP32 Clásico

Este módulo implementa el pipeline de **localización de la ventanilla del medidor de agua, corrección de perspectiva/rotación y segmentación de dígitos** diseñado específicamente para operar con los límites estrictos de hardware de una **ESP32 clásico (Xtensa LX6 @ 240 MHz, 520 KB SRAM, 4 MB PSRAM)**.

Entrega exactamente $N$ recortes de dígitos de tamaño estricto **$64 \times 32 \times 1$ en escala de grises (uint8)**, listos para alimentar de forma directa el modelo de clasificación en [`TinyML_Digit_Classifier`](file:///c:/Users/vanes/Documents/desarrollos/poc_smart_water/TinyML_Digit_Classifier).

---

## 1. Presupuesto de Hardware (ESP32 Clásico)

| Recurso | Límite ESP32 Clásico | Presupuesto TinyML_Segmentation | Margen Libre |
| :--- | :---: | :---: | :---: |
| **Flash ROM (INT8)** | 1.5 - 2.5 MB (App) | **$< 80\text{ KB}$** (~38 KB típico) | **> 95% libre** |
| **Tensor Arena (SRAM)** | 520 KB (heap ~300 KB) | **$< 35\text{ KB}$** (~24.5 KB) | **> 90% libre** |
| **Búfer de Captura** | 4 MB PSRAM | **16 KB** ($128 \times 128 \times 1$) | Despreciable |
| **Dependencias C** | Sin OpenCV | **C puro estándar (`segmenter_mcu.c`)** | 100% portable |

---

## 2. Flujo de Procesamiento

```
[Captura OV2640] (Cualquier ángulo, barro, suciedad)
       │
       ▼
1. Submuestreo a Thumbnail (128×128×1 Grayscale) ──► Solo 16 KB en RAM
       │
       ▼
2. Micro-Corner-Regressor (TFLite Micro INT8)    ──► Predice 4 esquinas [TL, TR, BR, BL]
       │
       ▼
3. Mapeo Bilineal Cuadrilátero en C              ──► Rectifica el ángulo a tira 64 × (N*32)
       │
       ▼
4. Corte Fijo por Ranuras Mecánicas (*Slots*)    ──► Divide en N recortes de 64×32×1
       │
       ▼
[Salida: N tensores (64, 32, 1) uint8]           ──► Entrada a Micro-MobileNet
```

---

## 3. Estructura de Archivos

```
TinyML_Segmentation/
├── config.py                 # Parámetros globales y presupuestos para ESP32
├── audit_dataset.py          # Auditoría de imágenes, esquinas y ángulos de rotación
├── model_builder.py          # Arquitectura Micro-Corner-Regressor (Depthwise-Separable)
├── dataset.py                # Loader, aumentación de datos y generador de calibración PTQ
├── train.py                  # Script de entrenamiento con callbacks automáticos
├── evaluate_metrics.py       # Cálculo de MAE de esquinas, IoU de polígono y error angular
├── quantize_and_export.py    # Cuantización Full INT8 y exportación a header C (.h)
├── warp_utils.py             # Rectificación bilineal y corte de slots (Python/NumPy)
├── segmenter.py              # Clase orquestadora end-to-end TinyMLSegmenter
├── test_segmenter.py         # Script de validación de contrato y diagnóstico visual
├── experiment_logger.py      # Gestor automático de experimentos y bitácora
├── BITACORA_EXPERIMENTOS.md  # Tabla maestra histórica de experimentos
└── mcu_c_code/               # Código en C nativo para firmware ESP32
    ├── segmenter_mcu.h       # Cabeceras en C puro
    └── segmenter_mcu.c       # Implementación de downsample, warp y slot slice
```

---

## 4. Guía de Ejecución en Google Colab

Puedes entrenar el modelo en Google Colab con aceleración por GPU siguiendo estos pasos:

### Paso 1: Subir o clonar el repositorio
```bash
!git clone <URL_DE_TU_REPOSITORIO>
%cd poc_smart_water/TinyML_Segmentation
```

### Paso 2: Auditar el dataset
```bash
!python audit_dataset.py
```

### Paso 3: Entrenar el modelo
```bash
!python train.py --alpha 0.50 --epochs 40 --batch-size 32 --notes "Experimento inicial en Colab"
```
*Esto generará automáticamente el registro en `BITACORA_EXPERIMENTOS.md` y `experiments/EXP_001/report.md`.*

### Paso 4: Cuantizar a Full INT8 y Generar Header en C
```bash
!python quantize_and_export.py
```
*Se generará `models/micro_corner_regressor_int8.tflite` y `mcu_c_code/corner_model_quantized.h` listo para descargar.*

---

## 5. Pruebas Locales en PC

Para auditar el dataset y verificar que el extractor cumple con el contrato de $64 \times 32$:

```powershell
# 1. Auditar dataset
python audit_dataset.py

# 2. Probar segmentación y verificar recortes de 64x32
python test_segmenter.py
```
Las imágenes intermedias de diagnóstico se guardarán en la carpeta `TinyML_Segmentation/debug/`.

---

## 6. Origen del Dataset y Extracción de Anotaciones (Labels)

### 6.1 Paper Científico y Repositorio de Datos
El dataset base proviene del trabajo de investigación:
> **Paper:** *"A Comprehensive Dataset for Word-Wheel Water Meter Reading Under Challenging Conditions"*  
> **Repositorio Dryad:** [doi:10.5061/dryad.7d7wm3860](https://doi.org/10.5061/dryad.7d7wm3860)  
> **Características:** Más de 50.000 imágenes tomadas en condiciones adversas reales (rotaciones, suciedad/tierra, manchas de barro, reflejos de luz solar y desenfoque).

### 6.2 Proceso de Extracción de Esquinas (de Máscaras a Coordenadas)
En el dataset original, cada imagen contiene una **máscara binaria** de segmentación (`masks/*.png`) donde el área de la ventanilla/dial es blanca (`255`) y el fondo es negro (`0`).

A través del script de conversión [`convert_dataset_to_yolo_format.py`](file:///c:/Users/vanes/Documents/desarrollos/poc_smart_water/convert_dataset_to_yolo_format.py), se derivaron las coordenadas geométricas exactas mediante el siguiente proceso:
1. **Detección de Contorno Principal:** Se extrae el contorno de la máscara con `cv2.findContours`.
2. **Rectángulo Mínimo Rotado:** Se calcula la orientación y caja ajustada con `cv2.minAreaRect(contorno)`. A diferencia de una caja horizontal, `minAreaRect` captura el ángulo real de inclinación ($\theta$).
3. **Obtención de Vértices:** Con `cv2.boxPoints` se obtienen las 4 esquinas del polígono.
4. **Ordenamiento Estricto de Puntos:** Los vértices se ordenan consistentemente como:
   $$\text{P}_0 \text{ (Arriba-Izq)} \longrightarrow \text{P}_1 \text{ (Arriba-Der)} \longrightarrow \text{P}_2 \text{ (Abajo-Der)} \longrightarrow \text{P}_3 \text{ (Abajo-Izq)}$$
5. **Normalización en $[0, 1]$:** Se divide cada coordenada $(x, y)$ por el ancho y alto original de la imagen.

### 6.3 Estructura Exacta de los Archivos de Labels (`water_meter/labels/*.txt`)
Cada línea de anotación consta de **13 valores numéricos** normalizados:
```text
<clase>  <cx> <cy> <w> <h>  <x1> <y1>  <x2> <y2>  <x3> <y3>  <x4> <y4>
```

* **`clase`** (1 valor): `0` (ventanilla del medidor).
* **`cx, cy, w, h`** (4 valores): Centro, ancho y alto de la caja horizontal envolvente (*Bounding Box*).
* **`x1..y4`** (8 valores): Las 4 coordenadas de las esquinas en 2D que alimentan directamente a nuestro modelo `Micro-Corner-Regressor`:
  - `(x1, y1)`: Esquina Superior Izquierda (TL)
  - `(x2, y2)`: Esquina Superior Derecha (TR)
  - `(x3, y3)`: Esquina Inferior Derecha (BR)
  - `(x4, y4)`: Esquina Inferior Izquierda (BL)

> [!NOTE]
> **Diferencia entre Esquinas y Dígitos:** La ventanilla es un cuadrilátero único delimitado por **4 esquinas (8 coordenadas)**. Los **6 dígitos** ($N=6$) no se anotan individualmente en esta etapa: se obtienen dividiendo la ventanilla rectificada en 6 ranuras mecánicas de $64 \times 32$ px de forma geométrica y determinista.

