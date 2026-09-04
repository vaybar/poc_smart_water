# FAQ & Decisiones de Diseño - TinyML Digit Classifier

Este documento registra las preguntas frecuentes, decisiones arquitectónicas y explicaciones técnicas sobre el diseño y despliegue de modelos ultra-ligeros para microcontroladores (ESP32-S3).

---

## ❓ Pregunta 1: ¿Por qué armar una Micro-MobileNet desde cero en lugar de usar MobileNet preentrenada reemplazando la capa de clasificación?

### Respuesta:
Existen **3 razones técnicas fundamentales** ligadas a las restricciones estrictas de memoria RAM y Flash en microcontroladores de bajo costo como la **ESP32-S3**:

#### 1.1. El tamaño del modelo preentrenado excede por mucho el límite (< 256 KB - 500 KB)
Incluso usando las versiones estándar más pequeñas de Keras (`tf.keras.applications.MobileNetV3Small` o `MobileNetV2`) con un multiplicador de ancho bajo y eliminando la capa superior (`include_top=False`), el mapa de pesos base sigue teniendo más de 1.5 millones de parámetros.

| Modelo | Parámetros | Tamaño en Flash (INT8) | ¿Entra en < 256 KB - 500 KB? |
| :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** (Estándar) | ~1.530.000 | **~1.300 KB (1.3 MB)** | ❌ No (Supera 5x el límite de 256 KB) |
| **MobileNetV2 ($\alpha=0.35$)** | ~400.000 | **~450 KB - 600 KB** | ❌ No / Al límite de Flash |
| **Nuestra Micro-MobileNet ($\alpha=0.25$)** | **~3.300** | **~30 KB – 50 KB** | **Excelente (Consume solo el 12% del límite)** |

#### 1.2. Memoria RAM de Activaciones (*Tensor Arena*) y Resolución de Entrada
En microcontroladores no solo importa el espacio de almacenamiento (Flash), sino la **RAM dinámica que consumen las capas intermedias durante la inferencia** (*Tensor Arena* en TFLite Micro):

- **MobileNet estándar:** Requiere imágenes de al menos $96 \times 96 \times 3$ (RGB) o $224 \times 224 \times 3$. En la primera capa de $96 \times 96$, un solo mapa de características consume: $96 \times 96 \times 32 \text{ canales} = \mathbf{295\text{ KB de RAM}}$. Esto causa un desbordamiento inmediato de memoria en la ESP32-S3.
- **Nuestra Micro-MobileNet:** Utiliza entrada de **$32 \times 32 \times 1$ (Escala de Grises)**. El primer mapa de activación consume solo $16 \times 16 \times 8 = \mathbf{2\text{ KB de RAM}}$, logrando que el *Tensor Arena* completo de la red sea de solo **~28 KB - 35 KB de RAM**.

#### 1.3. Profundidad de Canales Excesiva para Clasificar 10 Dígitos
MobileNetV2/V3 fue diseñada para clasificar 1.000 clases complejas de ImageNet, expandiendo los canales internos hasta 576 o 1280 filtros. Para clasificar dígitos (0 al 9) no se requieren 1280 filtros. 

Nuestra Micro-MobileNet mantiene los **mismos bloques de construcción de MobileNet** (*Depthwise Conv 3x3* + *Pointwise Conv 1x1* + *ReLU6* + *Batch Normalization*), pero escalados a 16, 32 y 64 canales máximo.

---

## ❓ Pregunta 2: ¿Necesitamos TensorFlow instalado en la ESP32-S3 para ejecutar el modelo?

### Respuesta:
**No.** 
- **En la computadora (Entorno de desarrollo):** Usamos Python con **TensorFlow / Keras** para diseñar, entrenar, evaluar y realizar la cuantización INT8.
- **En el microcontrolador ESP32-S3 (Entorno de despliegue):** TensorFlow **NO** corre ni está instalado. El modelo se exporta a una cabecera C (`digit_model_quantized.h`) conteniendo una matriz constante de bytes (`const unsigned char g_digit_model[] PROGMEM`). En la ESP32-S3 corremos una librería ultraligera en C++ llamada **TensorFlow Lite for Microcontrollers (TFLite Micro)** o **ESP-NN**, que ocupa solo ~25 KB de binario C++.

---

## ❓ Pregunta 3: ¿Qué es la memoria *Tensor Arena* y por qué es crítica en TinyML?

### Respuesta:
La *Tensor Arena* es un bloque continuo de memoria RAM (SRAM) reservado al iniciar la aplicación en el microcontrolador. TFLite Micro utiliza este espacio para:
1. Almacenar los tensores de entrada y salida.
2. Almacenar los mapas de activaciones temporales producidos entre capa y capa.

A diferencia de un PC donde la RAM es abundante, en la ESP32-S3 disponemos de ~512 KB de SRAM compartida con el sistema operativo y el stack de comunicaciones. Garantizar que la Tensor Arena sea **< 40 KB** permite ejecutar Edge AI con margen de estabilidad total.

---

## ❓ Pregunta 4: ¿Qué ventajas aporta la Convolución Profunda Separable (*Depthwise Separable Convolution*)?

### Respuesta:
Dividir una convolución 2D estándar en dos pasos:
1. **Depthwise Conv:** Aplica un filtro espacial $3 \times 3$ individual a cada canal de entrada.
2. **Pointwise Conv:** Aplica una convolución $1 \times 1$ para combinar linealmente los canales.

Esto reduce la carga de cálculos (FLOPs) y el número de parámetros por un factor de aproximadamente **8 a 9 veces** en comparación con una convolución tradicional $3 \times 3$, manteniendo una capacidad de representación equivalente.

---

## ❓ Pregunta 5: ¿Cómo funciona la cuantización INT8 y cuánto reduce el tamaño?

### Respuesta:
La cuantización INT8 convierte los pesos y activaciones del modelo de punto flotante de 32 bits (`float32`) a enteros de 8 bits (`int8` / `uint8`).

- **Reducción de tamaño:** Disminuye el peso del modelo en un **75%** (4 bytes por peso $\to$ 1 byte por peso).
- **Aceleración en hardware:** Permite utilizar instrucciones de enteros SIMD del procesador Xtensa LX7 de la ESP32-S3 (vectorización DSP), acelerando la inferencia hasta 4x respecto a operaciones en coma flotante.
