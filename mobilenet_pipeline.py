"""
mobilenet_pipeline.py

Entrena MobileNetV3-Small para clasificar dígitos individuales de medidores
de agua, exporta a TFLite INT8 y realiza inferencia en dispositivos edge.

Mejoras respecto a la versión anterior:
  - MobileNetV3-Small en vez de V2 (~20% más rápido en ARM)
  - Entrada en escala de grises convertida a RGB (compatible con pesos ImageNet)
  - Entrenamiento en dos fases (backbone congelado → fine-tuning)
  - Augmentation en tiempo de entrenamiento
  - Early stopping + ReduceLROnPlateau
  - Exportación INT8 real con dataset de calibración
  - Inferencia robusta con manejo de modelo INT8

Uso:
  Fase 1+2 : python mobilenet_pipeline.py --modo train
  Exportar  : python mobilenet_pipeline.py --modo export
  Inferir   : python mobilenet_pipeline.py --modo infer --imagen digito.png
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from sklearn.utils.class_weight import compute_class_weight
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

DATASET_DIR      = "dataset_mobilenet"
MODELO_PATH      = "mobilenet_model_claude/best_mobilenet.keras"  # formato SavedModel de Keras 3
MODELO_TFLITE    = "best_mobilenet_int8.tflite"

# MobileNetV3-Small acepta múltiples resoluciones.
# 96×96 es el mínimo recomendado — buen equilibrio velocidad/precisión en RPi.
IMG_H      = 96
IMG_W      = 96
NUM_CLASES = 10

# Fases de entrenamiento
EPOCHS_FASE1 = 15   # backbone congelado
EPOCHS_FASE2 = 25   # fine-tuning de las últimas capas
BATCH_SIZE   = 32


# ─────────────────────────────────────────────────────────────
# PREPROCESO — mismo pipeline en entrenamiento e inferencia
# ─────────────────────────────────────────────────────────────

def preprocesar_imagen_np(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preproceso para inferencia — idéntico a _preprocesar_tf:
      1. Escala de grises
      2. Resize estirando a IMG_H x IMG_W (mismo que Keras en entrenamiento)
      3. Replicar canal gris a RGB
      4. Normalizar a [-1, 1]

    No se usa letterbox para mantener consistencia con el entrenamiento,
    donde Keras estira las imágenes al cargar el dataset.
    """
    if img_bgr.ndim == 3:
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gris = img_bgr.copy()

    # Mismo estiramiento que image_dataset_from_directory
    img_96 = cv2.resize(gris, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    """
    1. rgb.astype(np.float32): Las imágenes de OpenCV están compuestas de píxeles 
    con valores enteros del 0 al 255 (0 es negro puro, 255 es blanco puro o color 
    máximo). Las redes neuronales no trabajan bien con números enteros grandes; 
    prefieren números decimales pequeños. Esto convierte los valores enteros a 
    números con decimales (float32).
    2. / 127.5: Divide cada valor de la imagen por 127.5 (que es exactamente la mitad de 255).
    ◦ Si el valor original era 0, ahora es 0.0.
    ◦Si el valor original era 127.5, ahora es 1.0.
    ◦Si el valor original era 255, ahora es 2.0.
    Con esto logramos "aplastar" el rango de [0, 255] a un rango mucho más pequeño: [0.0, 2.0].
    3. - 1.0: Le resta 1.0 a todos los valores anteriores para "centrar" los datos alrededor 
    del cero.
    ◦El 0.0 original (negro) ahora es -1.0.
    ◦El 1.0 original (gris medio) ahora es 0.0.
    ◦El 2.0 original (blanco) ahora es 1.0.
    
    ¿Cuál es el resultado final?
    La imagen completa, que antes tenía valores gigantes entre 0 y 255, ahora tiene todos 
    sus valores comprimidos en el rango entre -1.0 y 1.0.
    
    ¿Por qué MobileNet necesita esto?
    Cuando Google entrenó los modelos MobileNet originales (V2, V3) usó exactamente esta 
    misma fórmula de normalización. Si le pasáramos la imagen directamente en 
    formato [0, 255], las matemáticas dentro de la red explotarían (los gradientes se 
    volverían enormes) y el modelo pensaría que está viendo un rectángulo completamente 
    blanco y brillante, fallando la lectura estrepitosamente.
    """

    rgb    = cv2.cvtColor(img_96, cv2.COLOR_GRAY2RGB)
    tensor = (rgb.astype(np.float32) / 127.5) - 1.0

    return np.expand_dims(tensor, axis=0)   # (1, H, W, 3)


def _preprocesar_tf(image, label):
    """
    Preproceso para el pipeline de entrenamiento.

    La estrategia más robusta para evitar errores de rank dentro de
    tf.data.map es no hacer resize dinámico en absoluto: la imagen ya
    llega a 96x96 redimensionada por Keras, y simplemente convertimos
    a gris, replicamos a RGB y normalizamos.

    El problema de deformación de aspecto (dígitos 64x32 estirados a
    96x96) se resuelve en la carga del dataset usando image_size con
    el ratio correcto o — más simple — aceptando que el modelo aprende
    los dígitos en esa proporción y usando exactamente el mismo
    estiramiento tanto en entrenamiento como en inferencia.

    Ambos pipelines (entrenamiento e inferencia) aplican el mismo
    estiramiento → el modelo ve imágenes consistentes en ambos casos.
    """
    gris = tf.image.rgb_to_grayscale(image)        # (H, W, 1)
    rgb  = tf.repeat(gris, 3, axis=-1)             # (H, W, 3)
    rgb  = (tf.cast(rgb, tf.float32) / 127.5) - 1.0
    return rgb, label


def _augmentar(image, label):
    """
    Augmentaciones basadas en color y brillo — apropiadas para dígitos.

    Para clasificación de dígitos de medidores las transformaciones
    geométricas (rotación, traslación, zoom) son contraproducentes:
    un dígito desplazado o recortado puede volverse irreconocible o
    parecerse a otro dígito. El dominio útil de augmentation es el
    fotométrico: variaciones de iluminación, contraste y ruido que
    simulan las condiciones reales de campo sin alterar la forma del dígito.
    """
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)

    # Ruido gaussiano leve — simula cámaras de baja calidad
    noise = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=0.04, dtype=tf.float32
    )
    image = image + noise
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────

def construir_modelo() -> tf.keras.Model:
    """
    MobileNetV3-Small con cabeza de clasificación para 10 dígitos.

    Arquitectura:
      backbone  : MobileNetV3-Small preentrenado en ImageNet (congelado en fase 1)
      pooling   : GlobalAveragePooling2D
      head      : Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(10, softmax)

    El Dropout doble ayuda a prevenir sobreajuste dado el tamaño
    moderado del dataset (~7000 imágenes de entrenamiento).
    """
    entradas = layers.Input(shape=(IMG_H, IMG_W, 3))

    backbone = applications.MobileNetV3Small(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # ya normalizamos a [-1,1] nosotros
    )
    backbone.trainable = False   # congelado en fase 1

    """
    En Keras (y en Deep Learning en general), este patrón de código es el estándar absoluto 
    cuando se usa lo que se llama la API Funcional de Keras.
    No estamos guardando "datos" en x, estamos guardando "tubos"
    Si x fuera un número (como x = 5, luego x = x + 1), sí estaríamos sobrescribiéndolo. Pero 
    en Keras, x no representa una imagen ni un número, representa un nodo de conexión en un 
    grafo computacional (imagina tuberías de agua).
    Cada línea hace lo siguiente:
    1. Crea una capa matemática nueva (ej. layers.Dense(128)).
    2. Conecta la capa anterior (x) a la entrada de esta nueva capa (eso es lo que hace el (x) 
    al final de cada línea).
    3. Toma la salida de esta nueva capa y la vuelve a llamar x, para que la siguiente capa se 
    pueda conectar a ella.
    """

    x = backbone(entradas, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    salidas = layers.Dense(NUM_CLASES, activation="softmax")(x)

    return models.Model(entradas, salidas)


# ─────────────────────────────────────────────────────────────
# 1. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def _cargar_datasets():
    """Carga y preprocesa los datasets de train y val."""
    train_dir = Path(DATASET_DIR) / "train"
    val_dir   = Path(DATASET_DIR) / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"No existe {train_dir}")

    """
    • labels="inferred": Le confirma a Keras: "Por favor, infiere (deduce) las etiquetas 
    mirando los nombres de las subcarpetas".
    • label_mode="categorical": Convierte el número de la clase en un formato que la red 
    entiende mejor llamado One-Hot Encoding. En lugar de decirle "esta imagen es un 3", 
    le pasa un vector de 10 ceros donde solo la posición 3 es un uno: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    • image_size=(96, 96): Lee las imágenes del disco duro (que en tu caso son rectangulares 
    de 64x32) y las redimensiona automáticamente a 96x96 píxeles antes de dárselas a la red.
    • batch_size=32: No le pasa las imágenes al modelo de a una. Las empaqueta en "lotes" de 
    32 imágenes a la vez. Esto hace que el entrenamiento sea muchísimo más rápido, porque 
    procesa 32 imágenes en paralelo aprovechando la memoria RAM y el procesador/GPU.
    • color_mode="rgb": Se asegura de cargar la imagen con 3 canales de color, que es exactamente 
    lo que el MobileNet espera recibir.
    """
    kwargs_ds = dict(
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
    )

    """
    Lo que hace image_dataset_from_directory es ir a una carpeta, encontrar todas las 
    imágenes que hay allí, darse cuenta automáticamente a qué clase pertenece cada imagen 
    basándose en el nombre de la subcarpeta donde está guardada, y empaquetar todo eso en 
    un formato especial llamado Dataset que la red neuronal puede procesar directamente.
    1. train_dir
    Es la ruta a la carpeta donde están tus datos de entrenamiento (por ejemplo, 
    dataset_mobilenet/train). La función exige que dentro de esa carpeta haya subcarpetas, 
    y asume que el nombre de la subcarpeta es la etiqueta (clase). Como tienes carpetas 
    llamadas 0, 1, 2, etc., ya sabe que las imágenes dentro de 0 son ceros.
    
    2. shuffle=True
    Le dice a TensorFlow que mezcle aleatoriamente el orden de las imágenes. Esto es vital 
    en el entrenamiento. Si no lo mezclaras, el modelo vería primero todas las fotos de 
    los "0", luego todas las de los "1", y así sucesivamente. Al modelo le daría "amnesia" 
    y olvidaría cómo son los ceros para cuando termine de ver los nueves. Mezclarlas 
    asegura que en cada lote vea un poco de todo.
    
    3. **kwargs_ds (Los parámetros empaquetados)
    Los dos asteriscos ** son un truco de Python para "desempaquetar" un diccionario y 
    pasarlo como argumentos. 
    """
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **kwargs_ds
    )
    val_ds_raw = (
        tf.keras.utils.image_dataset_from_directory(
            val_dir, shuffle=False, **kwargs_ds
        )
        if val_dir.exists() else None
    )

    """
    Este bloque de código es el corazón de la eficiencia en TensorFlow. Representa la 
    construcción de un Pipeline de Datos (una línea de ensamblaje) usando la API tf.data.
    En lugar de cargar todas las imágenes en la memoria RAM de tu computadora (lo cual la 
    saturaría y haría que la PC se cuelgue si tuvieras miles de fotos), train_ds crea un 
    "tubo" inteligente que va procesando las imágenes al vuelo, justo a tiempo para 
    dárselas al modelo.
    1. La materia prima: train_ds_raw
    Como vimos antes, train_ds_raw es el lote básico de imágenes (en tu caso, bloques 
    de 32 imágenes de 96x96 píxeles) extraídas directamente del disco duro. Sin embargo, 
    estas imágenes aún están "crudas" (valores de píxeles entre 0 y 255).
    2. Estación 1: .map(_preprocesar_tf, ...)
    El método .map() toma cada lote de imágenes que sale de train_ds_raw y le aplica la 
    función _preprocesar_tf que definimos más arriba.
    • ¿Qué hace aquí? Agarra el lote de 32 imágenes, las convierte a escala de grises, 
    replica el canal para hacerlas RGB de nuevo, y aplica la matemática crucial para 
    comprimir los píxeles al rango decimal [-1.0, 1.0].
    • A partir de este punto, la imagen ya es digerible por MobileNet.
    3. Estación 2: .map(_augmentar, ...)
    Vuelve a tomar el lote (que ya está procesado) y le aplica la función _augmentar.
    • ¿Qué hace aquí? TensorFlow aplica variaciones aleatorias al vuelo. A algunas imágenes
     les sube el brillo, a otras les baja el contraste, o les añade ruido gaussiano.
    • Lo genial de esto es que nunca modificas tus archivos originales en el disco duro. 
    Cada vez que una imagen pasa por este tubo en cada "época" (epoch), recibe una 
    alteración distinta. ¡Es como tener datos de entrenamiento infinitos!
    4. El sistema de entrega: .prefetch(tf.data.AUTOTUNE)
    Esta es la capa de optimización más importante para que el entrenamiento no sea lento. 
    Normalmente, el proceso funciona así:
        1. La CPU lee la imagen y la procesa.
        2. La CPU se la manda a la GPU (o al procesador principal del modelo).
        3. El modelo entrena con ese lote.
        4. Mientras el modelo entrena, la CPU está ociosa esperando a que termine para 
        recién ir a buscar el siguiente lote.
    prefetch (Pre-carga) rompe ese cuello de botella. Le dice a la CPU: "Mientras el 
    modelo está ocupado entrenando con el Lote 1, tú no te quedes de brazos cruzados; 
    ve al disco duro, lee el Lote 2, preprocésalo, auméntalo y déjalo listo en una 
    bandeja de entrada (buffer)". De esta forma, en el milisegundo en que el modelo 
    termina con el Lote 1, el Lote 2 ya está listo para servirse, ahorrando muchísimo 
    tiempo.
    ¿Qué es num_parallel_calls=tf.data.AUTOTUNE?
    Aplicar filtros (map) a 32 imágenes a la vez puede ser pesado. Si le pasamos 
    num_parallel_calls=tf.data.AUTOTUNE, le estamos dando permiso a TensorFlow para 
    que tome el control total de los núcleos (cores) de tu procesador.
    TensorFlow analizará en tiempo real qué tan rápido está tu PC y decidirá dinámicamente 
    si usa 2, 4 u 8 hilos de procesamiento en paralelo para aplicar el preproceso y la 
    aumentación lo más rápido posible sin trabar tu computadora.
    En Resumen:
    La estructura final de train_ds no es una simple lista de imágenes. Es un generador 
    infinito y altamente optimizado que, cada vez que el modelo se lo pide, le escupe un 
    tensor de forma (32, 96, 96, 3) lleno de imágenes normalizadas y con variaciones 
    aleatorias, listas para que el modelo aprenda de ellas a máxima velocidad.
    """
    train_ds = (
        train_ds_raw
        .map(_preprocesar_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_augmentar,      num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(_preprocesar_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        if val_ds_raw else None
    )

    return train_ds, val_ds


def entrenar(
    fases: list[int] | None = None,
    checkpoint: str | None = None,
    capas_descongelar: int = 5,
    lr_fase1: float = 1e-3,
    lr_fase2: float = 1e-6,
    epochs_fase1: int = EPOCHS_FASE1,
    epochs_fase2: int = EPOCHS_FASE2,
):
    """
    Entrena MobileNetV3-Small en una o dos fases.

    Parámetros:
        fases            : lista de fases a ejecutar, ej. [1], [2], [1, 2].
                           Default: [1, 2] (ambas fases).
        checkpoint       : ruta a un modelo .keras para continuar desde ese
                           punto. Si se pasa con fases=[2], se hace fine-tuning
                           sobre ese modelo sin re-entrenar la Fase 1.
                           Si se pasa con fases=[1], se ignora y se construye
                           el modelo desde cero.
        capas_descongelar: número de capas del backbone a descongelar en Fase 2.
                           Valor recomendado: 5 (conservador) a 20 (agresivo).
                           Default: 5 — más seguro que 20 para evitar que el
                           fine-tuning destruya los pesos aprendidos en Fase 1.
        lr_fase1         : learning rate para Fase 1 (default 1e-3).
        lr_fase2         : learning rate para Fase 2 (default 1e-6).
                           Debe ser mucho más bajo que lr_fase1 para no
                           sobreescribir los pesos preentrenados.
        epochs_fase1     : épocas máximas para Fase 1 (default EPOCHS_FASE1).
        epochs_fase2     : épocas máximas para Fase 2 (default EPOCHS_FASE2).

    Ejemplos de uso:
        # Entrenamiento completo desde cero (comportamiento original)
        entrenar()

        # Solo Fase 1 — útil para exportar sin correr Fase 2
        entrenar(fases=[1])

        # Solo Fase 2 sobre un checkpoint existente
        entrenar(fases=[2], checkpoint="best_mobilenet.keras")

        # Fase 2 más conservadora: pocas capas, lr muy bajo
        entrenar(fases=[2], checkpoint="best_mobilenet.keras",
                 capas_descongelar=5, lr_fase2=1e-6)

        # Re-entrenar Fase 1 con más épocas
        entrenar(fases=[1], epochs_fase1=30)
    """
    if fases is None:
        fases = [1, 2]

    fases_validas = {1, 2}
    for f in fases:
        if f not in fases_validas:
            raise ValueError(f"Fase inválida: {f}. Las fases válidas son 1 y 2.")

    print("=" * 60)
    print("ENTRENAMIENTO MobileNetV3-Small — clasificador de dígitos")
    print(f"Fases a ejecutar  : {fases}")
    if checkpoint:
        print(f"Checkpoint inicial: {checkpoint}")
    print(f"Capas descongelar : {capas_descongelar} (solo Fase 2)")
    print(f"LR Fase 1 / Fase 2: {lr_fase1} / {lr_fase2}")
    print("=" * 60)

    train_ds, val_ds = _cargar_datasets()
    monitor = "val_accuracy" if val_ds else "accuracy"

    # ── Fase 1: backbone congelado ────────────────────────────
    if 1 in fases:
        print(f"\n[Fase 1] Backbone congelado — {epochs_fase1} épocas máx.")

        # Fase 1 siempre construye el modelo desde cero
        model = construir_modelo()
        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fase1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        clases = np.arange(10)
        train_dir = Path("dataset_mobilenet/train")
        n_real = {}
        for clase_dir in sorted(train_dir.iterdir()):
            n_real[clase_dir.name] = len(list(clase_dir.glob("*.png")))

        pesos = compute_class_weight('balanced', classes=clases, y=np.repeat(clases, np.array(list(n_real.values()))))

        print("Distribución real del dataset:")
        for clase, n in n_real.items():
            print(f"  clase {clase}: {n}")

        class_weights = dict(enumerate(pesos))
        print("Class weights:", {k: f"{v:.2f}" for k, v in class_weights.items()})

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_fase1,
            class_weight=class_weights,
            callbacks=[
                callbacks.ModelCheckpoint(
                    MODELO_PATH, save_best_only=True,
                    monitor=monitor, verbose=1,
                ),
                callbacks.EarlyStopping(
                    monitor=monitor, patience=5,
                    restore_best_weights=True, verbose=1,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=monitor, factor=0.5, patience=3, verbose=1,
                ),
            ],
        )
        print(f"[Fase 1] Completada. Modelo guardado en: {MODELO_PATH}")

    # ── Fase 2: fine-tuning de las últimas capas ──────────────
    if 2 in fases:
        print(f"\n[Fase 2] Fine-tuning — {epochs_fase2} épocas máx.")
        print(f"         Capas descongeladas: últimas {capas_descongelar}")
        print(f"         Learning rate      : {lr_fase2}")

        # Determinar el modelo de partida:
        #   - Si se ejecutó Fase 1 en esta sesión → usar MODELO_PATH (recién guardado)
        #   - Si se pasó --checkpoint explícito   → usar ese archivo
        #   - Si no hay ninguno de los dos         → error claro
        if 1 in fases:
            ruta_base = MODELO_PATH
        elif checkpoint:
            ruta_base = checkpoint
        elif Path(MODELO_PATH).exists():
            print(f"  Usando checkpoint existente: {MODELO_PATH}")
            ruta_base = MODELO_PATH
        else:
            raise FileNotFoundError(
                "Para ejecutar solo Fase 2 necesitás un modelo de partida.\n"
                "Opciones:\n"
                "  1. Pasar --checkpoint ruta/al/modelo.keras\n"
                "  2. Ejecutar primero Fase 1 con --fases 1"
            )

        if not Path(ruta_base).exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint: {ruta_base}"
            )

        print(f"  Cargando checkpoint: {ruta_base}")
        model = tf.keras.models.load_model(ruta_base)

        # Descongelar solo las últimas N capas del backbone
        backbone = model.layers[1]
        backbone.trainable = True
        for capa in backbone.layers[:-capas_descongelar]:
            capa.trainable = False

        n_entrenables = sum(1 for c in backbone.layers if c.trainable)
        print(f"  Capas entrenables en backbone: {n_entrenables} / {len(backbone.layers)}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fase2),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_fase2,
            callbacks=[
                callbacks.ModelCheckpoint(
                    MODELO_PATH, save_best_only=True,
                    monitor=monitor, verbose=1,
                ),
                callbacks.EarlyStopping(
                    monitor=monitor, patience=8,
                    restore_best_weights=True, verbose=1,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=monitor, factor=0.3, patience=4, verbose=1,
                ),
            ],
        )
        print(f"[Fase 2] Completada. Modelo guardado en: {MODELO_PATH}")

    print(f"\nEntrenamiento finalizado. Modelo final: {MODELO_PATH}")


# ─────────────────────────────────────────────────────────────
# 2. EXPORTACIÓN A TFLITE INT8
# ─────────────────────────────────────────────────────────────

def exportar_a_tflite():
    """
    Exporta el modelo a TFLite con cuantización INT8 completa.

    INT8 real (vs pesos cuantizados):
      - Pesos cuantizados (DEFAULT sin calibración): pesos en int8,
        activaciones en float32. Reduce tamaño pero no acelera en ARM.
      - INT8 completo (con calibración): pesos Y activaciones en int8.
        Acelera 2-3x en RPi y reduce consumo de memoria.

    La calibración usa 200 imágenes del dataset de validación para
    medir el rango de valores de activación de cada capa.
    """
    print("=" * 60)
    print("EXPORTACIÓN A TFLITE INT8")
    print("=" * 60)

    if not Path(MODELO_PATH).exists():
        raise FileNotFoundError(
            f"No se encontró '{MODELO_PATH}'. Ejecutá --modo train primero."
        )

    model = tf.keras.models.load_model(MODELO_PATH)

    # Dataset de calibración: 200 imágenes de validación preprocesadas
    val_dir = Path(DATASET_DIR) / "val"
    imagenes_cal = []

    for clase_dir in sorted(val_dir.iterdir()):
        if not clase_dir.is_dir():
            continue
        archivos = list(clase_dir.glob("*.png"))[:20]  # 20 por clase = 200 total
        for ruta in archivos:
            img = cv2.imread(str(ruta))
            if img is not None:
                tensor = preprocesar_imagen_np(img)
                imagenes_cal.append(tensor)

    imagenes_cal = np.vstack(imagenes_cal).astype(np.float32)
    print(f"Dataset de calibración: {len(imagenes_cal)} imágenes")

    def generador_calibracion():
        for img in imagenes_cal:
            yield [img[np.newaxis, ...]]

    # Convertir con INT8 completo
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = generador_calibracion
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.float32   # entrada sigue siendo float
    converter.inference_output_type     = tf.float32   # salida sigue siendo float

    tflite_model = converter.convert()

    with open(MODELO_TFLITE, "wb") as f:
        f.write(tflite_model)

    size_mb = Path(MODELO_TFLITE).stat().st_size / 1024 / 1024
    print(f"\nExportación exitosa.")
    print(f"  Archivo : {MODELO_TFLITE}")
    print(f"  Tamaño  : {size_mb:.2f} MB")


# ─────────────────────────────────────────────────────────────
# 3. INFERENCIA
# ─────────────────────────────────────────────────────────────

def inferir(ruta_imagen: str):
    """
    Clasifica un recorte individual de dígito usando el modelo TFLite.

    El preproceso es idéntico al del entrenamiento para garantizar
    que el modelo recibe el mismo tipo de dato que aprendió a clasificar.
    """
    print("=" * 60)
    print("INFERENCIA")
    print("=" * 60)

    if not Path(MODELO_TFLITE).exists():
        raise FileNotFoundError(
            f"No se encontró '{MODELO_TFLITE}'. Ejecutá --modo export primero."
        )
    if not Path(ruta_imagen).exists():
        raise FileNotFoundError(f"No se encontró la imagen '{ruta_imagen}'.")

    # Cargar imagen y preprocesar
    img    = cv2.imread(ruta_imagen)
    tensor = preprocesar_imagen_np(img)   # (1, H, W, 3) float32 en [-1, 1]

    # Cargar intérprete TFLite
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=MODELO_TFLITE)
    except ImportError:
        interp = tf.lite.Interpreter(model_path=MODELO_TFLITE)

    interp.allocate_tensors()
    input_det  = interp.get_input_details()[0]
    output_det = interp.get_output_details()[0]

    # Si el modelo INT8 espera uint8 en la entrada, convertir
    if input_det["dtype"] == np.uint8:
        escala, zero_point = input_det["quantization"]
        tensor = (tensor / escala + zero_point).astype(np.uint8)

    interp.set_tensor(input_det["index"], tensor)
    interp.invoke()
    probs = interp.get_tensor(output_det["index"])[0]

    # Si la salida es uint8, desescalar
    if output_det["dtype"] == np.uint8:
        escala, zero_point = output_det["quantization"]
        probs = (probs.astype(np.float32) - zero_point) * escala

    digito    = int(np.argmax(probs))
    confianza = float(probs[digito])

    print(f"\nImagen   : {ruta_imagen}")
    print(f"Dígito   : {digito}")
    print(f"Confianza: {confianza:.2%}")
    print("\nProbabilidades:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 20)
        print(f"  [{i}]  {p:.4f}  {bar}")

    # Imagen de debug — muestra cómo la vio el modelo
    debug = cv2.resize(img, (IMG_W * 3, IMG_H * 3), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        debug, f"{digito} ({confianza:.0%})",
        (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    ruta_debug = Path(ruta_imagen).stem + "_debug.jpg"
    cv2.imwrite(ruta_debug, debug)
    print(f"\nDebug guardado en: {ruta_debug}")

    return digito, confianza


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small — clasificador de dígitos para medidores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Entrenamiento completo (fase 1 + fase 2)
  python mobilenet_pipeline.py --modo train

  # Solo Fase 1
  python mobilenet_pipeline.py --modo train --fases 1

  # Solo Fase 2 sobre un checkpoint existente
  python mobilenet_pipeline.py --modo train --fases 2 --checkpoint mobilenet_model_claude\\best_mobilenet.keras

  # Fase 2 conservadora (recomendada si Fase 2 empeoró el modelo)
  python mobilenet_pipeline.py --modo train --fases 2 \
      --checkpoint best_mobilenet.keras \
      --capas-descongelar 5 --lr-fase2 1e-6

  # Exportar el modelo actual a TFLite
  python mobilenet_pipeline.py --modo export

  # Inferencia sobre un recorte de dígito
  python mobilenet_pipeline.py --modo infer --imagen digito.png
        """
    )
    parser.add_argument(
        "--modo", choices=["train", "export", "infer"], required=True,
        help="train: entrenar | export: exportar a TFLite | infer: clasificar imagen"
    )

    # ── Parámetros de entrenamiento ───────────────────────────
    parser.add_argument(
        "--fases", type=int, nargs="+", default=[1, 2],
        metavar="N",
        help="Fases a ejecutar: 1, 2, o '1 2' para ambas (default: 1 2)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Modelo .keras de partida para Fase 2 (si no se ejecuta Fase 1)"
    )
    parser.add_argument(
        "--capas-descongelar", type=int, default=5,
        dest="capas_descongelar",
        help="Capas del backbone a descongelar en Fase 2 (default: 5)"
    )
    parser.add_argument(
        "--lr-fase1", type=float, default=1e-3,
        dest="lr_fase1",
        help="Learning rate para Fase 1 (default: 1e-3)"
    )
    parser.add_argument(
        "--lr-fase2", type=float, default=1e-6,
        dest="lr_fase2",
        help="Learning rate para Fase 2 (default: 1e-6)"
    )
    parser.add_argument(
        "--epochs-fase1", type=int, default=EPOCHS_FASE1,
        dest="epochs_fase1",
        help=f"Épocas máximas Fase 1 (default: {EPOCHS_FASE1})"
    )
    parser.add_argument(
        "--epochs-fase2", type=int, default=EPOCHS_FASE2,
        dest="epochs_fase2",
        help=f"Épocas máximas Fase 2 (default: {EPOCHS_FASE2})"
    )

    # ── Parámetros generales ──────────────────────────────────
    parser.add_argument(
        "--imagen", type=str,
        help="Ruta al recorte de dígito (requerido en --modo infer)"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_DIR,
        help=f"Directorio del dataset (default: {DATASET_DIR})"
    )
    args = parser.parse_args()

    if args.dataset != DATASET_DIR:
        DATASET_DIR = args.dataset

    if args.modo == "train":
        entrenar(
            fases=args.fases,
            checkpoint=args.checkpoint,
            capas_descongelar=args.capas_descongelar,
            lr_fase1=args.lr_fase1,
            lr_fase2=args.lr_fase2,
            epochs_fase1=args.epochs_fase1,
            epochs_fase2=args.epochs_fase2,
        )
    elif args.modo == "export":
        exportar_a_tflite()
    elif args.modo == "infer":
        if not args.imagen:
            parser.error("--imagen es requerido en modo infer")
        inferir(args.imagen)
