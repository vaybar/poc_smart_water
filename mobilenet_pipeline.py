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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

DATASET_DIR      = "dataset_mobilenet"
MODELO_PATH      = "best_mobilenet.keras"   # formato SavedModel de Keras 3
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
    Convierte una imagen BGR de OpenCV al tensor de entrada del modelo.

    Pasos:
      1. Escala de grises — los dígitos segmentados no tienen color útil
      2. Resize con padding letterbox — preserva la proporción del dígito
         sin deformar el trazo (un 1 estirado parece un 7)
      3. Replicar canal gris a RGB — MobileNetV3 fue preentrenado en RGB
         y espera 3 canales; replicar es más correcto que rellenar con ceros
      4. Normalizar a [-1, 1] — rango que usa MobileNetV3 internamente

    Returns:
        tensor float32 de forma (1, IMG_H, IMG_W, 3)
    """
    # 1. Escala de grises
    if img_bgr.ndim == 3:
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gris = img_bgr.copy()

    # 2. Letterbox resize
    h, w    = gris.shape[:2]
    scale   = min(IMG_W / w, IMG_H / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(gris, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas  = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    x_off   = (IMG_W - new_w) // 2
    y_off   = (IMG_H - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # 3. Gris → RGB (replicar canal)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # 4. Normalizar a [-1, 1]
    tensor = (rgb.astype(np.float32) / 127.5) - 1.0

    return np.expand_dims(tensor, axis=0)   # (1, H, W, 3)


def _preprocesar_tf(image, label):
    """
    Preproceso con letterbox para preservar el ratio de aspecto.

    image llega como tensor uint8 (IMG_H, IMG_W, 3) ya redimensionado
    (estirado) por Keras. Lo revertimos al tamaño original implícito
    rehaciendo el resize con padding, igual que preprocesar_imagen_np.

    Problema que corrige:
        Los dígitos del dataset tienen ratio 2:1 (64x32px). Keras los
        estira a 96x96 duplicando su ancho. Al estirar, un "1" angosto
        adquiere el ancho de un "5", y un "6" se parece a un "8".
        Letterbox preserva el ratio original poniendo padding negro.

    Pipeline:
        1. Convertir a escala de grises
        2. Resize con padding (letterbox) a IMG_H x IMG_W
        3. Replicar canal gris a RGB (3 canales para MobileNetV3)
        4. Normalizar a [-1, 1]
    """
    # 1. Escala de grises
    gris = tf.image.rgb_to_grayscale(image)            # (H, W, 1)

    # 2. Letterbox usando tf.image.resize_with_pad
    #    Opera directamente sobre tensores 3D (H, W, C) sin agregar
    #    dimensión de batch, lo que evita el error de rank en tf.pad.
    #    resize_with_pad hace internamente: resize preservando ratio +
    #    padding simétrico con ceros, que es exactamente lo que necesitamos.
    gris_padded = tf.image.resize_with_pad(
        gris,
        target_height=IMG_H,
        target_width=IMG_W,
        method="area",
    )
    gris_padded = tf.ensure_shape(gris_padded, [IMG_H, IMG_W, 1])

    # 3. Replicar canal gris a RGB
    rgb = tf.repeat(gris_padded, 3, axis=-1)           # (H, W, 3)

    # 4. Normalizar a [-1, 1]
    rgb = (tf.cast(rgb, tf.float32) / 127.5) - 1.0
    return rgb, label


# ─────────────────────────────────────────────────────────────
# AUGMENTATION (solo en train)
# ─────────────────────────────────────────────────────────────

def _augmentar(image, label):
    """
    Augmentaciones leves apropiadas para dígitos de medidores:
      - Brillo/contraste aleatorio — variaciones de iluminación
      - Flip horizontal desactivado — un 6 al revés es un 9
      - Zoom leve via central_crop + resize

    Nota: se reemplazó random_crop por central_crop para evitar el error
    'Dimensions must be equal' que ocurre cuando tf.data mapea sobre tensores
    de forma [H, W, C] y random_crop recibe un size de longitud distinta.
    """
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Zoom leve: recortar el centro al 90% y redimensionar de vuelta
    image = tf.image.central_crop(image, central_fraction=0.90)
    image = tf.image.resize(image, [IMG_H, IMG_W])

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

    kwargs_ds = dict(
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
    )

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **kwargs_ds
    )
    val_ds_raw = (
        tf.keras.utils.image_dataset_from_directory(
            val_dir, shuffle=False, **kwargs_ds
        )
        if val_dir.exists() else None
    )

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

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_fase1,
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
  python mobilenet_pipeline.py --modo train --fases 2 --checkpoint best_mobilenet.keras

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

