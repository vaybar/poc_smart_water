# En Colab, correr inferencia con el modelo .keras directamente
# usando el preproceso CORRECTO (sin letterbox, igual que entrenamiento)
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

model    = tf.keras.models.load_model("mobilenet_model_claude/best_mobilenet.keras")
IMG_H, IMG_W = 96, 96

def preprocesar_igual_que_entrenamiento(ruta_img):
    img  = cv2.imread(str(ruta_img))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (IMG_W, IMG_H))          # ESTIRA, sin letterbox
    gris = tf.image.rgb_to_grayscale(img[None,...])
    rgb  = tf.repeat(gris, 3, axis=-1)
    return (tf.cast(rgb, tf.float32) / 127.5) - 1.0

# Probar con imágenes del sample
sample_dir = Path("dataset_mobilenet//sample")
correctas  = 0
total      = 0
for clase_dir in sorted(sample_dir.iterdir()):
    for ruta in list(clase_dir.glob("*.png"))[:10]:
        tensor = preprocesar_igual_que_entrenamiento(ruta)
        pred   = model.predict(tensor, verbose=0)
        clase_pred = np.argmax(pred)
        clase_real = int(clase_dir.name)
        correctas += clase_pred == clase_real
        total     += 1
        print(f"  {ruta.name}: real={clase_real} pred={clase_pred} "
              f"conf={pred[0][clase_pred]:.0%} {'OK' if clase_pred==clase_real else 'FALLO'}")

print(f"\nAccuracy con preproceso correcto: {correctas/total:.1%}")