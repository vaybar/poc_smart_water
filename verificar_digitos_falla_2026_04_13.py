# Correr esto en Colab para inspeccionar visualmente las imágenes
# que fallan consistentemente en TODOS los entrenamientos
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

persistentes = [
    ("1", "ds5_train206_pos4.png"),
    ("3", "ds5_train465_pos4.png"),
    ("4", "ds5_train691_pos4.png"),
    ("4", "ds5_train521_pos4.png"),
    ("5", "ds5_train541_pos4.png"),
    ("6", "ds5_train567_pos4.png"),
    ("6", "ds5_train568_pos4.png"),
    ("6", "ds5_train719_pos4.png"),
    ("9", "ds5_train889_pos4.png"),
]

fig, axes = plt.subplots(1, len(persistentes), figsize=(18, 3))
for ax, (clase, nombre) in zip(axes, persistentes):
    ruta = Path(f"dataset_mobilenet/sample/{clase}/{nombre}")
    img  = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title(f"clase={clase}\n{nombre[-12:]}", fontsize=7)
    ax.axis('off')
plt.suptitle("Imágenes que fallan en TODOS los entrenamientos — ¿son correctas?")
plt.tight_layout()
plt.show()