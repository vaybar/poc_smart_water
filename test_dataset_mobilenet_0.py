import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

carpeta_0 = Path("dataset_mobilenet/train/0")
muestras  = random.sample(list(carpeta_0.glob("*.png")), 20)

fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for ax, ruta in zip(axes.flat, muestras):
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(ruta.stem[-6:], fontsize=6)
plt.suptitle("Muestra de clase 0 — verificar que sean realmente ceros")
plt.tight_layout()
plt.show()