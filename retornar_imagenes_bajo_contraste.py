from pathlib import Path
import cv2
import numpy as np

dataset_dir = Path("dataset_mobilenet/train")
sospechosas = []

for clase_dir in sorted(dataset_dir.iterdir()):
    for ruta in clase_dir.glob("*.png"):
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.std() < 30:
            sospechosas.append((clase_dir.name, ruta.name, img.std()))

print(f"Imágenes con std < 30: {len(sospechosas)}")
for clase, nombre, std in sorted(sospechosas, key=lambda x: x[2])[:20]:
#for clase, nombre, std in sospechosas:
    print(f"  clase={clase}  std={std:.1f}  {nombre}")