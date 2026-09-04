from pathlib import Path
import cv2
import numpy as np

train_dir = Path("dataset_mobilenet/train")

print(f"{'Clase':<6} {'N imágenes':>10} {'Mean std':>10} {'Ejemplos únicos (approx)':>25}")
print("-" * 55)

for clase_dir in sorted(train_dir.iterdir()):
    imgs = list(clase_dir.glob("*.png"))
    stds = []
    means = []
    for ruta in imgs[:50]:  # muestra de 50
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stds.append(img.std())
            means.append(img.mean())

    # Contar cuántas imágenes provienen de imágenes originales distintas
    # El nombre tiene formato: ds{N}_train{M}_pos{P}.png
    # Imágenes del mismo M son el mismo dial original
    origenes = set(ruta.stem.rsplit('_pos', 1)[0] for ruta in imgs)

    print(f"  {clase_dir.name:<4}  {len(imgs):>10}  {np.mean(stds):>10.1f}  "
          f"{len(origenes):>10} imágenes originales")