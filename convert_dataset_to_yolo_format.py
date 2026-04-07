"""
convert_to_yolo.py

Convierte el dataset del paper "A Comprehensive Dataset for Word-Wheel
Water Meter Reading Under Challenging Conditions" al formato YOLO TXT
con keypoints, listo para entrenar YOLOv8n.

Estructura esperada del dataset original:
    dataset_original/
    ├── images/          *.png
    ├── masks/           *.png  (mismo nombre que la imagen)
    └── labels.csv       sin header, columnas posicionales:
                           [0] filename       ej. train0.png
                           [1] clear          1 = imagen en condiciones normales
                           [2] blurry         1 = imagen borrosa
                           [3] dial-stained   1 = dial manchado
                           [4] soil-covered   1 = cubierto de tierra/suciedad
                           [5] dark           1 = poca iluminación
                           [6] reflective     1 = reflejo en el vidrio
                           [7] six-digit      1 = medidor de 6 dígitos

Estructura de salida (formato YOLO keypoints):
    water_meter_yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── water_meter.yaml
    └── dataset_stats.csv    resumen de flags por split

Uso:
    python convert_to_yolo.py \\
        --dataset /ruta/al/dataset_original \\
        --output  /ruta/al/water_meter_yolo \\
        --split   0.8 0.1 0.1 \\
        --verify

Dependencias:
    pip install opencv-python-headless numpy pandas tqdm pyyaml
"""

import cv2
import numpy as np
import pandas as pd
import shutil
import argparse
import yaml
import random
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# COLUMNAS DEL CSV (posicionales, sin header)
# ─────────────────────────────────────────────────────────────

CSV_COLUMNAS = [
    "filename",     # 0
    "clear",        # 1
    "blurry",       # 2
    "dial_stained", # 3
    "soil_covered", # 4
    "dark",         # 5
    "reflective",   # 6
    "six_digit",    # 7
]

FLAGS_CONDICION = ["blurry", "dial_stained", "soil_covered", "dark", "reflective"]


# ─────────────────────────────────────────────────────────────
# LECTURA DEL CSV
# ─────────────────────────────────────────────────────────────

def cargar_csv(ruta_csv: Path) -> pd.DataFrame:
    """
    Carga el CSV sin header asignando los nombres de columna conocidos.

    Valida que tenga exactamente 8 columnas. Si tiene más o menos,
    informa cuántas encontró para facilitar el diagnóstico.
    """
    df = pd.read_csv(ruta_csv, header=None)

    if df.shape[1] != len(CSV_COLUMNAS):
        raise ValueError(
            f"El CSV tiene {df.shape[1]} columnas pero se esperaban {len(CSV_COLUMNAS)}.\n"
            f"Columnas esperadas: {CSV_COLUMNAS}\n"
            f"Primeras filas del CSV:\n{df.head(3).to_string()}"
        )

    df.columns = CSV_COLUMNAS

    # Convertir flags a entero por si vienen como float debido a NaN
    for col in CSV_COLUMNAS[1:]:
        df[col] = df[col].fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN DE BBOX Y KEYPOINTS DESDE MÁSCARA PNG
# ─────────────────────────────────────────────────────────────

def extraer_bbox_normalizado(
    mascara: np.ndarray,
    ancho_img: int,
    alto_img: int,
) -> tuple[float, float, float, float] | None:
    """
    Calcula el bounding box del dial en la máscara y lo normaliza.

    La máscara es binaria: dial en blanco (255), fondo en negro (0).
    cv2.boundingRect devuelve el rectángulo alineado a los ejes que
    contiene todos los píxeles blancos.

    Si la máscara tiene distinto tamaño que la imagen (puede ocurrir
    en algunos datasets) la redimensiona antes de procesar.

    Returns:
        (cx, cy, w, h) normalizados entre 0 y 1, o None si vacía.
        Todos los valores están garantizados en el rango [0, 1].
    """
    if mascara.ndim == 3:
        mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)

    # Si la máscara tiene distinto tamaño que la imagen, redimensionar
    if mascara.shape[1] != ancho_img or mascara.shape[0] != alto_img:
        mascara = cv2.resize(
            mascara, (ancho_img, alto_img), interpolation=cv2.INTER_NEAREST
        )

    _, binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None

    contorno_principal = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_principal)

    # Clamp de píxeles antes de normalizar — evita valores fuera de imagen
    x = max(0, min(x, ancho_img - 1))
    y = max(0, min(y, alto_img - 1))
    w = max(1, min(w, ancho_img - x))
    h = max(1, min(h, alto_img - y))

    cx  = (x + w / 2) / ancho_img
    cy  = (y + h / 2) / alto_img
    w_n = w / ancho_img
    h_n = h / alto_img

    # Garantía final: todos los valores normalizados en [0, 1]
    cx  = max(0.0, min(1.0, cx))
    cy  = max(0.0, min(1.0, cy))
    w_n = max(0.0, min(1.0, w_n))
    h_n = max(0.0, min(1.0, h_n))

    return cx, cy, w_n, h_n


def extraer_keypoints_normalizados(
    mascara: np.ndarray,
    ancho_img: int,
    alto_img: int,
) -> list[tuple[float, float]] | None:
    """
    Extrae los 4 vértices del dial usando el rectángulo mínimo rotado.

    cv2.minAreaRect encuentra el rectángulo más pequeño posible aunque
    esté inclinado, capturando la perspectiva real del medidor.

    Orden de salida: arriba-izq, arriba-der, abajo-der, abajo-izq.
    Todos normalizados entre 0 y 1.
    """
    if mascara.ndim == 3:
        mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)

    if mascara.shape[1] != ancho_img or mascara.shape[0] != alto_img:
        mascara = cv2.resize(
            mascara, (ancho_img, alto_img), interpolation=cv2.INTER_NEAREST
        )

    _, binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None

    contorno_principal = max(contornos, key=cv2.contourArea)
    rect     = cv2.minAreaRect(contorno_principal)
    vertices = cv2.boxPoints(rect).astype(np.float32)
    vertices = _ordenar_vertices(vertices)

    keypoints = [
        (
            max(0.0, min(1.0, float(pt[0] / ancho_img))),
            max(0.0, min(1.0, float(pt[1] / alto_img))),
        )
        for pt in vertices
    ]
    return keypoints


def _ordenar_vertices(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos como: arriba-izq, arriba-der, abajo-der, abajo-izq.

    Mismo algoritmo que RectificadorPerspectiva en water_meter_pipeline.py
    para garantizar consistencia entre entrenamiento e inferencia.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]

    ordered[0] = pts[np.argmin(s)]   # arriba-izquierda  (menor x+y)
    ordered[2] = pts[np.argmax(s)]   # abajo-derecha     (mayor x+y)
    ordered[1] = pts[np.argmax(d)]   # arriba-derecha    (mayor x-y)
    ordered[3] = pts[np.argmin(d)]   # abajo-izquierda   (menor x-y)

    return ordered


# ─────────────────────────────────────────────────────────────
# FORMATO YOLO TXT
# ─────────────────────────────────────────────────────────────

def construir_linea_yolo(
    bbox: tuple[float, float, float, float],
    keypoints: list[tuple[float, float]] | None,
    clase: int = 0,
) -> str:
    """
    Construye una línea en formato YOLO keypoints:

        <clase> <cx> <cy> <w> <h> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

    Con kpt_shape: [4, 2] YOLO espera solo x e y por keypoint, sin valor
    de visibilidad. Incluir el valor de visibilidad (v) desplaza todos los
    índices y hace que YOLO interprete los 2 como coordenadas fuera de rango.

    Si no hay keypoints disponibles genera detección simple sin los puntos.
    """
    cx, cy, w, h = bbox
    partes = [str(clase), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    if keypoints:
        for kx, ky in keypoints:
            partes += [f"{kx:.6f}", f"{ky:.6f}"]

    return " ".join(partes)


# ─────────────────────────────────────────────────────────────
# CONVERSIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def convertir_dataset(
    dir_dataset: Path,
    dir_salida: Path,
    proporciones: tuple[float, float, float] = (0.8, 0.1, 0.1),
    semilla: int = 42,
    nombre_csv: str = "labels.csv",
):
    """
    Convierte el dataset completo al formato YOLO.

    El split se hace de forma estratificada por la columna "clear":
    se garantiza que cada split tenga proporción similar de imágenes
    normales vs imágenes con condiciones desafiantes. Esto evita que
    el modelo se entrene solo con imágenes limpias y se evalúe solo
    con las difíciles, o viceversa.
    """
    assert abs(sum(proporciones) - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    dir_imagenes = dir_dataset / "images"
    dir_mascaras = dir_dataset / "masks"
    ruta_csv     = dir_dataset / nombre_csv

    for ruta, nombre in [(dir_imagenes, "images/"), (dir_mascaras, "masks/")]:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró '{nombre}' en {dir_dataset}\n"
                f"Estructura esperada:\n"
                f"  {dir_dataset}/\n"
                f"  ├── images/\n"
                f"  ├── masks/\n"
                f"  └── {nombre_csv}"
            )
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {ruta_csv}")

    df = cargar_csv(ruta_csv)
    print(f"[convert] CSV cargado: {len(df)} filas")
    _imprimir_stats_flags(df, "Dataset completo")

    for split in ["train", "val", "test"]:
        (dir_salida / "images" / split).mkdir(parents=True, exist_ok=True)
        (dir_salida / "labels" / split).mkdir(parents=True, exist_ok=True)

    asignaciones = _split_estratificado(df, proporciones, semilla)

    stats        = {"train": 0, "val": 0, "test": 0, "sin_mascara": 0, "mascara_vacia": 0}
    filas_salida = []

    print(f"\n[convert] Procesando {len(df)} imágenes...")
    for idx, fila in tqdm(df.iterrows(), total=len(df), unit="img"):
        nombre_archivo = fila["filename"]
        stem           = Path(nombre_archivo).stem
        split          = asignaciones.get(idx, "train")

        ruta_img     = dir_imagenes / nombre_archivo
        ruta_mascara = dir_mascaras / nombre_archivo

        if not ruta_img.exists() or not ruta_mascara.exists():
            stats["sin_mascara"] += 1
            continue

        mascara  = cv2.imread(str(ruta_mascara), cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread(str(ruta_img))

        if mascara is None or img_orig is None:
            stats["sin_mascara"] += 1
            continue

        alto_img, ancho_img = img_orig.shape[:2]

        bbox = extraer_bbox_normalizado(mascara, ancho_img, alto_img)
        if bbox is None:
            stats["mascara_vacia"] += 1
            continue

        keypoints = extraer_keypoints_normalizados(mascara, ancho_img, alto_img)
        linea     = construir_linea_yolo(bbox, keypoints)

        (dir_salida / "labels" / split / f"{stem}.txt").write_text(
            linea + "\n", encoding="utf-8"
        )
        shutil.copy2(str(ruta_img), str(dir_salida / "images" / split / nombre_archivo))

        stats[split] += 1
        filas_salida.append({
            "filename":    nombre_archivo,
            "split":       split,
            "clear":       fila["clear"],
            "blurry":      fila["blurry"],
            "dial_stained":fila["dial_stained"],
            "soil_covered":fila["soil_covered"],
            "dark":        fila["dark"],
            "reflective":  fila["reflective"],
            "six_digit":   fila["six_digit"],
        })

    _generar_yaml(dir_salida)
    _generar_stats_csv(dir_salida, filas_salida)

    print(f"\n[convert] Conversión completada:")
    print(f"  train        : {stats['train']:>6} imágenes")
    print(f"  val          : {stats['val']:>6} imágenes")
    print(f"  test         : {stats['test']:>6} imágenes")
    print(f"  sin máscara  : {stats['sin_mascara']:>6} (saltadas)")
    print(f"  máscara vacía: {stats['mascara_vacia']:>6} (saltadas)")
    print(f"\n  Salida: {dir_salida}")
    print(f"  YAML  : {dir_salida / 'water_meter.yaml'}")
    print(f"  Stats : {dir_salida / 'dataset_stats.csv'}")

    df_out = pd.DataFrame(filas_salida)
    for split in ["train", "val", "test"]:
        subset = df_out[df_out["split"] == split]
        if len(subset):
            _imprimir_stats_flags(subset, f"Split {split} ({len(subset)} imgs)")


def _split_estratificado(
    df: pd.DataFrame,
    proporciones: tuple[float, float, float],
    semilla: int,
) -> dict[int, str]:
    """
    Divide el dataset garantizando balance de condiciones en cada split.

    Agrupa en dos grupos:
      - Grupo A: clear=1 (condiciones normales)
      - Grupo B: clear=0 (al menos una condición desafiante)

    Aplica las proporciones de forma independiente a cada grupo y combina.
    """
    random.seed(semilla)
    asignaciones = {}

    for grupo in [1, 0]:
        indices = df.index[df["clear"] == grupo].tolist()
        random.shuffle(indices)

        n       = len(indices)
        n_train = int(n * proporciones[0])
        n_val   = int(n * proporciones[1])

        for i in indices[:n_train]:
            asignaciones[i] = "train"
        for i in indices[n_train:n_train + n_val]:
            asignaciones[i] = "val"
        for i in indices[n_train + n_val:]:
            asignaciones[i] = "test"

    return asignaciones


def _imprimir_stats_flags(df: pd.DataFrame, titulo: str):
    print(f"\n  [{titulo}]")
    for col in CSV_COLUMNAS[1:]:
        if col in df.columns:
            n   = int(df[col].sum())
            pct = 100 * n / len(df) if len(df) else 0
            print(f"    {col:<15}: {n:>6} ({pct:.1f}%)")


def _generar_yaml(dir_salida: Path):
    contenido = {
        "path":      str(dir_salida.resolve()),
        "train":     "images/train",
        "val":       "images/val",
        "test":      "images/test",
        "nc":        1,
        "names":     ["meter_dial"],
        "kpt_shape": [4, 2],
    }
    with open(dir_salida / "water_meter.yaml", "w", encoding="utf-8") as f:
        yaml.dump(contenido, f, default_flow_style=False, allow_unicode=True)


def _generar_stats_csv(dir_salida: Path, filas: list[dict]):
    if filas:
        pd.DataFrame(filas).to_csv(
            dir_salida / "dataset_stats.csv", index=False
        )


# ─────────────────────────────────────────────────────────────
# VERIFICACIÓN VISUAL
# ─────────────────────────────────────────────────────────────

def verificar_conversion(dir_salida: Path, n_muestras: int = 5):
    """
    Genera imágenes de diagnóstico para verificar la conversión.

    Dibuja sobre cada muestra:
      - Bounding box en verde
      - 4 keypoints con etiquetas AI/AD/BD/BI en colores distintos
      - Flags activos de esa imagen en el borde superior

    Las imágenes se guardan en dir_salida/verificacion/.
    """
    dir_verificacion = dir_salida / "verificacion"
    dir_verificacion.mkdir(exist_ok=True)

    ruta_stats = dir_salida / "dataset_stats.csv"
    df_stats   = pd.read_csv(ruta_stats) if ruta_stats.exists() else None

    imagenes = list((dir_salida / "images" / "train").glob("*.png"))
    if not imagenes:
        imagenes = list((dir_salida / "images" / "train").glob("*.jpg"))
    if not imagenes:
        print("[verify] No se encontraron imágenes en images/train/")
        return

    muestras = random.sample(imagenes, min(n_muestras, len(imagenes)))

    print(f"\n[verify] Generando {len(muestras)} imágenes de verificación...")
    for ruta_img in muestras:
        ruta_lbl = dir_salida / "labels" / "train" / f"{ruta_img.stem}.txt"
        if not ruta_lbl.exists():
            continue

        img = cv2.imread(str(ruta_img))
        if img is None:
            continue
        h, w = img.shape[:2]

        partes = ruta_lbl.read_text().strip().split()
        vals   = [float(v) if i > 0 else int(v) for i, v in enumerate(partes)]

        # Bounding box
        cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Keypoints: desde índice 5, pares x,y sin visibilidad
        colores_kp = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        labels_kp  = ["AI", "AD", "BD", "BI"]
        if len(vals) >= 5 + 4 * 2:
            for i in range(4):
                base = 5 + i * 2
                kx   = int(vals[base]     * w)
                ky   = int(vals[base + 1] * h)
                cv2.circle(img, (kx, ky), 6, colores_kp[i], -1)
                cv2.putText(img, labels_kp[i], (kx + 8, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores_kp[i], 1)

        # Flags activos
        if df_stats is not None:
            fila = df_stats[df_stats["filename"] == ruta_img.name]
            if not fila.empty:
                activos = [
                    col for col in CSV_COLUMNAS[1:]
                    if col in fila.columns and int(fila.iloc[0][col]) == 1
                ]
                texto = " | ".join(activos) if activos else "sin flags"
                cv2.putText(img, texto, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, texto, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imwrite(str(dir_verificacion / f"check_{ruta_img.name}"), img)

    print(f"[verify] Guardadas en: {dir_verificacion}")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convierte el dataset de máscaras PNG + CSV a formato YOLO keypoints"
    )
    parser.add_argument("--dataset",  required=True,
                        help="Raíz del dataset original")
    parser.add_argument("--output",   required=True,
                        help="Destino de la estructura YOLO")
    parser.add_argument("--split",    nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--csv",      default="labels.csv")
    parser.add_argument("--verify",   action="store_true",
                        help="Generar imágenes de verificación")
    parser.add_argument("--n-verify", type=int, default=5)
    args = parser.parse_args()

    convertir_dataset(
        dir_dataset=Path(args.dataset),
        dir_salida=Path(args.output),
        proporciones=tuple(args.split),
        semilla=args.seed,
        nombre_csv=args.csv,
    )

    if args.verify:
        verificar_conversion(Path(args.output), n_muestras=args.n_verify)