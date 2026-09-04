"""
preparar_dataset_mobilenet.py

Procesa el dataset de recortes de dígitos, segmenta cada imagen en 6 dígitos
individuales y arma la estructura de carpetas que necesita MobileNetV3 para
entrenar con ImageFolder de PyTorch o flow_from_directory de Keras.

Estructura esperada del dataset original:
    dataset_digitos/
    ├── images/          *.png  (recortes del dial ya extraídos)
    └── labels.csv       sin header:
                           [0] filename
                           [1] numero      ej. "018273"
                           [2] reversed    1 = dígitos en orden inverso
                           [3] half_word   1 = algún dígito en transición
                           [4] six_digit   1 = medidor de 6 dígitos

Criterio de filtrado:
    half_word=0, reversed=0  →  train/val  (número del CSV tal cual)
    half_word=0, reversed=1  →  train/val  (número invertido: "018273" → "372810")
    half_word=1              →  test        (imagen completa sin segmentar)

Estructura de salida:
    dataset_mobilenet/
    ├── train/
    │   ├── 0/   recortes individuales de dígitos clase 0
    │   ├── 1/
    │   ...
    │   └── 9/
    ├── val/
    │   ├── 0/
    │   ...
    │   └── 9/
    └── test/
        ├── imagenes/        imágenes completas del dial (sin segmentar)
        └── labels_test.csv  filename, numero, reversed, half_word

Uso:
    python preparar_dataset_mobilenet.py \\
        --dataset /ruta/dataset_digitos \\
        --output  /ruta/dataset_mobilenet \\
        --split   0.85 0.15

Dependencias:
    pip install opencv-python-headless numpy pandas tqdm
"""

import cv2
import shutil
import pandas as pd
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from segmentar_digitos import segmentar


# ─────────────────────────────────────────────────────────────
# COLUMNAS DEL CSV
# ─────────────────────────────────────────────────────────────

CSV_COLUMNAS = [
    "filename",  # 0
    "numero",    # 1  ej. "018273"
    "reversed",  # 2
    "half_word", # 3
    "six_digit", # 4
]


# ─────────────────────────────────────────────────────────────
# CARGA DEL CSV
# ─────────────────────────────────────────────────────────────

def cargar_csv(ruta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv, header=None, dtype=str)

    if df.shape[1] != len(CSV_COLUMNAS):
        raise ValueError(
            f"El CSV tiene {df.shape[1]} columnas, se esperaban {len(CSV_COLUMNAS)}.\n"
            f"Columnas esperadas: {CSV_COLUMNAS}\n"
            f"Primeras filas:\n{df.head(3).to_string()}"
        )

    df.columns = CSV_COLUMNAS

    for col in ["reversed", "half_word", "six_digit"]:
        df[col] = df[col].fillna("0").astype(int)

    df["numero"] = df["numero"].str.strip()

    return df


# ─────────────────────────────────────────────────────────────
# PREPARACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def preparar_dataset(
    dir_dataset: Path,
    dir_salida: Path,
    proporciones: tuple[float, float] = (0.85, 0.15),
    semilla: int = 42,
    nombre_csv: str = "labels.csv",
    n_digitos: int = 6,
    alto_digito: int = 64,
    ancho_digito: int = 32,
    prefijo: str = "ds",   # ← nuevo parámetro,
):
    """
    Pipeline completo: CSV → segmentación → carpetas por clase.

    Las imágenes con reversed=1 se incluyen en train/val invirtiendo
    el número del CSV para que las etiquetas sean correctas.
    Las imágenes con half_word=1 van a test como imágenes completas
    porque la ambigüedad no se puede resolver automáticamente.
    """
    assert abs(sum(proporciones) - 1.0) < 1e-6, \
        "Las proporciones train+val deben sumar 1.0"

    dir_imagenes = dir_dataset / "images"
    ruta_csv     = dir_dataset / nombre_csv

    if not dir_imagenes.exists():
        raise FileNotFoundError(f"No se encontró images/ en {dir_dataset}")
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró {nombre_csv} en {dir_dataset}")

    df = cargar_csv(ruta_csv)
    print(f"[prep] CSV cargado: {len(df)} filas")

    # ── Separar grupos ────────────────────────────────────────
    # Train/val: half_word=0 (reversed must be 0)
    # Test:      half_word=1 (reversed is 1)
    df_trainval = df[(df["half_word"] == 0) & (df["reversed"] == 0)].copy()
    df_test = df[(df["half_word"] == 1) | (df["reversed"] == 1)].copy()

    n_normal   = (df_trainval["reversed"] == 0).sum()
    n_invertido = (df_trainval["reversed"] == 1).sum()
    print(f"[prep] Train/val: {len(df_trainval)} imágenes")
    print(f"         normal   : {n_normal}")
    print(f"         invertido: {n_invertido}  (número se leerá al revés)")
    print(f"[prep] Test      : {len(df_test)} imágenes (half_word=1)")

    # ── Crear carpetas ────────────────────────────────────────
    for split in ["train", "val"]:
        for clase in range(10):
            (dir_salida / split / str(clase)).mkdir(parents=True, exist_ok=True)

    # Carpeta de test: imágenes completas + CSV, sin subcarpetas por clase
    (dir_salida / "test" / "imagenes").mkdir(parents=True, exist_ok=True)

    # ── Split estratificado train/val ─────────────────────────
    asignaciones = _split_estratificado(df_trainval, proporciones, semilla)

    # ── Estadísticas ──────────────────────────────────────────
    stats = {
        "train": 0, "val": 0, "test": 0,
        "sin_imagen": 0, "error_segmentacion": 0, "numero_invalido": 0,
    }
    conteo_clases = {s: {str(c): 0 for c in range(10)} for s in ["train", "val"]}

    # ── Procesar train/val ────────────────────────────────────
    print(f"\n[prep] Segmentando imágenes para train/val...")
    for idx, fila in tqdm(df_trainval.iterrows(), total=len(df_trainval), unit="img"):
        _procesar_trainval(
            fila, dir_imagenes, dir_salida,
            asignaciones[idx], n_digitos, alto_digito, ancho_digito,
            stats, conteo_clases,prefijo
        )

    # ── Procesar test ─────────────────────────────────────────
    print(f"\n[prep] Copiando imágenes de test (half_word=1)...")
    filas_test = []
    for _, fila in tqdm(df_test.iterrows(), total=len(df_test), unit="img"):
        ruta_img = dir_imagenes / fila["filename"]
        if not ruta_img.exists():
            stats["sin_imagen"] += 1
            continue

        dst = dir_salida / "test" / "imagenes" / fila["filename"]
        shutil.copy2(str(ruta_img), str(dst))
        filas_test.append({
            "filename": fila["filename"],
            "numero":   fila["numero"],
            "reversed": fila["reversed"],
            "half_word": fila["half_word"],
        })
        stats["test"] += 1

    # Guardar CSV de test
    if filas_test:
        pd.DataFrame(filas_test).to_csv(
            dir_salida / "test" / "labels_test.csv",
            index=False,
        )

    # ── Resumen ───────────────────────────────────────────────
    _imprimir_resumen(stats, conteo_clases, dir_salida)


def _procesar_trainval(
    fila: pd.Series,
    dir_imagenes: Path,
    dir_salida: Path,
    split: str,
    n_digitos: int,
    alto: int,
    ancho: int,
    stats: dict,
    conteo_clases: dict,
    prefijo=str,
) -> None:
    """
    Segmenta una imagen y guarda cada dígito en su carpeta de clase.

    Si reversed=1, invierte el string del número antes de asignar
    etiquetas para que coincidan con el orden visual real de los dígitos.
    """
    nombre = fila["filename"]
    numero = fila["numero"]

    if not numero.isdigit() or len(numero) != n_digitos:
        stats["numero_invalido"] += 1
        return

    # Invertir el número si la imagen está al revés
    if fila["reversed"] == 1:
        numero = numero[::-1]

    ruta_img = dir_imagenes / nombre
    if not ruta_img.exists():
        stats["sin_imagen"] += 1
        return

    imagen = cv2.imread(str(ruta_img))
    if imagen is None:
        stats["sin_imagen"] += 1
        return

    try:
        digitos, _, _ = segmentar(imagen, n_digitos, alto, ancho)
    except Exception:
        stats["error_segmentacion"] += 1
        return

    stem = Path(nombre).stem
    for i, (recorte, clase) in enumerate(zip(digitos, numero)):
        dst = dir_salida / split / clase / f"{prefijo}_{stem}_pos{i}.png"
        cv2.imwrite(str(dst), recorte)
        conteo_clases[split][clase] += 1

    stats[split] += 1


def _split_estratificado(
    df: pd.DataFrame,
    proporciones: tuple[float, float],
    semilla: int,
) -> dict[int, str]:
    """
    Divide train/val garantizando representación de todas las clases
    en ambos splits, agrupando por primer dígito del número.
    """
    random.seed(semilla)
    asignaciones = {}

    # Primer dígito del número (ya invertido si corresponde, pero acá
    # solo necesitamos distribuir uniformemente — el primer dígito del
    # CSV es suficiente como criterio de estratificación)
    primer_digito = df["numero"].str[0].fillna("0")

    for digito in primer_digito.unique():
        indices = df.index[primer_digito == digito].tolist()
        random.shuffle(indices)
        n_train = int(len(indices) * proporciones[0])
        for i in indices[:n_train]:
            asignaciones[i] = "train"
        for i in indices[n_train:]:
            asignaciones[i] = "val"

    return asignaciones


def _imprimir_resumen(stats: dict, conteo_clases: dict, dir_salida: Path):
    print(f"\n[prep] Dataset preparado:")
    print(f"  Imágenes procesadas:")
    print(f"    train             : {stats['train']:>6}")
    print(f"    val               : {stats['val']:>6}")
    print(f"    test              : {stats['test']:>6} (imágenes completas)")
    print(f"    sin imagen        : {stats['sin_imagen']:>6} (saltadas)")
    print(f"    error segmentación: {stats['error_segmentacion']:>6} (saltadas)")
    print(f"    número inválido   : {stats['numero_invalido']:>6} (saltadas)")

    print(f"\n  Dígitos por clase (train):")
    total = sum(conteo_clases["train"].values())
    for clase in sorted(conteo_clases["train"].keys()):
        n   = conteo_clases["train"][clase]
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"    clase {clase}: {n:>6} ({pct:4.1f}%)  {bar}")

    print(f"\n  Salida: {dir_salida}")
    print(f"\n  Para usar en PyTorch:")
    print(f"    train_ds = ImageFolder('{dir_salida}/train')")
    print(f"    val_ds   = ImageFolder('{dir_salida}/val')")
    print(f"  Para usar en Keras:")
    print(f"    flow_from_directory('{dir_salida}/train')")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara el dataset de dígitos para MobileNetV3"
    )
    parser.add_argument("--dataset",   required=True,
                        help="Raíz del dataset (contiene images/ y labels.csv)")
    parser.add_argument("--output",    required=True,
                        help="Destino de la estructura MobileNetV3")
    parser.add_argument("--split",     nargs=2, type=float, default=[0.85, 0.15],
                        metavar=("TRAIN", "VAL"),
                        help="Proporciones train/val (default: 0.85 0.15)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--csv",       default="labels.csv")
    parser.add_argument("--n-digitos", type=int, default=6)
    parser.add_argument("--alto",      type=int, default=64,
                        help="Alto del recorte en píxeles (default: 64)")
    parser.add_argument("--ancho",     type=int, default=32,
                        help="Ancho del recorte en píxeles (default: 32)")
    parser.add_argument("--prefijo", default="ds",
                        help="Prefijo para evitar colisión de nombres entre datasets")
    args = parser.parse_args()

    preparar_dataset(
        dir_dataset=Path(args.dataset),
        dir_salida=Path(args.output),
        proporciones=tuple(args.split),
        semilla=args.seed,
        nombre_csv=args.csv,
        n_digitos=args.n_digitos,
        alto_digito=args.alto,
        ancho_digito=args.ancho,
        prefijo=args.prefijo,
    )