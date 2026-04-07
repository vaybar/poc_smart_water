"""
segmentar_digitos.py

Dado un recorte rectangular del dial de un medidor, segmenta los X dígitos
individuales usando proyección vertical de píxeles oscuros.

Genera imágenes de diagnóstico para verificar que la segmentación es correcta
antes de pasar al entrenamiento del clasificador.

Uso:
    # Una imagen — muestra la figura en pantalla
    python segmentar_digitos.py --imagen recorte.png

    # Una imagen — guarda el diagnóstico como PNG
    python segmentar_digitos.py --imagen recorte.png --guardar

    # Carpeta completa (genera diagnósticos para todas)
    python segmentar_digitos.py --carpeta /ruta/a/imagenes --salida /ruta/salida

    # Con número esperado distinto a X
    python segmentar_digitos.py --imagen recorte.png --n-digitos 5

Dependencias:
    pip install opencv-python-headless numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# PREPROCESO
# ─────────────────────────────────────────────────────────────

def preprocesar(
    imagen: np.ndarray,
    roi_v: tuple[float, float] = (0.15, 0.85),
) -> np.ndarray:
    """
    Convierte la imagen a binaria para facilitar la proyección.

    Pasos:
      1. Recorte de ROI vertical — excluye franjas horizontales donde
         suelen estar los separadores físicos del display (bordes superior
         e inferior del dial). El recorte se aplica SOLO para calcular la
         proyección; los recortes de dígitos finales usan la imagen completa.
      2. Escala de grises
      3. CLAHE — mejora contraste local
      4. Binarización adaptativa — dígitos en blanco, fondo en negro
      5. Cierre morfológico — une trazos fragmentados

    Args:
        imagen : imagen del recorte del dial (BGR o gris)
        roi_v  : (frac_superior, frac_inferior) fracción del alto a conservar.
                 Por defecto (0.15, 0.85) descarta el 15% superior e inferior,
                 donde suelen aparecer líneas separadoras del display.

    Returns:
        imagen binaria con dígitos en blanco y fondo en negro,
        recortada verticalmente según roi_v.
    """
    if imagen.ndim == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # Recorte vertical para excluir separadores horizontales del display
    h = gris.shape[0]
    y0 = int(h * roi_v[0])
    y1 = int(h * roi_v[1])
    gris = gris[y0:y1, :]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gris  = clahe.apply(gris)

    binaria = cv2.adaptiveThreshold(
        gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=8,
    )

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    return binaria


# ─────────────────────────────────────────────────────────────
# PROYECCIÓN VERTICAL Y DETECCIÓN DE VALLES
# ─────────────────────────────────────────────────────────────

def calcular_proyeccion(binaria: np.ndarray) -> np.ndarray:
    """
    Suma los píxeles blancos por columna.

    El resultado es un vector de longitud = ancho de la imagen.
    Las columnas con valor alto tienen muchos píxeles de dígito.
    Las columnas con valor bajo (valles) son los separadores entre dígitos.
    """
    return binaria.sum(axis=0).astype(np.float32)


def encontrar_cortes(
    proyeccion: np.ndarray,
    n_digitos: int = 6,
    margen: int = 2,
) -> list[tuple[int, int]]:
    """
    Encuentra los n_digitos segmentos en la proyección.

    Estrategia:
      1. Suavizar la proyección
      2. Filtrar segmentos demasiado angostos para ser dígitos reales
         (separadores físicos del display, ruido de borde)
      3. Búsqueda adaptativa de umbral para encontrar exactamente n_digitos
      4. Fusionar o dividir si hay más o menos segmentos que los esperados
      5. Fallback a división uniforme si todo falla

    El filtro por ancho mínimo (paso 2) es la mejora clave respecto a la
    versión anterior: un separador físico del display tiene típicamente
    5-15px de ancho, mientras que un dígito ocupa ancho/n_digitos píxeles.
    Filtrar por debajo de ancho/(n_digitos*3) elimina los separadores sin
    afectar ningún dígito real.

    Returns:
        lista de (x_inicio, x_fin) para cada dígito, en píxeles
    """
    ancho        = len(proyeccion)
    ancho_minimo = max(ancho // (n_digitos * 3), 5)   # mín. absoluto: 5px

    # Suavizado para eliminar ruido de pequeños artefactos
    suavizada = np.convolve(
        proyeccion,
        np.hanning(max(ancho // 20, 5)),
        mode="same"
    )

    # Búsqueda adaptativa de umbral con filtro por ancho mínimo
    mejor_segmentos = None
    mejor_diff      = float("inf")

    for pct in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]:
        umbral  = suavizada.max() * pct
        activas = (suavizada > umbral).astype(np.uint8)
        segs    = _columnas_a_segmentos(activas)

        # Filtrar separadores: descartar segmentos más angostos que el mínimo
        segs = [(x0, x1) for x0, x1 in segs if (x1 - x0) >= ancho_minimo]

        diff = abs(len(segs) - n_digitos)
        if diff < mejor_diff or (diff == mejor_diff and len(segs) >= n_digitos):
            mejor_diff      = diff
            mejor_segmentos = segs
        if diff == 0:
            break

    segmentos = mejor_segmentos if mejor_segmentos else []

    if len(segmentos) == n_digitos:
        cortes = [
            (max(0, x0 - margen), min(ancho - 1, x1 + margen))
            for x0, x1 in segmentos
        ]
    elif len(segmentos) > n_digitos:
        cortes = _fusionar_segmentos(segmentos, n_digitos, ancho, margen)
    else:
        cortes = _division_uniforme(ancho, n_digitos)

    return cortes


def _columnas_a_segmentos(activas: np.ndarray) -> list[tuple[int, int]]:
    """Convierte un vector binario en lista de rangos (inicio, fin)."""
    segmentos = []
    en_segmento = False
    inicio = 0

    for i, v in enumerate(activas):
        if v and not en_segmento:
            inicio = i
            en_segmento = True
        elif not v and en_segmento:
            segmentos.append((inicio, i - 1))
            en_segmento = False

    if en_segmento:
        segmentos.append((inicio, len(activas) - 1))

    return segmentos


def _fusionar_segmentos(
    segmentos: list[tuple[int, int]],
    n_objetivo: int,
    ancho: int,
    margen: int,
) -> list[tuple[int, int]]:
    """
    Fusiona segmentos cercanos hasta llegar a n_objetivo.
    Fusiona primero los pares con menor distancia entre ellos.
    """
    segs = list(segmentos)

    while len(segs) > n_objetivo:
        # Calcular distancia entre segmentos consecutivos
        distancias = [segs[i+1][0] - segs[i][1] for i in range(len(segs) - 1)]
        idx_min    = int(np.argmin(distancias))
        # Fusionar el par más cercano
        fusionado  = (segs[idx_min][0], segs[idx_min + 1][1])
        segs       = segs[:idx_min] + [fusionado] + segs[idx_min + 2:]

    return [
        (max(0, x0 - margen), min(ancho - 1, x1 + margen))
        for x0, x1 in segs
    ]


def _division_uniforme(ancho: int, n: int) -> list[tuple[int, int]]:
    """Fallback: divide el ancho en n partes iguales."""
    paso = ancho // n
    return [(i * paso, (i + 1) * paso - 1) for i in range(n)]


# ─────────────────────────────────────────────────────────────
# SEGMENTACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def segmentar(
    imagen: np.ndarray,
    n_digitos: int = 6,
    alto_salida: int = 64,
    ancho_salida: int = 32,
    overlap: int = 15,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray]:
    """
    Segmenta una imagen de dial en n_digitos recortes individuales.

    Args:
        imagen      : imagen del recorte del dial (BGR o gris)
        n_digitos   : cantidad de dígitos esperados (default 6)
        alto_salida : alto en píxeles de cada dígito recortado
        ancho_salida: ancho en píxeles de cada dígito recortado
        overlap     : píxeles extra que se agregan a cada lado del recorte
                      para capturar dígitos que la proyección cortó justo
                      en el borde. El clasificador tolera que el recorte
                      incluya parte del dígito vecino — lo que no tolera
                      es que le falte parte del dígito propio.
                      Valor recomendado: 8-12px. (default 8)

    Returns:
        digitos   : lista de n_digitos imágenes en escala de grises
        cortes    : lista de (x0, x1) de los segmentos base (sin overlap)
        proyeccion: vector de proyección vertical (para diagnóstico)
    """
    binaria    = preprocesar(imagen)
    proyeccion = calcular_proyeccion(binaria)
    cortes     = encontrar_cortes(proyeccion, n_digitos)

    # Convertir a gris para los recortes de salida
    if imagen.ndim == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    ancho_img = gris.shape[1]

    digitos = []
    for x0, x1 in cortes:
        # Expandir el recorte con overlap hacia ambos lados
        # sin salir de los límites de la imagen
        x0_exp = max(0,         x0 - overlap)
        x1_exp = min(ancho_img, x1 + overlap + 1)

        recorte = gris[:, x0_exp:x1_exp]

        # Redimensionar al tamaño estándar para el clasificador
        recorte = cv2.resize(
            recorte, (ancho_salida, alto_salida),
            interpolation=cv2.INTER_AREA
        )
        digitos.append(recorte)

    return digitos, cortes, proyeccion


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO VISUAL
# ─────────────────────────────────────────────────────────────

def generar_diagnostico(
    imagen: np.ndarray,
    digitos: list[np.ndarray],
    cortes: list[tuple[int, int]],
    proyeccion: np.ndarray,
    titulo: str = "",
    guardar_en: Path | None = None,
) -> None:
    """
    Genera una figura de diagnóstico con 4 paneles:

      1. Imagen original con líneas de corte superpuestas
      2. Proyección vertical con los cortes marcados
      3. Imagen binarizada preprocesada
      4. Los 6 dígitos recortados en fila

    Si guardar_en es None, muestra la figura en pantalla.
    Si guardar_en es una ruta, guarda la figura como PNG.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(titulo or "Diagnóstico de segmentación", fontsize=13)

    # ── Panel 1: imagen original con cortes ──────────────────
    ax1 = fig.add_subplot(3, 1, 1)
    if imagen.ndim == 3:
        ax1.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(imagen, cmap="gray")
    ax1.set_title("Imagen original con líneas de corte", fontsize=10)
    ax1.axis("off")

    colores = plt.cm.tab10.colors
    for i, (x0, x1) in enumerate(cortes):
        color = colores[i % len(colores)]
        ax1.axvline(x=x0, color=color, linewidth=1.5, linestyle="--", alpha=0.8)
        ax1.axvline(x=x1, color=color, linewidth=1.5, linestyle="-",  alpha=0.8)
        centro = (x0 + x1) / 2
        ax1.text(centro, 5, str(i), ha="center", va="top",
                 color=color, fontsize=9, fontweight="bold")

    # ── Panel 2: proyección vertical ─────────────────────────
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(proyeccion, color="steelblue", linewidth=1)
    ax2.fill_between(range(len(proyeccion)), proyeccion, alpha=0.3, color="steelblue")
    ax2.set_title("Proyección vertical (suma de píxeles por columna)", fontsize=10)
    ax2.set_xlim(0, len(proyeccion) - 1)
    ax2.set_ylabel("píxeles")

    for i, (x0, x1) in enumerate(cortes):
        color = colores[i % len(colores)]
        ax2.axvspan(x0, x1, alpha=0.15, color=color)
        ax2.axvline(x=x0, color=color, linewidth=1, linestyle="--")
        ax2.axvline(x=x1, color=color, linewidth=1)

    # ── Panel 3: dígitos recortados ───────────────────────────
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.axis("off")
    ax3.set_title("Dígitos segmentados", fontsize=10)

    if digitos:
        # Concatenar los dígitos horizontalmente con separador
        sep    = np.ones((digitos[0].shape[0], 4), dtype=np.uint8) * 180
        tira   = digitos[0]
        for d in digitos[1:]:
            tira = np.hstack([tira, sep, d])
        ax3.imshow(tira, cmap="gray", aspect="auto")

    plt.tight_layout()

    if guardar_en:
        plt.savefig(str(guardar_en), dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────
# PROCESAMIENTO DE CARPETA COMPLETA
# ─────────────────────────────────────────────────────────────

def procesar_carpeta(
    dir_entrada: Path,
    dir_salida: Path,
    n_digitos: int = 6,
    extensiones: tuple = (".png", ".jpg", ".jpeg"),
):
    """
    Procesa todas las imágenes de una carpeta y guarda:
      - dir_salida/diagnosticos/  → figuras PNG de diagnóstico
      - dir_salida/digitos/img_N/ → los 6 recortes individuales por imagen

    Imprime un resumen con cuántas imágenes usaron el método de
    proyección vs. el fallback de división uniforme.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_diag   = dir_salida / "diagnosticos"
    dir_dig    = dir_salida / "digitos"
    dir_diag.mkdir(exist_ok=True)
    dir_dig.mkdir(exist_ok=True)

    imagenes = [
        p for p in sorted(dir_entrada.iterdir())
        if p.suffix.lower() in extensiones
    ]

    if not imagenes:
        print(f"No se encontraron imágenes en {dir_entrada}")
        return

    print(f"Procesando {len(imagenes)} imágenes...")
    stats = {"proyeccion": 0, "fallback": 0, "error": 0}

    for ruta in imagenes:
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"  [!] No se pudo leer: {ruta.name}")
            stats["error"] += 1
            continue

        try:
            digitos, cortes, proyeccion = segmentar(imagen, n_digitos)

            # Determinar si usó proyección o fallback (heurística:
            # fallback da cortes perfectamente uniformes)
            anchos = [x1 - x0 for x0, x1 in cortes]
            es_uniforme = (max(anchos) - min(anchos)) < 3
            if es_uniforme:
                stats["fallback"] += 1
            else:
                stats["proyeccion"] += 1

            # Guardar diagnóstico
            generar_diagnostico(
                imagen, digitos, cortes, proyeccion,
                titulo=ruta.name,
                guardar_en=dir_diag / f"diag_{ruta.stem}.png",
            )

            # Guardar dígitos individuales
            dir_img = dir_dig / ruta.stem
            dir_img.mkdir(exist_ok=True)
            for i, d in enumerate(digitos):
                cv2.imwrite(str(dir_img / f"digito_{i}.png"), d)

        except Exception as e:
            print(f"  [!] Error en {ruta.name}: {e}")
            stats["error"] += 1

    print(f"\nResultados:")
    print(f"  Proyección vertical : {stats['proyeccion']:>4} imágenes")
    print(f"  Fallback uniforme   : {stats['fallback']:>4} imágenes")
    print(f"  Errores             : {stats['error']:>4} imágenes")
    print(f"\n  Diagnósticos → {dir_diag}")
    print(f"  Dígitos      → {dir_dig}")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmenta los dígitos de un recorte de medidor de agua"
    )

    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagen",   help="Ruta a una imagen individual")
    grupo.add_argument("--carpeta",  help="Carpeta con múltiples imágenes")

    parser.add_argument(
        "--salida", default="salida_segmentacion",
        help="Carpeta de salida para modo --carpeta (default: salida_segmentacion)"
    )
    parser.add_argument(
        "--n-digitos", type=int, default=6,
        help="Cantidad de dígitos esperados (default: 6)"
    )
    parser.add_argument(
        "--guardar", action="store_true",
        help="En modo --imagen, guardar el diagnóstico en vez de mostrarlo"
    )
    args = parser.parse_args()

    if args.imagen:
        ruta  = Path(args.imagen)
        img   = cv2.imread(str(ruta))
        if img is None:
            print(f"Error: no se pudo leer {ruta}")
            exit(1)

        digitos, cortes, proyeccion = segmentar(img, args.n_digitos)

        print(f"Imagen : {ruta.name}  ({img.shape[1]}×{img.shape[0]} px)")
        print(f"Cortes :")
        for i, (x0, x1) in enumerate(cortes):
            print(f"  dígito {i}: columnas {x0}–{x1}  ({x1-x0+1} px de ancho)")

        guardar_en = None
        if args.guardar:
            guardar_en = ruta.parent / f"diag_{ruta.stem}.png"
            print(f"\nDiagnóstico guardado en: {guardar_en}")

        generar_diagnostico(
            img, digitos, cortes, proyeccion,
            titulo=ruta.name,
            guardar_en=guardar_en,
        )

    else:
        procesar_carpeta(
            dir_entrada=Path(args.carpeta),
            dir_salida=Path(args.salida),
            n_digitos=args.n_digitos,
        )