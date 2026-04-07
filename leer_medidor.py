"""
leer_medidor.py

Pipeline completo de lectura de medidor de agua optimizado para Raspberry Pi.

Etapas:
  1. Detección del dial     — YOLOv8n TFLite
  2. Rectificación          — homografía con OpenCV (water_meter_pipeline.py)
  3. Segmentación           — proyección vertical (segmentar_digitos.py)
  4. Clasificación          — MobileNetV3-Small TFLite × N dígitos
  5. Validación             — plausibilidad numérica

Uso básico:
    python leer_medidor.py --imagen foto.jpg

Con parámetros explícitos:
    python leer_medidor.py \\
        --imagen      foto.jpg \\
        --modelo-yolo best_yolo_int8.tflite \\
        --modelo-ocr  best_mobilenet_int8.tflite \\
        --n-digitos   6 \\
        --threads     4 \\
        --debug

Dependencias:
    pip install opencv-python-headless numpy

    En Raspberry Pi:
    pip install tflite-runtime
    (no instalar tensorflow completo — ocupa ~500MB innecesarios)
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Módulos del proyecto
from water_meter_pipeline import RectificadorPerspectiva, DeteccionDial
from segmentar_digitos import segmentar


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO
# ─────────────────────────────────────────────────────────────

MODELO_YOLO_DEFAULT = "best_yolo_int8.tflite"
MODELO_OCR_DEFAULT  = "best_mobilenet_int8.tflite"
N_DIGITOS_DEFAULT   = 6
CONF_YOLO_MIN       = 0.45   # umbral de confianza del detector
CONF_OCR_MIN        = 0.60   # umbral de confianza del clasificador
THREADS_DEFAULT     = 4      # ajustar según el modelo de RPi:
                             #   RPi Zero 2W → 4
                             #   RPi 3/4/5   → 4 (todos tienen 4 cores)


# ─────────────────────────────────────────────────────────────
# TIPOS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class ResultadoDigito:
    posicion:  int
    clase:     int           # 0-9
    confianza: float
    valido:    bool          # confianza >= umbral


@dataclass
class ResultadoLectura:
    lectura:              str | None       # ej. "018273"
    exitoso:              bool
    motivo_falla:         str | None
    confianza_deteccion:  float
    confianza_ocr_media:  float
    digitos:              list[ResultadoDigito] = field(default_factory=list)
    tiempos_ms:           dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# CARGA DEL INTÉRPRETE TFLITE
# ─────────────────────────────────────────────────────────────

def cargar_interprete(ruta_modelo: str, n_threads: int):
    """
    Carga un modelo TFLite con el número de threads indicado.

    En Raspberry Pi, más threads no siempre es más rápido — 4 es
    el valor óptimo para todos los modelos RPi actuales porque todos
    tienen exactamente 4 cores físicos. Con más threads el overhead
    de sincronización supera la ganancia de paralelismo.

    Intenta primero tflite-runtime (liviano, recomendado en RPi)
    y cae a tensorflow como fallback para desarrollo en PC.
    """
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(
            model_path=ruta_modelo,
            num_threads=n_threads,
        )
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(
            model_path=ruta_modelo,
            num_threads=n_threads,
        )

    interp.allocate_tensors()
    return interp


# ─────────────────────────────────────────────────────────────
# ETAPA 1: DETECCIÓN CON YOLO
# ─────────────────────────────────────────────────────────────

class DetectorYOLO:
    """
    Envuelve el modelo YOLOv8n TFLite para detectar el dial del medidor.

    El modelo devuelve para cada predicción:
      [cx, cy, w, h, confianza, x1, y1, x2, y2, x3, y3, x4, y4]
      (bbox normalizado + confianza + 4 keypoints normalizados)

    Si el modelo no tiene head de keypoints devuelve solo
      [cx, cy, w, h, confianza]
    y la rectificación usará el bbox como aproximación.
    """

    def __init__(self, ruta_modelo: str, n_threads: int, conf_min: float):
        self.conf_min  = conf_min
        self.interp    = cargar_interprete(ruta_modelo, n_threads)
        self.in_det    = self.interp.get_input_details()[0]
        self.out_det   = self.interp.get_output_details()[0]
        self.in_shape  = self.in_det["shape"]  # [1, H, W, 3]

    def detectar(self, imagen: np.ndarray) -> DeteccionDial | None:
        """
        Corre el modelo YOLO sobre la imagen y devuelve la mejor detección.

        Returns:
            DeteccionDial con bbox y keypoints normalizados, o None si
            no se detecta ningún dial con confianza suficiente.
        """
        h_orig, w_orig = imagen.shape[:2]
        h_mod  = self.in_shape[1]
        w_mod  = self.in_shape[2]

        # Preproceso: CLAHE + resize + normalizar
        img_proc = self._preprocesar(imagen, w_mod, h_mod)

        self.interp.set_tensor(self.in_det["index"], img_proc)
        self.interp.invoke()
        output = self.interp.get_tensor(self.out_det["index"])[0]

        # El output de YOLOv8 exportado a TFLite puede venir transpuesto:
        # formato correcto : [n_predicciones, n_valores]  ej. [2100, 13]
        # formato transpuesto: [n_valores, n_predicciones] ej. [13, 2100]
        # Si la primera dimensión es menor que la segunda y menor a 50
        # (ningún modelo razonable tiene menos de 50 predicciones),
        # es el formato transpuesto — corregir.
        if output.shape[0] < output.shape[1] and output.shape[0] < 50:
            output = output.T   # → [2100, 13]

        # Filtrar por confianza y tomar la mejor detección
        confs      = output[:, 4]
        mejor_idx  = int(np.argmax(confs))
        mejor_conf = float(confs[mejor_idx])

        if mejor_conf < self.conf_min:
            return None

        mejor = output[mejor_idx]

        # Coordenadas normalizadas o en píxeles según el modelo exportado
        if mejor[2] <= 1.0 and mejor[3] <= 1.0:
            scale_w, scale_h = 1.0, 1.0
        else:
            scale_w, scale_h = float(w_mod), float(h_mod)

        bbox = [
            float(mejor[0]) / scale_w,
            float(mejor[1]) / scale_h,
            float(mejor[2]) / scale_w,
            float(mejor[3]) / scale_h,
        ]

        # Keypoints — soporta formato sin visibilidad (8 valores)
        # y con visibilidad (12 valores: x, y, v por cada punto)
        keypoints = None
        resto     = len(mejor) - 5
        if resto == 8:
            kpts_raw = mejor[5:13].reshape(4, 2).copy()
            kpts_raw[:, 0] /= scale_w
            kpts_raw[:, 1] /= scale_h
            kpts = kpts_raw.tolist()
            if all(0.0 <= v <= 1.0 for pt in kpts for v in pt):
                keypoints = kpts
        elif resto == 12:
            kpts_raw = mejor[5:17].reshape(4, 3).copy()
            kpts_raw[:, 0] /= scale_w
            kpts_raw[:, 1] /= scale_h
            kpts = kpts_raw[:, :2].tolist()
            if all(0.0 <= v <= 1.0 for pt in kpts for v in pt):
                keypoints = kpts

        return DeteccionDial(
            bbox=bbox,
            keypoints=keypoints,
            confianza=mejor_conf,
            ancho_imagen=w_orig,
            alto_imagen=h_orig,
        )

    def _preprocesar(
        self, imagen: np.ndarray, w_mod: int, h_mod: int
    ) -> np.ndarray:
        """CLAHE + resize + normalización a [0, 1]."""
        lab   = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        img   = cv2.resize(img, (w_mod, h_mod))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img.astype(np.float32) / 255.0)[None, ...]


# ─────────────────────────────────────────────────────────────
# ETAPA 4: CLASIFICACIÓN CON MOBILENETV3
# ─────────────────────────────────────────────────────────────

class ClasificadorDigitos:
    """
    Envuelve el modelo MobileNetV3-Small TFLite para clasificar dígitos.

    Preproceso idéntico al entrenamiento:
      gris → RGB replicado → normalizar a [-1, 1] → resize 96×96
    """

    IMG_H = 96
    IMG_W = 96

    def __init__(self, ruta_modelo: str, n_threads: int, conf_min: float):
        self.conf_min = conf_min
        self.interp   = cargar_interprete(ruta_modelo, n_threads)
        self.in_det   = self.interp.get_input_details()[0]
        self.out_det  = self.interp.get_output_details()[0]

    def clasificar(self, recorte_gris: np.ndarray) -> tuple[int, float]:
        """
        Clasifica un recorte de dígito individual.

        Args:
            recorte_gris: imagen en escala de grises (cualquier tamaño)

        Returns:
            (clase, confianza) donde clase está en [0, 9]
        """
        tensor = self._preprocesar(recorte_gris)

        # Manejar entrada uint8 si el modelo está cuantizado
        if self.in_det["dtype"] == np.uint8:
            escala, zp = self.in_det["quantization"]
            tensor = (tensor / escala + zp).astype(np.uint8)

        self.interp.set_tensor(self.in_det["index"], tensor)
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out_det["index"])[0]

        # Desescalar salida si está cuantizada
        if self.out_det["dtype"] == np.uint8:
            escala, zp = self.out_det["quantization"]
            probs = (probs.astype(np.float32) - zp) * escala

        clase     = int(np.argmax(probs))
        confianza = float(probs[clase])
        return clase, confianza

    def clasificar_secuencia(
        self, recortes: list[np.ndarray]
    ) -> list[ResultadoDigito]:
        """
        Clasifica una lista de recortes y devuelve ResultadoDigito por cada uno.
        """
        resultados = []
        for i, recorte in enumerate(recortes):
            clase, conf = self.clasificar(recorte)
            resultados.append(ResultadoDigito(
                posicion=i,
                clase=clase,
                confianza=conf,
                valido=(conf >= self.conf_min),
            ))
        return resultados

    def _preprocesar(self, gris: np.ndarray) -> np.ndarray:
        """
        Preproceso idéntico al pipeline de entrenamiento en mobilenet_pipeline.py:
          1. Asegurar escala de grises
          2. Letterbox resize a 96×96
          3. Replicar a 3 canales RGB
          4. Normalizar a [-1, 1]
        """
        if gris.ndim == 3:
            gris = cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)

        # Letterbox
        h, w    = gris.shape[:2]
        scale   = min(self.IMG_W / w, self.IMG_H / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(gris, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas  = np.zeros((self.IMG_H, self.IMG_W), dtype=np.uint8)
        x_off   = (self.IMG_W - new_w) // 2
        y_off   = (self.IMG_H - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Gris → RGB replicado → normalizar
        rgb    = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        tensor = (rgb.astype(np.float32) / 127.5) - 1.0
        return tensor[None, ...]   # (1, 96, 96, 3)


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────

class LectorMedidor:
    """
    Pipeline completo: imagen → lectura numérica del medidor.

    Integra las cuatro etapas en un único objeto reutilizable.
    En Raspberry Pi, instanciar una sola vez y llamar a leer()
    repetidamente — la carga de modelos es el paso costoso.

    Parámetros:
        modelo_yolo  : ruta al TFLite del detector YOLOv8n
        modelo_ocr   : ruta al TFLite del clasificador MobileNetV3
        n_digitos    : cantidad de dígitos del medidor (default 6)
        conf_yolo    : umbral de confianza del detector (default 0.45)
        conf_ocr     : umbral de confianza del clasificador (default 0.60)
        n_threads    : threads para TFLite (default 4, óptimo en todos los RPi)
        debug        : si True guarda imágenes intermedias en debug/
    """

    def __init__(
        self,
        modelo_yolo:  str = MODELO_YOLO_DEFAULT,
        modelo_ocr:   str = MODELO_OCR_DEFAULT,
        n_digitos:    int = N_DIGITOS_DEFAULT,
        conf_yolo:    float = CONF_YOLO_MIN,
        conf_ocr:     float = CONF_OCR_MIN,
        n_threads:    int = THREADS_DEFAULT,
        debug:        bool = False,
    ):
        self.n_digitos = n_digitos
        self.debug     = debug

        print("[lector] Cargando modelos...")
        t0 = time.monotonic()

        self.detector     = DetectorYOLO(modelo_yolo, n_threads, conf_yolo)
        self.rectificador = RectificadorPerspectiva(
            ancho_salida=400,
            alto_salida=120,
        )
        self.clasificador = ClasificadorDigitos(modelo_ocr, n_threads, conf_ocr)

        print(f"[lector] Modelos cargados en {(time.monotonic()-t0)*1000:.0f}ms")

        if debug:
            Path("debug").mkdir(exist_ok=True)

    def leer(self, imagen: np.ndarray, nombre: str = "img") -> ResultadoLectura:
        """
        Ejecuta el pipeline completo sobre una imagen BGR de OpenCV.

        Args:
            imagen : imagen del medidor en formato BGR (numpy array)
            nombre : identificador para los archivos de debug

        Returns:
            ResultadoLectura con la lectura final y metadata de diagnóstico
        """
        tiempos = {}

        # ── Etapa 1: Detección ────────────────────────────────
        t = time.monotonic()
        deteccion = self.detector.detectar(imagen)
        tiempos["deteccion_ms"] = round((time.monotonic() - t) * 1000, 1)

        if deteccion is None:
            return ResultadoLectura(
                lectura=None,
                exitoso=False,
                motivo_falla=f"Dial no detectado (conf < {self.detector.conf_min})",
                confianza_deteccion=0.0,
                confianza_ocr_media=0.0,
                tiempos_ms=tiempos,
            )

        # ── Etapa 2: Rectificación de perspectiva ─────────────
        t = time.monotonic()
        dial_rect = self.rectificador.rectificar(imagen, deteccion)
        tiempos["rectificacion_ms"] = round((time.monotonic() - t) * 1000, 1)

        if self.debug:
            cv2.imwrite(f"debug/{nombre}_1_dial_rect.jpg", dial_rect)

        # ── Etapa 3: Segmentación de dígitos ──────────────────
        t = time.monotonic()
        try:
            recortes, cortes, _ = segmentar(
                dial_rect,
                n_digitos=self.n_digitos,
                alto_salida=64,
                ancho_salida=32,
            )
        except Exception as e:
            return ResultadoLectura(
                lectura=None,
                exitoso=False,
                motivo_falla=f"Error en segmentación: {e}",
                confianza_deteccion=deteccion.confianza,
                confianza_ocr_media=0.0,
                tiempos_ms=tiempos,
            )
        tiempos["segmentacion_ms"] = round((time.monotonic() - t) * 1000, 1)

        # ── Etapa 4: Clasificación de dígitos ─────────────────
        t = time.monotonic()
        resultados_digitos = self.clasificador.clasificar_secuencia(recortes)
        tiempos["clasificacion_ms"] = round((time.monotonic() - t) * 1000, 1)
        tiempos["total_ms"] = round(sum(tiempos.values()), 1)

        # ── Validación ────────────────────────────────────────
        digitos_invalidos = [r for r in resultados_digitos if not r.valido]
        if digitos_invalidos:
            posiciones = [r.posicion for r in digitos_invalidos]
            resultado_baja = ResultadoLectura(
                lectura=".".join(str(r.clase) for r in resultados_digitos),
                exitoso=False,
                motivo_falla=(
                    f"Confianza baja en posiciones {posiciones} "
                    f"(umbral: {self.clasificador.conf_min:.0%})"
                ),
                confianza_deteccion=deteccion.confianza,
                confianza_ocr_media=self._media_confianza(resultados_digitos),
                digitos=resultados_digitos,
                tiempos_ms=tiempos,
            )
            if self.debug:
                self._guardar_debug_completo(
                    imagen, dial_rect, deteccion, cortes, recortes,
                    resultados_digitos, resultado_baja, nombre,
                )
            return resultado_baja

        lectura = "".join(str(r.clase) for r in resultados_digitos)

        resultado_ok = ResultadoLectura(
            lectura=lectura,
            exitoso=True,
            motivo_falla=None,
            confianza_deteccion=deteccion.confianza,
            confianza_ocr_media=self._media_confianza(resultados_digitos),
            digitos=resultados_digitos,
            tiempos_ms=tiempos,
        )
        if self.debug:
            self._guardar_debug_completo(
                imagen, dial_rect, deteccion, cortes, recortes,
                resultados_digitos, resultado_ok, nombre,
            )
        return resultado_ok

    def leer_desde_archivo(self, ruta: str) -> ResultadoLectura:
        """
        Carga una imagen desde disco y ejecuta el pipeline.
        Wrapper conveniente sobre leer() para uso desde línea de comandos.
        """
        imagen = cv2.imread(ruta)
        if imagen is None:
            return ResultadoLectura(
                lectura=None,
                exitoso=False,
                motivo_falla=f"No se pudo leer la imagen: {ruta}",
                confianza_deteccion=0.0,
                confianza_ocr_media=0.0,
            )
        return self.leer(imagen, nombre=Path(ruta).stem)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _media_confianza(digitos: list[ResultadoDigito]) -> float:
        if not digitos:
            return 0.0
        return round(sum(d.confianza for d in digitos) / len(digitos), 3)

    def _guardar_debug_completo(
        self,
        imagen_orig,
        dial_rect,
        deteccion,
        cortes,
        recortes,
        digitos,
        resultado,
        nombre,
    ):
        """
        Genera 4 imágenes de debug en debug/<nombre>/:

          01_original.jpg  — imagen original con bbox y keypoints
          02_dial_rect.jpg — dial rectificado con líneas de corte numeradas
          03_digitos.jpg   — cada dígito ampliado con predicción y confianza
          04_resumen.jpg   — panel consolidado con todas las etapas

        Se genera siempre, tanto si el resultado es exitoso como si falla.
        """
        import cv2 as _cv2
        import numpy as _np
        from pathlib import Path as _Path

        dir_debug = _Path("debug") / nombre
        dir_debug.mkdir(parents=True, exist_ok=True)

        COLORES = [
            (255,  80,  80), ( 80, 200,  80), ( 80, 120, 255),
            (255, 200,  50), ( 50, 220, 220), (200,  80, 220),
        ]
        VERDE  = ( 60, 200,  60)
        ROJO   = ( 60,  60, 220)
        BLANCO = (255, 255, 255)

        # ── 01: imagen original con detección ────────────────────
        img01    = imagen_orig.copy()
        h_o, w_o = img01.shape[:2]
        cx_b, cy_b, bw_b, bh_b = deteccion.bbox
        bx1 = int((cx_b - bw_b/2) * w_o)
        by1 = int((cy_b - bh_b/2) * h_o)
        bx2 = int((cx_b + bw_b/2) * w_o)
        by2 = int((cy_b + bh_b/2) * h_o)
        _cv2.rectangle(img01, (bx1, by1), (bx2, by2), VERDE, 2)
        _cv2.putText(img01, f"conf: {deteccion.confianza:.1%}",
                     (bx1, max(by1 - 6, 14)),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.55, VERDE, 2)
        if deteccion.keypoints:
            etiq = ["AI", "AD", "BD", "BI"]
            pts  = _np.array(deteccion.keypoints, dtype=_np.float32)
            pts[:, 0] *= w_o
            pts[:, 1] *= h_o
            for ki, (px, py) in enumerate(pts.astype(int)):
                col = COLORES[ki % len(COLORES)]
                _cv2.circle(img01, (px, py), 6, col, -1)
                _cv2.putText(img01, etiq[ki], (px + 7, py + 4),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        _cv2.imwrite(str(dir_debug / "01_original.jpg"), img01)

        # ── 02: dial rectificado con cortes numerados ─────────────
        dial_bgr = dial_rect if dial_rect.ndim == 3                    else _cv2.cvtColor(dial_rect, _cv2.COLOR_GRAY2BGR)
        img02    = _cv2.resize(dial_bgr, (800, 240),
                               interpolation=_cv2.INTER_NEAREST)
        sx = 800 / dial_rect.shape[1]
        for ci, (x0, x1) in enumerate(cortes):
            col  = COLORES[ci % len(COLORES)]
            lx0  = int(x0 * sx)
            lx1  = int(x1 * sx)
            _cv2.line(img02, (lx0, 0), (lx0, 240), col, 1)
            _cv2.line(img02, (lx1, 0), (lx1, 240), col, 1)
            cx_c = (lx0 + lx1) // 2
            _cv2.putText(img02, str(ci), (cx_c - 6, 18),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        _cv2.imwrite(str(dir_debug / "02_dial_rect.jpg"), img02)

        # ── 03: dígitos individuales ampliados ───────────────────
        PANEL_W, PANEL_H = 128, 180
        SEP_W            = 6
        paneles = []
        for rec, dig in zip(recortes, digitos):
            panel     = _np.ones((PANEL_H, PANEL_W, 3), dtype=_np.uint8) * 40
            rec_color = _cv2.cvtColor(rec, _cv2.COLOR_GRAY2BGR)                         if rec.ndim == 2 else rec.copy()
            rec_big   = _cv2.resize(rec_color, (PANEL_W, 128),
                                    interpolation=_cv2.INTER_NEAREST)
            panel[0:128, :] = rec_big
            borde = VERDE if dig.valido else ROJO
            _cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), borde, 3)
            txt_num        = str(dig.clase)
            (tw, th), _    = _cv2.getTextSize(
                txt_num, _cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            _cv2.putText(panel, txt_num,
                         ((PANEL_W - tw) // 2, 128 + th + 6),
                         _cv2.FONT_HERSHEY_SIMPLEX, 1.4, borde, 3)
            txt_conf    = f"{dig.confianza:.0%}"
            (cw, _), _  = _cv2.getTextSize(
                txt_conf, _cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _cv2.putText(panel, txt_conf,
                         ((PANEL_W - cw) // 2, PANEL_H - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)
            paneles.append(panel)

        sep_v    = _np.ones((PANEL_H, SEP_W, 3), dtype=_np.uint8) * 20
        tira_dig = paneles[0]
        for p in paneles[1:]:
            tira_dig = _np.hstack([tira_dig, sep_v, p])
        _cv2.imwrite(str(dir_debug / "03_digitos.jpg"), tira_dig)

        # ── 04: resumen consolidado ───────────────────────────────
        W_RES  = 800
        etapas = [
            ("Etapa 1 — deteccion",             img01),
            ("Etapa 2 — rectificacion y cortes", img02),
            ("Etapa 3 — digitos",                tira_dig),
        ]
        filas = []
        for etiqueta, img_e in etapas:
            h_e, w_e  = img_e.shape[:2]
            h_nuevo   = int(h_e * W_RES / w_e)
            img_esc   = _cv2.resize(img_e, (W_RES, h_nuevo),
                                    interpolation=_cv2.INTER_AREA)
            barra = _np.ones((22, W_RES, 3), dtype=_np.uint8) * 55
            _cv2.putText(barra, etiqueta, (8, 16),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)
            filas.append(barra)
            filas.append(img_esc)

        lectura_str = resultado.lectura if resultado.lectura else "FALLO"
        estado_str  = "OK" if resultado.exitoso                       else f"ERROR: {resultado.motivo_falla}"
        color_est   = VERDE if resultado.exitoso else ROJO
        barra_fin   = _np.ones((38, W_RES, 3), dtype=_np.uint8) * 25
        _cv2.putText(barra_fin, f"Lectura: {lectura_str}",
                     (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLANCO, 2)
        _cv2.putText(barra_fin, estado_str,
                     (8, 36), _cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_est, 1)
        filas.append(barra_fin)

        resumen = _np.vstack(filas)
        _cv2.imwrite(str(dir_debug / "04_resumen.jpg"), resumen)
        print(f"[debug] Imagenes guardadas en: {dir_debug}/")


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lectura automática de medidor de agua"
    )
    parser.add_argument(
        "--imagen",      required=True,
        help="Ruta a la imagen del medidor"
    )
    parser.add_argument(
        "--modelo-yolo", default=MODELO_YOLO_DEFAULT,
        help=f"Modelo TFLite YOLO (default: {MODELO_YOLO_DEFAULT})"
    )
    parser.add_argument(
        "--modelo-ocr",  default=MODELO_OCR_DEFAULT,
        help=f"Modelo TFLite OCR (default: {MODELO_OCR_DEFAULT})"
    )
    parser.add_argument(
        "--n-digitos",   type=int, default=N_DIGITOS_DEFAULT,
        help=f"Cantidad de dígitos del medidor (default: {N_DIGITOS_DEFAULT})"
    )
    parser.add_argument(
        "--conf-yolo",   type=float, default=CONF_YOLO_MIN,
        help=f"Umbral de confianza del detector (default: {CONF_YOLO_MIN})"
    )
    parser.add_argument(
        "--conf-ocr",    type=float, default=CONF_OCR_MIN,
        help=f"Umbral de confianza del clasificador (default: {CONF_OCR_MIN})"
    )
    parser.add_argument(
        "--threads",     type=int, default=THREADS_DEFAULT,
        help=f"Threads TFLite (default: {THREADS_DEFAULT})"
    )
    parser.add_argument(
        "--debug",       action="store_true",
        help="Guardar imágenes intermedias en debug/"
    )
    parser.add_argument(
        "--json",        action="store_true",
        help="Salida en formato JSON (útil para integración con otros sistemas)"
    )
    args = parser.parse_args()

    lector = LectorMedidor(
        modelo_yolo=args.modelo_yolo,
        modelo_ocr=args.modelo_ocr,
        n_digitos=args.n_digitos,
        conf_yolo=args.conf_yolo,
        conf_ocr=args.conf_ocr,
        n_threads=args.threads,
        debug=args.debug,
    )

    resultado = lector.leer_desde_archivo(args.imagen)

    if args.json:
        salida = {
            "lectura":             resultado.lectura,
            "exitoso":             resultado.exitoso,
            "motivo_falla":        resultado.motivo_falla,
            "confianza_deteccion": resultado.confianza_deteccion,
            "confianza_ocr_media": resultado.confianza_ocr_media,
            "digitos": [
                {
                    "posicion":  d.posicion,
                    "clase":     d.clase,
                    "confianza": round(d.confianza, 3),
                }
                for d in resultado.digitos
            ],
            "tiempos_ms": resultado.tiempos_ms,
        }
        print(json.dumps(salida, ensure_ascii=False, indent=2))
    else:
        print(f"\nImagen  : {args.imagen}")
        print(f"Lectura : {resultado.lectura or 'FALLO'}")
        print(f"Exitoso : {resultado.exitoso}")
        if not resultado.exitoso:
            print(f"Motivo  : {resultado.motivo_falla}")
        print(f"Conf. detección : {resultado.confianza_deteccion:.1%}")
        print(f"Conf. OCR media : {resultado.confianza_ocr_media:.1%}")
        if resultado.digitos:
            print("\nDígitos:")
            for d in resultado.digitos:
                estado = "OK" if d.valido else "BAJA CONF"
                print(f"  [{d.posicion}] → {d.clase}  ({d.confianza:.1%})  {estado}")
        if resultado.tiempos_ms:
            print(f"\nTiempos:")
            for k, v in resultado.tiempos_ms.items():
                print(f"  {k:<22}: {v} ms")


if __name__ == "__main__":
    main()