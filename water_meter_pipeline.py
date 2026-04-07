"""
water_meter_pipeline.py

Módulo de rectificación de perspectiva y pipeline completo de lectura.

Conecta las tres etapas:
  1. Detección      : YOLOv8n + EMA + BiFPN  (water_meter_yolo_modules.py)
  2. Rectificación  : homografía con OpenCV   (este archivo)
  3. OCR            : PP-OCRv3 o CRNN        (este archivo)

Dependencias:
  pip install opencv-python-headless numpy paddlepaddle paddleocr

Uso rápido:
  pipeline = WaterMeterPipeline("best_int8.tflite")
  resultado = pipeline.leer("foto_medidor.jpg")
  print(resultado)  # {"lectura": "001823", "confianza": 0.94, "imagen_debug": ...}
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# TIPOS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class DeteccionDial:
    """Resultado del detector YOLOv8."""
    bbox: list[float]              # [cx, cy, w, h] normalizados (0..1)
    keypoints: list[list[float]]   # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] normalizados
    confianza: float
    ancho_imagen: int              # píxeles de la imagen original
    alto_imagen: int


@dataclass
class ResultadoLectura:
    lectura: str | None            # ej. "001823" — None si falló
    confianza_deteccion: float
    confianza_ocr: float
    exitoso: bool
    motivo_falla: str | None       # descripción si exitoso=False
    imagen_rectificada: np.ndarray | None   # para debug


# ─────────────────────────────────────────────────────────────
# ETAPA 2: RECTIFICACIÓN DE PERSPECTIVA
# ─────────────────────────────────────────────────────────────

class RectificadorPerspectiva:
    """
    Corrige la perspectiva del dial usando los 4 keypoints del detector.

    El problema que resuelve:
        La cámara casi nunca está perfectamente de frente al medidor.
        Si está ladeada, los números aparecen distorsionados y el OCR falla.
        La transformación de homografía "aplana" esa distorsión calculando
        cómo mapear los 4 vértices detectados a un rectángulo perfecto.
    """

    def __init__(
        self,
        ancho_salida: int = 400,
        alto_salida: int = 120,
        margen: int = 5,
    ):
        self.ancho_salida = ancho_salida
        self.alto_salida  = alto_salida
        self.margen       = margen

        # Destino: rectángulo perfecto en la imagen de salida
        # cv2.warpPerspective espera el orden estricto:
        # 0: Arriba-Izquierda
        # 1: Arriba-Derecha
        # 2: Abajo-Derecha
        # 3: Abajo-Izquierda
        self.destino = np.array([
            [margen,              margen             ],
            [ancho_salida-margen, margen             ],
            [ancho_salida-margen, alto_salida-margen ],
            [margen,              alto_salida-margen ],
        ], dtype=np.float32)

    def _desnormalizar_keypoints(
        self,
        keypoints: list[list[float]],
        ancho: int,
        alto: int,
    ) -> np.ndarray:
        """
        Convierte keypoints de coordenadas normalizadas (0..1) a píxeles.
        """
        pts = np.array(keypoints, dtype=np.float32)
        pts[:, 0] *= ancho   # x → píxeles
        pts[:, 1] *= alto    # y → píxeles
        return pts

    def _ordenar_keypoints(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordena los 4 keypoints geométricamente para garantizar que el dial
        siempre se rectifique en formato horizontal y de izquierda a derecha.
        Ignora el orden en el que el modelo YOLO devuelve los puntos.

        Corrección aplicada: después del ordenamiento por ángulos se valida
        que el borde superior tenga Y media menor que el borde inferior.
        Si no se cumple (dial de cabeza) se rota el array 2 posiciones.
        Esto corrige casos donde arctan2 empieza desde un ángulo distinto
        y los puntos quedan invertidos verticalmente.
        """
        # 1. Ordenar los puntos en un círculo (sentido antihorario por arctan2)
        centro  = np.mean(pts, axis=0)
        angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
        pts_cw  = pts[np.argsort(angulos)]

        # 2. Calcular longitudes de los 4 bordes
        e01 = np.linalg.norm(pts_cw[0] - pts_cw[1])
        e12 = np.linalg.norm(pts_cw[1] - pts_cw[2])
        e23 = np.linalg.norm(pts_cw[2] - pts_cw[3])
        e30 = np.linalg.norm(pts_cw[3] - pts_cw[0])

        # 3. Determinar orientación horizontal o vertical
        if e01 + e23 > e12 + e30:
            # Medidor horizontal: los bordes largos son p0-p1 y p2-p3
            y_01 = (pts_cw[0, 1] + pts_cw[1, 1]) / 2.0
            y_23 = (pts_cw[2, 1] + pts_cw[3, 1]) / 2.0
            if y_01 < y_23:
                ordenados = pts_cw
            else:
                ordenados = np.roll(pts_cw, 2, axis=0)
        else:
            # Medidor vertical (cámara rotada ~90°)
            if pts_cw[1, 1] < pts_cw[0, 1]:
                ordenados = np.roll(pts_cw, -1, axis=0)
            else:
                ordenados = np.roll(pts_cw, 1, axis=0)

        # 4. Validación final: arriba-izq y arriba-der deben tener Y menor
        #    que abajo-der y abajo-izq. Si no, el dial quedó invertido
        #    verticalmente — rotar 180 grados.
        y_arriba = (ordenados[0, 1] + ordenados[1, 1]) / 2.0
        y_abajo  = (ordenados[2, 1] + ordenados[3, 1]) / 2.0
        if y_arriba > y_abajo:
            ordenados = np.roll(ordenados, 2, axis=0)

        # 5. Validación horizontal: arriba-izq debe tener X menor que arriba-der.
        #    Si no, los puntos de la fila superior están intercambiados.
        if ordenados[0, 0] > ordenados[1, 0]:
            ordenados[[0, 1]] = ordenados[[1, 0]]
            ordenados[[2, 3]] = ordenados[[3, 2]]

        return ordenados

    def _fallback_desde_bbox(
        self,
        bbox: list[float],
        ancho: int,
        alto: int,
    ) -> np.ndarray:
        """
        Genera 4 keypoints aproximados a partir del bounding box.
        """
        cx, cy, w, h = bbox
        x1 = (cx - w / 2) * ancho
        y1 = (cy - h / 2) * alto
        x2 = (cx + w / 2) * ancho
        y2 = (cy + h / 2) * alto

        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32)


    def rectificar(
        self,
        imagen: np.ndarray,
        deteccion: DeteccionDial,
    ) -> np.ndarray:
        """
        Aplica la corrección de perspectiva y devuelve el dial enderezado.
        """
        h_img, w_img = imagen.shape[:2]

        # Obtener puntos de origen en píxeles
        if deteccion.keypoints is not None and len(deteccion.keypoints) == 4:
            pts_origen = self._desnormalizar_keypoints(
                deteccion.keypoints, w_img, h_img
            )
        else:
            # Sin keypoints: usar el bounding box como aproximación
            pts_origen = self._fallback_desde_bbox(
                deteccion.bbox, w_img, h_img
            )

        # Garantizar orden espacial correcto para la homografía
        pts_origen = self._ordenar_keypoints(pts_origen)

        # Calcular la matriz de homografía H (3×3)
        H, _ = cv2.findHomography(pts_origen, self.destino, cv2.RANSAC, 5.0)

        if H is None:
            # findHomography falló
            return self._recorte_directo(imagen, deteccion)

        # Aplicar la transformación a toda la imagen
        rectificada = cv2.warpPerspective(
            imagen,
            H,
            (self.ancho_salida, self.alto_salida),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rectificada

    def _recorte_directo(
        self,
        imagen: np.ndarray,
        deteccion: DeteccionDial,
    ) -> np.ndarray:
        """
        Recorte simple sin corrección de perspectiva.
        """
        h_img, w_img = imagen.shape[:2]
        cx, cy, w, h = deteccion.bbox

        x1 = max(0, int((cx - w / 2) * w_img))
        y1 = max(0, int((cy - h / 2) * h_img))
        x2 = min(w_img, int((cx + w / 2) * w_img))
        y2 = min(h_img, int((cy + h / 2) * h_img))

        recorte = imagen[y1:y2, x1:x2]
        return cv2.resize(recorte, (self.ancho_salida, self.alto_salida))


# ─────────────────────────────────────────────────────────────
# ETAPA 3: OCR
# ─────────────────────────────────────────────────────────────

class LectorOCR:
    """
    Lee los dígitos del dial rectificado.
    """

    def __init__(self, backend: str = "paddleocr", solo_digitos: bool = True):
        self.backend      = backend
        self.solo_digitos = solo_digitos
        self._ocr         = None
        self._inicializar()

    def _inicializar(self):
        if self.backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                # use_angle_cls=False porque la imagen ya viene rectificada
                # use_gpu=False para edge devices
                self._ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                    rec_model_dir=None,   # descarga automática si no existe
                )
                print("[OCR] Backend: PP-OCRv3 (PaddleOCR)")
            except ImportError:
                print("[OCR] PaddleOCR no instalado, usando Tesseract como fallback")
                self.backend = "tesseract"

        if self.backend == "tesseract":
            try:
                import pytesseract
                self._ocr = pytesseract
                print("[OCR] Backend: Tesseract")
            except ImportError:
                print("[OCR] Tesseract no instalado, usando modo mock")
                self.backend = "mock"

        if self.backend == "mock":
            print("[OCR] Backend: mock (solo para tests)")

    def _preprocesar_para_ocr(self, imagen: np.ndarray) -> np.ndarray:
        """
        Preproceso específico para mejorar la lectura de dígitos.
        """
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gris  = clahe.apply(gris)

        # Binarización adaptativa
        binaria = cv2.adaptiveThreshold(
            gris, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # Suavizado leve
        binaria = cv2.GaussianBlur(binaria, (3, 3), 0)
        return binaria

    def leer(self, imagen_rectificada: np.ndarray) -> tuple[str | None, float]:
        """
        Ejecuta el OCR sobre la imagen rectificada del dial.
        """
        procesada = self._preprocesar_para_ocr(imagen_rectificada)

        if self.backend == "paddleocr":
            return self._leer_paddle(procesada)
        elif self.backend == "tesseract":
            return self._leer_tesseract(procesada)
        else:
            return self._leer_mock()

    def _leer_paddle(self, imagen: np.ndarray) -> tuple[str | None, float]:
        resultado = self._ocr.ocr(imagen, cls=False)

        if not resultado or not resultado[0]:
            return None, 0.0

        textos     = []
        confianzas = []
        for linea in resultado[0]:
            texto, conf = linea[1]
            textos.append(texto)
            confianzas.append(conf)

        texto_final = " ".join(textos)
        conf_final  = float(np.mean(confianzas)) if confianzas else 0.0

        if self.solo_digitos:
            texto_final = "".join(c for c in texto_final if c.isdigit())

        return (texto_final if texto_final else None), conf_final

    def _leer_tesseract(self, imagen: np.ndarray) -> tuple[str | None, float]:
        import pytesseract

        # Configuración para dígitos solamente
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        datos  = pytesseract.image_to_data(
            imagen,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        textos     = []
        confianzas = []
        for i, texto in enumerate(datos["text"]):
            if texto.strip():
                confianzas.append(int(datos["conf"][i]))
                textos.append(texto.strip())

        texto_final = "".join(textos)
        conf_final  = float(np.mean(confianzas)) / 100.0 if confianzas else 0.0

        if self.solo_digitos:
            texto_final = "".join(c for c in texto_final if c.isdigit())

        return (texto_final if texto_final else None), conf_final

    def _leer_mock(self) -> tuple[str | None, float]:
        return "001823", 0.95


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────

class WaterMeterPipeline:
    """
    Pipeline completo: detección → rectificación → OCR → validación.
    """

    def __init__(
        self,
        modelo_tflite: str,
        ocr_backend: str = "paddleocr",
        conf_deteccion: float = 0.5,
        conf_ocr: float = 0.6,
        guardar_debug: bool = False,
    ):
        self.conf_deteccion = conf_deteccion
        self.conf_ocr       = conf_ocr
        self.guardar_debug  = guardar_debug
        self.ultima_lectura: int | None = None

        self.detector      = self._cargar_detector(modelo_tflite)
        self.rectificador  = RectificadorPerspectiva(ancho_salida=400, alto_salida=120)
        self.ocr           = LectorOCR(backend=ocr_backend)

    def _cargar_detector(self, ruta_tflite: str):
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=ruta_tflite)
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=ruta_tflite)

        interpreter.allocate_tensors()
        return interpreter

    def _detectar(self, imagen: np.ndarray) -> DeteccionDial | None:
        input_details  = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        input_shape    = input_details[0]["shape"]
        
        if input_shape[1] == 3:
            h_model, w_model = input_shape[2], input_shape[3]
            nchw = True
        else:
            h_model, w_model = input_shape[1], input_shape[2]
            nchw = False
            
        h_orig, w_orig   = imagen.shape[:2]

        lab  = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        img_proc = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        img_proc = cv2.resize(img_proc, (w_model, h_model))
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
        input_dtype = input_details[0]["dtype"]
        if input_dtype == np.uint8:
            tensor = img_proc
        elif input_dtype == np.int8:
            tensor = (img_proc.astype(np.int16) - 128).astype(np.int8)
        else:
            tensor = img_proc.astype(np.float32) / 255.0
            
        if nchw:
            tensor = np.transpose(tensor, (2, 0, 1))
            
        tensor = np.expand_dims(tensor, axis=0)

        self.detector.set_tensor(input_details[0]["index"], tensor)
        self.detector.invoke()
        output = self.detector.get_tensor(output_details[0]["index"])[0]

        if output.shape[0] < output.shape[1] and output.shape[0] < 50:
            output = output.transpose()

        confidencias = output[:, 4]
        mejor_idx    = int(np.argmax(confidencias))
        mejor_conf   = float(confidencias[mejor_idx])

        if mejor_conf < self.conf_deteccion:
            return None

        mejor = output[mejor_idx]

        if mejor[2] <= 1.0 and mejor[3] <= 1.0:
            scale_w, scale_h = 1.0, 1.0
        else:
            scale_w, scale_h = w_model, h_model

        cx = float(mejor[0]) / scale_w
        cy = float(mejor[1]) / scale_h
        w  = float(mejor[2]) / scale_w
        h  = float(mejor[3]) / scale_h
        bbox = [cx, cy, w, h]

        keypoints = None
        resto = len(mejor) - 5
        
        if resto == 8:
            kpts_raw = mejor[5:13].reshape(4, 2)
            kpts_raw[:, 0] /= scale_w
            kpts_raw[:, 1] /= scale_h
            kpts = kpts_raw.tolist()
            if all(0.0 <= v <= 1.0 for pt in kpts for v in pt):
                keypoints = kpts
        elif resto == 12:
            kpts_raw = mejor[5:17].reshape(4, 3)
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

    def _validar_lectura(self, lectura: str) -> tuple[bool, str | None]:
        if not lectura or not lectura.isdigit():
            return False, "La lectura no es numérica"
        if not (4 <= len(lectura) <= 8):
            return False, f"Longitud inesperada: {len(lectura)} dígitos"
        valor_actual = int(lectura)
        if self.ultima_lectura is not None:
            if valor_actual < self.ultima_lectura:
                return False, f"La lectura bajó: {valor_actual} < {self.ultima_lectura}"
            incremento = valor_actual - self.ultima_lectura
            if incremento > 9999:
                return False, f"Incremento sospechosamente alto: {incremento} m³"
        return True, None

    def leer(self, ruta_imagen: str) -> ResultadoLectura:
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            return ResultadoLectura(
                lectura=None, confianza_deteccion=0.0, confianza_ocr=0.0,
                exitoso=False, motivo_falla=f"No se pudo leer: {ruta_imagen}",
                imagen_rectificada=None,
            )

        deteccion = self._detectar(imagen)
        if deteccion is None:
            return ResultadoLectura(
                lectura=None, confianza_deteccion=0.0, confianza_ocr=0.0,
                exitoso=False,
                motivo_falla=f"Dial no detectado (conf < {self.conf_deteccion})",
                imagen_rectificada=None,
            )

        imagen_rectificada = self.rectificador.rectificar(imagen, deteccion)

        if self.guardar_debug:
            self._guardar_debug(imagen, deteccion, imagen_rectificada, ruta_imagen)

        lectura, conf_ocr = self.ocr.leer(imagen_rectificada)

        if lectura is None or conf_ocr < self.conf_ocr:
            return ResultadoLectura(
                lectura=None,
                confianza_deteccion=deteccion.confianza,
                confianza_ocr=conf_ocr or 0.0,
                exitoso=False,
                motivo_falla=f"OCR falló o confianza baja ({conf_ocr:.2f})",
                imagen_rectificada=imagen_rectificada,
            )

        valida, motivo = self._validar_lectura(lectura)
        if not valida:
            return ResultadoLectura(
                lectura=lectura,
                confianza_deteccion=deteccion.confianza,
                confianza_ocr=conf_ocr,
                exitoso=False,
                motivo_falla=f"Lectura inválida: {motivo}",
                imagen_rectificada=imagen_rectificada,
            )

        self.ultima_lectura = int(lectura)

        return ResultadoLectura(
            lectura=lectura,
            confianza_deteccion=deteccion.confianza,
            confianza_ocr=conf_ocr,
            exitoso=True,
            motivo_falla=None,
            imagen_rectificada=imagen_rectificada,
        )

    def _guardar_debug(
        self,
        imagen_original: np.ndarray,
        deteccion: DeteccionDial,
        imagen_rectificada: np.ndarray,
        ruta_original: str,
    ):
        debug_dir = Path(ruta_original).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        stem = Path(ruta_original).stem

        h, w = imagen_original.shape[:2]
        cx, cy, bw, bh = deteccion.bbox
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        debug_orig = imagen_original.copy()
        cv2.rectangle(debug_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if deteccion.keypoints:
            pts = np.array(deteccion.keypoints, dtype=np.float32)
            pts[:, 0] *= w
            pts[:, 1] *= h
            
            # Reordenar los puntos para el dibujo
            pts_ordenados = RectificadorPerspectiva()._ordenar_keypoints(pts)
            etiquetas = ["0-AI", "1-AD", "2-BD", "3-BI"]
            colores = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            
            for i, (px, py) in enumerate(pts_ordenados.astype(int)):
                cv2.circle(debug_orig, (px, py), 5, colores[i], -1)
                cv2.putText(debug_orig, etiquetas[i], (px + 6, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[i], 1)

        cv2.imwrite(str(debug_dir / f"{stem}_deteccion.jpg"), debug_orig)
        cv2.imwrite(str(debug_dir / f"{stem}_rectificada.jpg"), imagen_rectificada)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pipeline de lectura de medidores")
    parser.add_argument("--modelo",  required=True, help="Ruta al modelo .tflite")
    parser.add_argument("--imagen",  required=True, help="Ruta a la imagen")
    parser.add_argument("--ocr",     default="paddleocr",
                        choices=["paddleocr", "tesseract", "mock"])
    parser.add_argument("--debug",   action="store_true",
                        help="Guardar imágenes de diagnóstico")
    parser.add_argument("--conf-det", type=float, default=0.5)
    parser.add_argument("--conf-ocr", type=float, default=0.6)
    args = parser.parse_args()

    pipeline  = WaterMeterPipeline(
        modelo_tflite=args.modelo,
        ocr_backend=args.ocr,
        conf_deteccion=args.conf_det,
        conf_ocr=args.conf_ocr,
        guardar_debug=args.debug,
    )
    resultado = pipeline.leer(args.imagen)

    salida = {
        "lectura":              resultado.lectura,
        "exitoso":              resultado.exitoso,
        "confianza_deteccion":  round(resultado.confianza_deteccion, 3),
        "confianza_ocr":        round(resultado.confianza_ocr, 3),
        "motivo_falla":         resultado.motivo_falla,
    }
    print(json.dumps(salida, ensure_ascii=False, indent=2))
