"""
ejemplo_uso.py

Uso completo del sistema, desde una imagen de cámara hasta la lectura final.

PASO 1 — Entrenar el modelo (una sola vez):
    python ejemplo_uso.py --modo train

PASO 2 — Exportar a TFLite para el dispositivo edge (una sola vez):
    python ejemplo_uso.py --modo export

PASO 3 — Leer un medidor desde una imagen:
    python ejemplo_uso.py --modo leer --imagen foto.jpg
"""

import argparse
import json

# ── Módulos del proyecto ──────────────────────────────────────
from water_meter_yolo_modules import train, export_tflite
from water_meter_pipeline import WaterMeterPipeline


def paso_train():
    """Entrena el modelo sobre el dataset ya convertido a YOLO."""
    train(
        data_yaml="water_meter/water_meter.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",       # cambiar a "cpu" si no hay GPU
    )


def paso_export():
    """Exporta el mejor modelo entrenado a TFLite INT8."""
    export_tflite(
        weights="runs/water_meter/phase2/weights/best.pt",
        imgsz=320,        # resolución reducida para edge device
        int8=True,
        data_yaml="water_meter/water_meter.yaml",
    )


def paso_leer(ruta_imagen: str, debug: bool = False):
    """Lee el medidor desde una imagen de cámara."""

    # Inicializar el pipeline completo
    pipeline = WaterMeterPipeline(
        #modelo_tflite="runs/water_meter/phase2/weights/best_saved_model/best_float32.tflite",
        modelo_tflite="best_float16.tflite",
        ocr_backend="paddleocr",   # "tesseract" en RPi Zero 2W
        conf_deteccion=0.5,
        conf_ocr=0.6,
        guardar_debug=debug,
    )

    # Una línea para obtener la lectura
    resultado = pipeline.leer(ruta_imagen)

    # Mostrar resultado
    print(json.dumps({
        "lectura":             resultado.lectura,
        "exitoso":             resultado.exitoso,
        "confianza_deteccion": round(resultado.confianza_deteccion, 3),
        "confianza_ocr":       round(resultado.confianza_ocr, 3),
        "motivo_falla":        resultado.motivo_falla,
    }, indent=2, ensure_ascii=False))

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo",   choices=["train", "export", "leer"], required=True)
    parser.add_argument("--imagen", default=None, help="Requerido si --modo leer")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    if args.modo == "train":
        paso_train()
    elif args.modo == "export":
        paso_export()
    elif args.modo == "leer":
        if not args.imagen:
            print("Error: --imagen es requerido con --modo leer")
        else:
            paso_leer(args.imagen, debug=args.debug)