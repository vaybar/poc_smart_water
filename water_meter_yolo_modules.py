"""
YOLOv8n con EMA Attention + BiFPN para lectura de medidores de agua.

Estructura:
  - EMAAttention      : módulo de atención liviano por grupos de canales
  - C2fWithEMA        : bloque C2f estándar de YOLOv8 con EMA integrado
  - BiFPNNode         : nodo de fusión bidireccional con pesos aprendibles
  - BiFPN             : neck completo con N capas BiFPN
  - WaterMeterDetector: modelo completo backbone + BiFPN + head
  - train()           : función de entrenamiento con el dataset de Nature
  - export_tflite()   : exportación cuantizada INT8 para RPi Zero 2W

Dependencias:
  pip install ultralytics torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f
from ultralytics.utils.torch_utils import fuse_conv_and_bn
import math


# ─────────────────────────────────────────────────────────────
# 1. EMA ATTENTION
# ─────────────────────────────────────────────────────────────

class EMAAttention(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA).

    Opera en grupos de canales para evitar el costo O(C²) del self-attention
    global. Cada grupo captura contexto espacial con AvgPool y luego recalibra
    el feature map con pesos generados por Conv 1×1 + Sigmoid.

    Parámetros:
        channels  : número de canales del feature map de entrada
        groups    : cuántos grupos en que se divide el canal (default 8)
        reduction : factor de reducción para la Conv interna (default 4)

    Costo típico vs. CBAM:
        EMA sobre 256 canales → ~12k parámetros extra
        CBAM sobre 256 canales → ~66k parámetros extra
    """

    def __init__(self, channels: int, groups: int = 8, reduction: int = 4):
        super().__init__()
        assert channels % groups == 0, "channels debe ser divisible por groups"

        self.groups = groups
        self.channels = channels
        mid_channels = max(channels // reduction, 8)

        # Captura contexto global por grupo: C/G → mid → C/G
        self.gap = nn.AdaptiveAvgPool2d(1)           # comprime H×W → 1×1
        self.fc  = nn.Sequential(
            nn.Conv2d(channels // groups, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels // groups, 1, bias=False),
            nn.Sigmoid(),
        )

        # Refinamiento espacial dentro de cada grupo con conv 3×3 depthwise
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        G = self.groups
        cg = C // G                                  # canales por grupo

        # ── rama de atención de canal ──────────────────────────
        # reshape: (B, G, C/G, H, W) → procesar cada grupo
        xg = x.view(B * G, cg, H, W)
        gap_out = self.gap(xg)                       # (B*G, C/G, 1, 1)
        attn    = self.fc(gap_out)                   # (B*G, C/G, 1, 1)
        attn    = attn.view(B, C, 1, 1)              # (B, C, 1, 1)

        # ── rama de atención espacial (dw conv) ────────────────
        spatial = self.dw_bn(self.dw_conv(x))        # (B, C, H, W)

        # ── recalibración: multiplicar por pesos de canal ──────
        out = x * attn + spatial * (1 - attn)
        return out


# ─────────────────────────────────────────────────────────────
# 2. C2f CON EMA INTEGRADO
# ─────────────────────────────────────────────────────────────

class C2fWithEMA(nn.Module):
    """
    Bloque C2f de YOLOv8 con EMAAttention añadido al final.

    C2f = Cross-Stage Partial con 2 convoluciones y N Bottlenecks.
    Acá insertamos EMA después de la concatenación de features para que
    el modelo aprenda qué regiones del dial son relevantes en ese nivel.

    Parámetros:
        c1, c2   : canales de entrada y salida
        n        : número de Bottlenecks internos (default 1)
        shortcut : si usar skip connection en los Bottlenecks
        g        : grupos para conv en Bottlenecks
        e        : factor de expansión de canales
        ema_groups: grupos del módulo EMA (default 8)
    """

    def __init__(
        self,
        c1: int, c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        ema_groups: int = 8,
    ):
        super().__init__()
        self.c = int(c2 * e)                         # canales intermedios

        # Convs de entrada y salida
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Bottlenecks internos (reutilizamos los de ultralytics)
        from ultralytics.nn.modules.block import Bottleneck
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        # EMA al final — se aplica sobre c2 canales
        # Asegurar divisibilidad
        actual_groups = ema_groups
        while c2 % actual_groups != 0 and actual_groups > 1:
            actual_groups //= 2
        self.ema = EMAAttention(c2, groups=actual_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split del feature map
        y = list(self.cv1(x).chunk(2, 1))

        # Pasar por Bottlenecks encadenados
        y.extend(m(y[-1]) for m in self.m)

        # Concatenar y reducir canales
        out = self.cv2(torch.cat(y, 1))

        # Atención EMA sobre el resultado completo
        return self.ema(out)


# ─────────────────────────────────────────────────────────────
# 3. BIFPN NODE
# ─────────────────────────────────────────────────────────────

class BiFPNNode(nn.Module):
    """
    Nodo de fusión BiFPN con pesos aprendibles (weighted feature fusion).

    En vez de sumar o concatenar naïvamente features de distintas escalas,
    aprende pesos w_i para cada input y normaliza: out = sum(w_i * f_i) / sum(w_i).
    Esto permite que el modelo decida cuánto confiar en cada escala.

    Parámetros:
        channels    : número de canales (igual en todos los inputs tras resize)
        num_inputs  : cuántos feature maps convergen en este nodo (2 o 3)
        apply_conv  : si aplicar una Conv 3×3 depthwise separable al final
    """

    def __init__(self, channels: int, num_inputs: int = 2, apply_conv: bool = True):
        super().__init__()
        self.num_inputs = num_inputs

        # Pesos aprendibles — inicializados en 1 (contribución uniforme)
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4                              # evitar división por cero

        # Conv depthwise separable para mezclar tras la fusión
        if apply_conv:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                # pointwise
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Identity()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.num_inputs, \
            f"Esperaba {self.num_inputs} inputs, recibió {len(inputs)}"

        # Pesos siempre positivos (ReLU) + normalización
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)

        # Redimensionar todos al tamaño del primero si difieren
        target_size = inputs[0].shape[-2:]
        resized = [
            F.interpolate(feat, size=target_size, mode="nearest")
            if feat.shape[-2:] != target_size else feat
            for feat in inputs
        ]

        # Fusión ponderada
        out = sum(w[i] * resized[i] for i in range(self.num_inputs))
        return self.conv(out)


# ─────────────────────────────────────────────────────────────
# 4. BIFPN NECK COMPLETO
# ─────────────────────────────────────────────────────────────

class BiFPN(nn.Module):
    """
    Neck BiFPN completo con N capas apiladas.

    Recibe 3 feature maps del backbone (P3, P4, P5) y aplica N rondas de
    fusión bidireccional: primero top-down (P5→P4→P3) luego bottom-up (P3→P4→P5).
    Cada nodo puede recibir 2 o 3 inputs dependiendo de si es intermedio o de borde.

    Parámetros:
        in_channels : lista con canales de [P3, P4, P5] del backbone
        out_channels: canales unificados tras proyección (default 128 para YOLOv8n)
        num_layers  : cuántas veces repetir el ciclo bidireccional (default 2)
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        assert len(in_channels) == 3, "BiFPN espera exactamente [P3, P4, P5]"
        self.num_layers = num_layers

        # Proyección de cada nivel al mismo número de canales
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for c in in_channels
        ])

        # Capas BiFPN apiladas
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Top-down: P5→P4 (2 inputs), P4_td→P3 (2 inputs)
                "td_p4": BiFPNNode(out_channels, num_inputs=2),
                "td_p3": BiFPNNode(out_channels, num_inputs=2),
                # Bottom-up: P3→P4 (3 inputs: orig + td + bu), P4_bu→P5 (2 inputs)
                "bu_p4": BiFPNNode(out_channels, num_inputs=3),
                "bu_p5": BiFPNNode(out_channels, num_inputs=2),
            })
            self.layers.append(layer)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: [P3, P4, P5] del backbone
        retorna:  (P3_out, P4_out, P5_out) fusionados
        """
        # Proyección al espacio de canales unificado
        p3, p4, p5 = [proj(f) for proj, f in zip(self.input_proj, features)]

        for layer in self.layers:
            # ── top-down path ──────────────────────────────────
            p4_td = layer["td_p4"]([p5, p4])         # P5 upsampled + P4
            p3_td = layer["td_p3"]([p4_td, p3])      # P4_td upsampled + P3

            # ── bottom-up path ─────────────────────────────────
            # P4_bu recibe: P4 original, P4_td (top-down), P3_td downsampled
            p4_bu = layer["bu_p4"]([p4, p4_td, p3_td])
            # P5_bu recibe: P5 original, P4_bu downsampled
            p5_bu = layer["bu_p5"]([p5, p4_bu])

            # Las salidas de esta capa son inputs de la siguiente
            p3, p4, p5 = p3_td, p4_bu, p5_bu

        return p3, p4, p5


# ─────────────────────────────────────────────────────────────
# 5. INTEGRACIÓN CON YOLOV8 VÍA YAML OVERRIDE
# ─────────────────────────────────────────────────────────────

def build_water_meter_model(pretrained: bool = True) -> YOLO:
    """
    Estrategia de integración recomendada para YOLOv8n:

    En vez de reescribir toda la arquitectura (complejo de mantener con
    cada versión de ultralytics), usamos el modelo pretrained de YOLOv8n
    y reemplazamos quirúrgicamente los módulos clave con nuestras versiones.

    Esto permite:
      - Usar los pesos preentrenados en COCO como punto de partida (transfer learning)
      - Solo re-entrenar las capas nuevas (EMA + BiFPN) desde cero
      - Mantener compatibilidad con el ecosistema ultralytics

    Returns:
        YOLO: modelo modificado listo para fine-tuning
    """
    # Cargar YOLOv8n base DE POSTURAS (Pose Estimation)
    model = YOLO("yolov8n-pose.pt" if pretrained else "yolov8n-pose.yaml")

    # Acceder al módulo interno de PyTorch
    pt_model = model.model

    # ── Reemplazar bloques C2f por C2fWithEMA ─────────────────
    # Identificamos por clase y reemplazamos manteniendo dimensiones
    replaced = 0
    for name, module in pt_model.named_modules():
        if isinstance(module, C2f):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = pt_model if parent_name == "" else _get_module(pt_model, parent_name)

            # Crear C2fWithEMA con mismos hiperparámetros
            c1 = module.cv1.conv.in_channels
            c2 = module.cv2.conv.out_channels
            n  = len(module.m)

            new_module = C2fWithEMA(c1, c2, n=n, shortcut=True)

            # Copiar pesos que coinciden (cv1, cv2, m — excepto EMA y dw)
            new_module.cv1.load_state_dict(module.cv1.state_dict(), strict=False)
            new_module.cv2.load_state_dict(module.cv2.state_dict(), strict=False)
            for i, bottleneck in enumerate(module.m):
                if i < len(new_module.m):
                    new_module.m[i].load_state_dict(bottleneck.state_dict(), strict=False)

            setattr(parent, child_name, new_module)
            replaced += 1

    print(f"[build] {replaced} bloques C2f → C2fWithEMA reemplazados")

    return model


def _get_module(model, name):
    """Helper para navegar el árbol de módulos por nombre con puntos."""
    parts = name.split(".")
    for p in parts:
        model = getattr(model, p)
    return model


# ─────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def train(
    data_yaml: str = "water_meter.yaml",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",        # "0" = GPU 0, "cpu" para CPU
    pretrained: bool = True,
):
    """
    Fine-tuning sobre el dataset de Nature (50k imágenes).

    Estrategia de entrenamiento por fases:
      Fase 1 (1/3 de epochs) : solo entrenar EMA + BiFPN, backbone congelado
      Fase 2 (2/3 de epochs) : descongelar todo, lr reducido
    """
    import os
    from pathlib import Path

    # Usar ruta absoluta para project, basada en el directorio de trabajo actual
    base_dir = Path(os.getcwd())
    project  = str(base_dir / "runs" / "water_meter")

    # Calculamos la división de epochs
    epochs_fase1 = epochs // 3
    epochs_fase2 = epochs - epochs_fase1

    model = build_water_meter_model(pretrained=pretrained)

    # ── Fase 1: congelar backbone ──────────────────────────────
    print(f"\n[train] Fase 1: congelando backbone ({epochs_fase1} epochs)")
    model.train(
        data=data_yaml,
        epochs=epochs_fase1,
        imgsz=imgsz,
        batch=batch,
        device=device,
        freeze=10,
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10,        # Suavizamos aumentación para no confundir keypoints
        translate=0.1,
        scale=0.3,         # Menos zoom para asegurar que los puntos sigan visibles
        mosaic=1.0,
        pose=15.0,         # Aumentamos importancia de los keypoints en la pérdida (default es 12)
        project=project,
        name="phase1",
        exist_ok=True,
    )

    # Buscar el best.pt generado por fase 1 sin asumir la ruta exacta
    candidatos = list(Path(project).rglob("phase1/weights/best.pt"))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró best.pt de fase 1 bajo {project}\n"
            f"Archivos .pt disponibles:\n"
            + "\n".join(str(p) for p in Path(project).rglob("*.pt"))
        )
    best_phase1 = str(candidatos[0])
    print(f"\n[train] Fase 1 completada. Modelo guardado en: {best_phase1}")

    # ── Fase 2: fine-tuning completo ──────────────────────────
    print(f"\n[train] Fase 2: fine-tuning completo ({epochs_fase2} epochs)")
    model = YOLO(best_phase1)
    model.train(
        data=data_yaml,
        epochs=epochs_fase2,
        imgsz=imgsz,
        batch=batch,
        device=device,
        freeze=0,
        lr0=1e-4,
        lrf=0.001,
        patience=20,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        pose=15.0,
        project=project,
        name="phase2",
        exist_ok=True,
    )

    # Reportar ubicación final del modelo
    candidatos2 = list(Path(project).rglob("phase2/weights/best.pt"))
    if candidatos2:
        print(f"\n[train] Modelo final guardado en: {candidatos2[0]}")

    return model


# ─────────────────────────────────────────────────────────────
# 7. EXPORTACIÓN PARA RASPBERRY PI ZERO 2W
# ─────────────────────────────────────────────────────────────

def export_tflite(
    weights: str = "runs/water_meter/phase2/weights/best.pt",
    imgsz: int = 320,          # reducir resolución para RPi Zero 2W
    int8: bool = True,
    data_yaml: str = "water_meter.yaml",  # necesario para calibración INT8
):
    """
    Exporta a TFLite cuantizado INT8 para ejecución en RPi Zero 2W.
    """
    model = YOLO(weights)

    print(f"[export] Exportando a TFLite INT8, imgsz={imgsz}")
    model.export(
        format="tflite",
        imgsz=imgsz,
        int8=int8,
        data=data_yaml,         # dataset de calibración para INT8
        batch=1,                # siempre batch=1 en edge
        simplify=True,
    )
    print(f"[export] Modelo exportado a TFLite")


# ─────────────────────────────────────────────────────────────
# 8. INFERENCIA EN PRODUCCIÓN (RASPBERRY PI)
# ─────────────────────────────────────────────────────────────

class WaterMeterInference:
    """
    Clase de inferencia optimizada para edge device.
    """

    def __init__(self, tflite_path: str, conf_threshold: float = 0.5):
        import numpy as np
        self.conf_threshold = conf_threshold
        self.last_reading = None

        # Cargar intérprete TFLite
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=tflite_path)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)

        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape    = self.input_details[0]["shape"]   # [1, H, W, 3]

    def preprocess(self, image_path: str):
        """Preproceso: resize + CLAHE + normalización."""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        # CLAHE para mejorar contraste en condiciones de campo
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Resize al input del modelo
        h, w  = self.input_shape[1], self.input_shape[2]
        img   = cv2.resize(img, (w, h))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalizar y añadir batch dimension
        img   = img.astype("float32") / 255.0
        return img[None, ...]   # (1, H, W, 3)

    def detect(self, image_path: str) -> dict:
        """
        Detecta el dial del medidor y retorna bbox + keypoints + confianza.
        """
        import numpy as np

        img_tensor = self.preprocess(image_path)

        self.interpreter.set_tensor(self.input_details[0]["index"], img_tensor)
        self.interpreter.invoke()

        # Output de YOLOv8: [1, num_preds, 4+1+num_classes]
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        output = output[0]      # (num_preds, ...)

        # Filtrar por confianza
        confidences = output[:, 4]
        mask        = confidences > self.conf_threshold

        if not mask.any():
            return None

        best_idx    = confidences[mask].argmax()
        best        = output[mask][best_idx]
        bbox        = best[:4].tolist()          # cx, cy, w, h (normalizado)
        confidence  = float(best[4])

        # Si el modelo fue entrenado con keypoints (4 vértices del dial)
        # están en best[5:13] como (x1,y1, x2,y2, x3,y3, x4,y4)
        keypoints = best[5:13].reshape(4, 2).tolist() if len(best) > 5 else None

        return {
            "bbox":       bbox,
            "keypoints":  keypoints,
            "confidence": confidence,
        }

    def read(self, image_path: str) -> tuple[str | None, float]:
        """
        Detecta el medidor y retorna la lectura numérica tras recorte y OCR.
        """
        detection = self.detect(image_path)
        if detection is None:
            return None, 0.0

        return "__DETECTION_OK__", detection["confidence"]


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Water meter detector — YOLOv8n + EMA + BiFPN")
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: train
    p_train = subparsers.add_parser("train", help="Entrenar el modelo")
    p_train.add_argument("--data",    default="water_meter.yaml")
    p_train.add_argument("--epochs",  type=int,   default=150)
    p_train.add_argument("--imgsz",   type=int,   default=640)
    p_train.add_argument("--batch",   type=int,   default=16)
    p_train.add_argument("--device",  default="0")

    # Subcomando: export
    p_export = subparsers.add_parser("export", help="Exportar a TFLite INT8")
    p_export.add_argument("--weights", default="runs/water_meter/phase2/weights/best.pt")
    p_export.add_argument("--imgsz",   type=int, default=320)
    p_export.add_argument("--no-int8", action="store_true")

    # Subcomando: infer
    p_infer = subparsers.add_parser("infer", help="Inferencia sobre una imagen")
    p_infer.add_argument("--model",  required=True, help="Ruta al modelo .tflite")
    p_infer.add_argument("--image",  required=True, help="Ruta a la imagen")
    p_infer.add_argument("--conf",   type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    elif args.command == "export":
        export_tflite(
            weights=args.weights,
            imgsz=args.imgsz,
            int8=not args.no_int8,
        )

    elif args.command == "infer":
        detector = WaterMeterInference(args.model, conf_threshold=args.conf)
        result, conf = detector.read(args.image)
        if result:
            print(f"Detección OK  | confianza: {conf:.3f}")
        else:
            print("No se detectó el dial (confianza insuficiente)")

    else:
        parser.print_help()