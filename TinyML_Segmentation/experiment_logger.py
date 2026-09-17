"""
experiment_logger.py - Detailed Experiment Tracking & Bitácora Automática for Segmentation

Logs training hyperparameters, keypoint regression metrics, quantization impact,
latency, and memory footprints to:
1. experiments/{EXP_ID}/report.json (machine-readable complete experiment dump)
2. experiments/{EXP_ID}/report.md (human-readable detailed experiment report)
3. experiments_log.csv (master tabular data for analysis)
4. BITACORA_EXPERIMENTOS.md (master Markdown table for GitHub repository)
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import config

LOG_CSV_PATH = config.BASE_DIR / "experiments_log.csv"
LOG_MD_PATH  = config.BASE_DIR / "BITACORA_EXPERIMENTOS.md"

def log_segmentation_experiment(
    alpha: float,
    input_shape: str,
    epochs: int,
    batch_size: int,
    train_loss: float,
    val_loss: float,
    corner_mae_px: float,
    polygon_iou: float,
    int8_size_kb: float,
    tensor_arena_kb: float,
    latency_ms: float | None = None,
    notes: str = "",
    hypothesis: str = "",
    conclusions: str = "",
    exp_id_override: str | None = None
) -> str:
    """
    Records experiment metrics, saves structured JSON/MD reports, and updates the Bitácora.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Determine Experiment ID
    if exp_id_override:
        exp_id = exp_id_override
    else:
        existing_nums = []
        if LOG_CSV_PATH.exists():
            try:
                df_existing = pd.read_csv(LOG_CSV_PATH)
                for item in df_existing["Exp_ID"].dropna():
                    if str(item).startswith("EXP_"):
                        try:
                            existing_nums.append(int(str(item).replace("EXP_", "")))
                        except ValueError:
                            pass
            except Exception:
                pass

        if config.EXPERIMENTS_DIR.exists():
            for folder in config.EXPERIMENTS_DIR.iterdir():
                if folder.is_dir() and folder.name.startswith("EXP_"):
                    try:
                        existing_nums.append(int(folder.name.replace("EXP_", "")))
                    except ValueError:
                        pass

        max_num = max(existing_nums) if existing_nums else 0
        exp_id = f"EXP_{max_num + 1:03d}"

    exp_dir = config.EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "Exp_ID": exp_id,
        "Timestamp": now,
        "Alpha": alpha,
        "Input_Shape": input_shape,
        "Epochs": epochs,
        "Batch_Size": batch_size,
        "Train_Loss": round(train_loss, 5),
        "Val_Loss": round(val_loss, 5),
        "Corner_MAE_px": round(corner_mae_px, 2),
        "Polygon_IoU": round(polygon_iou, 4),
        "Flash_INT8_KB": round(int8_size_kb, 2),
        "Tensor_Arena_KB": round(tensor_arena_kb, 2),
        "Latency_ESP32_ms": round(latency_ms, 1) if latency_ms else "N/A",
        "Hypothesis": hypothesis,
        "Notes": notes,
        "Conclusions": conclusions
    }

    # 1. Update master CSV
    if LOG_CSV_PATH.exists():
        df = pd.read_csv(LOG_CSV_PATH)
        df = df[df["Exp_ID"] != exp_id]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(LOG_CSV_PATH, index=False)

    # 2. Save JSON report
    with open(exp_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)

    # 3. Save individual Markdown report
    md_content = f"""# Reporte de Experimento: {exp_id}

- **Fecha:** {now}
- **Arquitectura:** Micro-Corner-Regressor ($\alpha={alpha}$)
- **Resolución Thumbnail:** {input_shape}
- **Hiperparámetros:** {epochs} épocas, Batch Size {batch_size}

## 1. Hipótesis
{hypothesis if hypothesis else "Línea base de localización de esquinas de ventanilla."}

## 2. Métricas de Precisión
- **Val Loss (Smooth L1 / MSE):** `{val_loss:.5f}`
- **Error Medio de Esquina (MAE):** `{corner_mae_px:.2f} px` (sobre 128x128)
- **Intersección sobre Unión (IoU Cuadrilátero):** `{polygon_iou * 100:.2f}%`

## 3. Huella de Recursos en ESP32 Clásico
- **Flash ROM (INT8):** `{int8_size_kb:.2f} KB` (Presupuesto máximo: 80 KB)
- **Tensor Arena (SRAM):** `{tensor_arena_kb:.2f} KB` (Presupuesto máximo: 35 KB)
- **Latencia estimada (240 MHz):** `{latency_ms if latency_ms else 'N/A'} ms`

## 4. Notas y Conclusiones
- **Notas:** {notes}
- **Conclusiones:** {conclusions}
"""
    with open(exp_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # 4. Regenerate BITACORA_EXPERIMENTOS.md
    _regenerate_bitacora_markdown()
    print(f"[experiment_logger] Experimento {exp_id} registrado exitosamente en BITACORA_EXPERIMENTOS.md")
    return exp_id

def _regenerate_bitacora_markdown():
    if not LOG_CSV_PATH.exists():
        return

    df = pd.read_csv(LOG_CSV_PATH)
    bitacora_text = """# Bitácora de Experimentos: TinyML_Segmentation (ESP32 Clásico)

Este documento registra sistemáticamente cada experimento realizado para la localización de la ventanilla del medidor de agua y su posterior rectificación geométrica en recortes de **64x32x1 en escala de grises**.

### Presupuesto de Hardware (ESP32 Clásico LX6):
- **Flash ROM Máximo:** `< 80 KB`
- **Tensor Arena Máximo:** `< 35 KB`
- **Formato de Salida Obligatorio:** $N$ recortes de `64x32x1` (uint8) para `TinyML_Digit_Classifier`.

---

## Tabla Histórica de Experimentos

| Exp ID | Fecha | Alpha | Input Shape | Val Loss | Corner MAE (px) | Polygon IoU | Flash INT8 (KB) | Tensor Arena (KB) | Notas / Hallazgos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for _, row in df.iterrows():
        bitacora_text += (
            f"| **{row['Exp_ID']}** | {row['Timestamp']} | {row['Alpha']} | {row['Input_Shape']} | "
            f"{row['Val_Loss']:.4f} | {row['Corner_MAE_px']:.1f} px | {row['Polygon_IoU']*100:.1f}% | "
            f"{row['Flash_INT8_KB']:.1f} KB | {row['Tensor_Arena_KB']:.1f} KB | {row['Notes']} |\n"
        )

    bitacora_text += """
---

## Resumen de Decisiones de Arquitectura
1. **Regresión Directa de 4 Esquinas:** Supera a YOLOv8n (3.3 MB vs <50 KB).
2. **Bilinear Quadrilateral Mapping:** Permite corregir ángulos de inclinación arbitrarios sin OpenCV.
3. **Corte Determinista por Ranuras (*Slots*):** Los tambores mecánicos tienen espaciado físico idéntico, eliminando la vulnerabilidad a barro y suciedad en el dial.
"""

    with open(LOG_MD_PATH, "w", encoding="utf-8") as f:
        f.write(bitacora_text)

if __name__ == "__main__":
    _regenerate_bitacora_markdown()
