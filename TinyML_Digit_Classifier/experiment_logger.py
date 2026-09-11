"""
experiment_logger.py - Detailed Experiment Tracking & Bitácora Automática

Logs training hyperparameters, full classification reports, confusion matrices,
quantization impact, latency, and memory footprints to:
1. experiments/{EXP_ID}/report.json (machine-readable complete experiment dump)
2. experiments/{EXP_ID}/report.md (human-readable detailed experiment report)
3. experiments_log.csv (master tabular data for Pandas/Excel analysis)
4. BITACORA_EXPERIMENTOS.md (master Markdown table for GitHub repository)
"""

import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import config

LOG_CSV_PATH = config.BASE_DIR / "experiments_log.csv"
LOG_MD_PATH  = config.BASE_DIR / "BITACORA_EXPERIMENTOS.md"

def log_experiment(
    alpha: float,
    input_shape: str,
    epochs: int,
    batch_size: int,
    float_acc: float | None,
    int8_acc: float,
    flash_kb: float,
    arena_kb: float,
    latency_ms: float | None = None,
    classification_report_dict: dict | None = None,
    confusion_matrix_arr: list | np.ndarray | None = None,
    notes: str = "",
    hypothesis: str = "",
    conclusions: str = "",
    exp_id_override: str | None = None,
    timestamp_override: str | None = None
) -> str:
    """
    Appends experiment results to CSV, generates per-experiment JSON and Markdown reports,
    and updates the main BITACORA_EXPERIMENTOS.md file.
    """
    now = timestamp_override if timestamp_override else datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Check or assign Experiment ID
    if exp_id_override:
        exp_id = exp_id_override
    else:
        existing_nums = []
        if LOG_CSV_PATH.exists():
            df_existing = pd.read_csv(LOG_CSV_PATH)
            for item in df_existing["Exp_ID"].dropna():
                if str(item).startswith("EXP_"):
                    try:
                        existing_nums.append(int(str(item).replace("EXP_", "")))
                    except ValueError:
                        pass
        if config.EXPERIMENTS_DIR.exists():
            for folder in config.EXPERIMENTS_DIR.iterdir():
                if folder.is_dir() and folder.name.startswith("EXP_"):
                    try:
                        existing_nums.append(int(folder.name.replace("EXP_", "")))
                    except ValueError:
                        pass
        max_num = max(existing_nums) if existing_nums else 0
        exp_num = max_num + 1
        exp_id = f"EXP_{exp_num:03d}"
        
    quant_loss = (float_acc - int8_acc) if float_acc is not None else 0.0

    # Extract macro / weighted metrics if report dictionary is provided
    macro_prec = classification_report_dict.get("macro avg", {}).get("precision", None) if classification_report_dict else None
    macro_rec  = classification_report_dict.get("macro avg", {}).get("recall", None) if classification_report_dict else None
    macro_f1   = classification_report_dict.get("macro avg", {}).get("f1-score", None) if classification_report_dict else None

    weighted_prec = classification_report_dict.get("weighted avg", {}).get("precision", None) if classification_report_dict else None
    weighted_rec  = classification_report_dict.get("weighted avg", {}).get("recall", None) if classification_report_dict else None
    weighted_f1   = classification_report_dict.get("weighted avg", {}).get("f1-score", None) if classification_report_dict else None

    # Convert confusion matrix to list if numpy array
    if isinstance(confusion_matrix_arr, np.ndarray):
        cm_list = confusion_matrix_arr.tolist()
    else:
        cm_list = confusion_matrix_arr

    # Create dedicated experiment folder
    exp_dir = config.EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_rel_path = f"experiments/{exp_id}/report.md"

    # 1. Save JSON Report
    json_data = {
        "exp_id": exp_id,
        "timestamp": now,
        "hyperparameters": {
            "alpha": alpha,
            "input_shape": str(input_shape),
            "epochs": epochs,
            "batch_size": batch_size
        },
        "summary_metrics": {
            "float32_accuracy_%": round(float_acc, 2) if float_acc is not None else None,
            "int8_accuracy_%": round(int8_acc, 2),
            "quantization_loss_%": round(quant_loss, 2),
            "latency_ms_per_image": round(latency_ms, 4) if latency_ms is not None else None,
            "flash_kb": round(flash_kb, 2),
            "ram_arena_kb": round(arena_kb, 2),
            "macro_precision": round(macro_prec, 4) if macro_prec is not None else None,
            "macro_recall": round(macro_rec, 4) if macro_rec is not None else None,
            "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
            "weighted_precision": round(weighted_prec, 4) if weighted_prec is not None else None,
            "weighted_recall": round(weighted_rec, 4) if weighted_rec is not None else None,
            "weighted_f1": round(weighted_f1, 4) if weighted_f1 is not None else None
        },
        "classification_report": classification_report_dict,
        "confusion_matrix": cm_list,
        "notes": notes,
        "hypothesis": hypothesis,
        "conclusions": conclusions
    }

    json_file_path = exp_dir / "report.json"
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"--> Saved detailed JSON report to: {json_file_path}")

    # 2. Save Individual Markdown Report
    generate_experiment_markdown_report(exp_dir / "report.md", json_data)

    # 3. Update Master CSV Log
    new_row = {
        "Exp_ID": exp_id,
        "Timestamp": now,
        "Alpha": alpha,
        "Input_Shape": str(input_shape),
        "Epochs": epochs,
        "Batch_Size": batch_size,
        "Float32_Acc_%": round(float_acc, 2) if float_acc is not None else "N/A",
        "INT8_Acc_%": round(int8_acc, 2),
        "Quant_Loss_%": round(quant_loss, 2),
        "Macro_F1": round(macro_f1, 4) if macro_f1 is not None else "N/A",
        "Weighted_F1": round(weighted_f1, 4) if weighted_f1 is not None else "N/A",
        "Latency_ms": round(latency_ms, 3) if latency_ms is not None else "N/A",
        "Flash_KB": round(flash_kb, 2),
        "RAM_Arena_KB": round(arena_kb, 2),
        "Notes": notes,
        "Report_Path": report_rel_path
    }

    if LOG_CSV_PATH.exists():
        df_existing = pd.read_csv(LOG_CSV_PATH)
        # Update existing row if exp_id exists, else append
        if exp_id in df_existing["Exp_ID"].values:
            df_existing.loc[df_existing["Exp_ID"] == exp_id, list(new_row.keys())] = list(new_row.values())
            df_combined = df_existing
        else:
            df_new = pd.DataFrame([new_row])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([new_row])

    df_combined.to_csv(LOG_CSV_PATH, index=False)
    print(f"--> Master CSV updated: {LOG_CSV_PATH}")

    # 4. Update Main Markdown Bitácora
    generate_master_markdown_log(df_combined)

    return exp_id

def generate_experiment_markdown_report(md_path: Path, data: dict):
    exp_id = data["exp_id"]
    metrics = data["summary_metrics"]
    hp = data["hyperparameters"]
    rep = data["classification_report"]
    cm = data["confusion_matrix"]

    md = f"""# 🔬 Reporte Detallado de Experimento - `{exp_id}`

- **Fecha / Hora:** {data['timestamp']}
- **Notas / Cambios:** {data['notes'] if data['notes'] else 'Sin notas adicionales'}

---

## 🎯 Hipótesis y Contexto del Experimento
{data.get('hypothesis') if data.get('hypothesis') else '_No se especificó hipótesis formal para este experimento._'}

---

## ⚙️ Hiperparámetros de Configuración

| Parámetro | Valor |
| :--- | :--- |
| **Multiplicador de Ancho ($\\alpha$)** | `{hp['alpha']}` |
| **Dimensión de Entrada** | `{hp['input_shape']}` |
| **Épocas de Entrenamiento** | `{hp['epochs']}` |
| **Tamaño de Batch** | `{hp['batch_size']}` |

---

## 📊 Métricas Globales y Rendimiento

| Métrica | Valor | Objetivo TinyML |
| :--- | :--- | :--- |
| **Exactitud Float32** | `{metrics['float32_accuracy_%']}%` | N/A |
| **Exactitud INT8** | **`{metrics['int8_accuracy_%']}%`** | **> 85.0%** |
| **Pérdida por Cuantización** | `{metrics['quantization_loss_%']}%` | **< 1.0%** |
| **Latencia Promedio por Imagen** | `{metrics['latency_ms_per_image']} ms` | **< 5.0 ms** |
| **Macro F1-Score** | `{metrics['macro_f1']}` | Max |
| **Weighted F1-Score** | `{metrics['weighted_f1']}` | Max |
| **Tamaño en Flash (TFLite INT8)** | **`{metrics['flash_kb']} KB`** | **< 256 KB** |
| **RAM Estimada (Tensor Arena)** | **`{metrics['ram_arena_kb']} KB`** | **< 40 KB** |

---

## 📈 Desglose por Clase (Classification Report)

"""
    if rep:
        md += "| Clase | Precision | Recall | F1-Score | Muestras (Support) |\n"
        md += "| :--- | :--- | :--- | :--- | :--- |\n"
        for key, val in rep.items():
            if isinstance(val, dict):
                p = f"{val['precision']:.3f}"
                r = f"{val['recall']:.3f}"
                f1 = f"{val['f1-score']:.3f}"
                sup = val['support']
                class_label = f"**{key}**" if "avg" in key or key == "accuracy" else f"Dígito '{key}'"
                md += f"| {class_label} | {p} | {r} | **{f1}** | {sup} |\n"
    else:
        md += "_No se incluyó desglose por clase en este reporte._\n"

    md += "\n---\n\n## 🧩 Matriz de Confusión\n\n"

    if cm:
        md += "```\nTrue \\ Pred | " + " ".join([f"{d:4d}" for d in range(len(cm))]) + "\n"
        md += "-" * (14 + 5 * len(cm)) + "\n"
        for i, row in enumerate(cm):
            row_str = " ".join([f"{val:4d}" for val in row])
            md += f" Digit '{i}'  | {row_str}\n"
        md += "```\n\n"

    # Reference to plot image if exists
    img_path = md_path.parent / "confusion_matrix.png"
    if img_path.exists():
        md += "### Visualización Gráfica\n\n"
        md += f"![Matriz de Confusión](confusion_matrix.png)\n\n"

    md += "---\n\n## 💡 Conclusiones y Aprendizajes\n\n"
    md += f"{data.get('conclusions') if data.get('conclusions') else '_No se registraron conclusiones explícitas para este experimento._'}\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"--> Saved individual Markdown report to: {md_path}")

def generate_master_markdown_log(df: pd.DataFrame):
    md_content = """# 📓 Bitácora de Experimentos y Comparativos - TinyML Digit Classifier

Este documento registra automáticamente los resultados de cada experimento, cambios de hiperparámetros y evolución de métricas de precisión, F1-Score y consumo de memoria (Flash/RAM).

---

## 📊 Tabla Comparativa de Experimentos

| Exp ID | Fecha | Alpha ($\\alpha$) | Entrada | Épocas | Batch | Float32 Acc | INT8 Acc | Pérdida INT8 | Macro F1 | Weighted F1 | Latencia (ms) | Flash (KB) | RAM Arena (KB) | Reporte Detallado | Notas / Cambios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in df.iterrows():
        f32_acc = f"{row['Float32_Acc_%']}%" if pd.notnull(row['Float32_Acc_%']) else "N/A"
        quant_loss = f"{row['Quant_Loss_%']}%" if pd.notnull(row['Quant_Loss_%']) else "N/A"
        macro_f1 = f"{row['Macro_F1']:.3f}" if pd.notnull(row.get('Macro_F1')) and row.get('Macro_F1') != "N/A" else "-"
        weighted_f1 = f"{row['Weighted_F1']:.3f}" if pd.notnull(row.get('Weighted_F1')) and row.get('Weighted_F1') != "N/A" else "-"
        latency = f"{row['Latency_ms']:.3f}" if pd.notnull(row.get('Latency_ms')) and row.get('Latency_ms') != "N/A" else "-"
        report_link = f"[📄 Ver Reporte]({row['Report_Path']})" if pd.notnull(row.get('Report_Path')) else "-"
        notes = row.get('Notes', '')

        md_content += (
            f"| `{row['Exp_ID']}` | {row['Timestamp']} | {row['Alpha']} | `{row['Input_Shape']}` | "
            f"{row['Epochs']} | {row['Batch_Size']} | {f32_acc} | **{row['INT8_Acc_%']}%** | "
            f"{quant_loss} | {macro_f1} | {weighted_f1} | {latency} | **{row['Flash_KB']} KB** | **{row['RAM_Arena_KB']} KB** | "
            f"{report_link} | {notes} |\n"
        )
        
    md_content += """
---

### 💡 Leyenda de Métricas:
- **Flash (KB):** Peso del binario cuantizado en disco (Límite objetivo: **< 256 KB**).
- **RAM Arena (KB):** Memoria de activaciones intermedias requerida en la ESP32-S3 (Límite objetivo: **< 40 KB**).
- **Pérdida INT8:** Degradación de exactitud al cuantizar (Pérdida recomendada: **< 1.0%**).
- **Macro F1:** Promedio no ponderado de F1-Score entre las 10 clases (Evalúa equilibrio general).
- **Weighted F1:** Promedio ponderado por la cantidad de imágenes de prueba por clase.
"""

    with open(LOG_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"--> Bitácora Markdown maestra actualizada: {LOG_MD_PATH}")

