"""
analyze_misclassifications.py - Automated Misclassification Diagnosis & Label Noise Detection

Evaluates the trained model (INT8 TFLite or Float32 Keras) on the test dataset:
1. Identifies all misclassified samples (y_true != y_pred).
2. Calculates error confidence (Softmax probability of wrong prediction).
3. Computes geometric & quality metrics (center of mass shift, Laplacian sharpness variance).
4. Flags high-risk Label Noise candidates (wrong predictions with > 70% confidence).
5. Exports:
   - models/label_noise_candidates.csv
   - models/error_analysis_summary.md
   - models/error_mosaic.png (5x5 grid of top errors)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

import config
from dataset import load_digit_dataset

def compute_center_of_mass(img_2d: np.ndarray):
    """Calculates x, y centroid coordinates of pixel mass."""
    moments = cv2.moments(img_2d)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = 16.0, 16.0
    return cx, cy

def analyze_errors(
    confidence_threshold: float = 0.70,
    max_mosaic_samples: int = 25
):
    print("=" * 65)
    print("  TinyML Digit Classifier - Automated Error & Label Noise Diagnosis")
    print("=" * 65)
    
    # 1. Load Test Dataset
    _, _, (x_test, y_test) = load_digit_dataset()
    print(f"Loaded test dataset: {len(x_test)} images\n")
    
    # 2. Obtain predictions (TFLite INT8 or Float32 Keras)
    y_probs = []
    
    if config.TFLITE_INT8_PATH.exists():
        print(f"--- Running inference with INT8 TFLite Model ({config.TFLITE_INT8_PATH.name}) ---")
        tflite_bytes = open(config.TFLITE_INT8_PATH, "rb").read()
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']
        
        for i in range(len(x_test)):
            img_float = x_test[i].astype(np.float32)
            img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
            img_input = np.expand_dims(img_quant, axis=0)
            
            interpreter.set_tensor(input_details['index'], img_input)
            interpreter.invoke()
            out_raw = interpreter.get_tensor(output_details['index'])[0]
            
            # De-quantize output probabilities or apply softmax
            out_dequant = (out_raw.astype(np.float32) - output_zero_point) * output_scale
            # Apply softmax to normalize
            exp_scores = np.exp(out_dequant - np.max(out_dequant))
            prob = exp_scores / np.sum(exp_scores)
            y_probs.append(prob)
            
        y_probs = np.array(y_probs)
    elif config.FLOAT_MODEL_PATH.exists():
        print(f"--- Running inference with Float32 Keras Model ({config.FLOAT_MODEL_PATH.name}) ---")
        float_model = tf.keras.models.load_model(str(config.FLOAT_MODEL_PATH))
        y_probs = float_model.predict(x_test, verbose=0)
    else:
        print("Error: No trained model checkpoint found! Run train.py or quantize_and_export.py first.")
        return

    y_preds = np.argmax(y_probs, axis=1)
    errors_idx = np.where(y_preds != y_test)[0]
    total_errors = len(errors_idx)
    acc = (1.0 - total_errors / len(x_test)) * 100.0
    
    print(f"\nEvaluation Results:")
    print(f"  - Total Test Samples : {len(x_test)}")
    print(f"  - Total Misclassified: {total_errors} ({total_errors / len(x_test) * 100:.2f}%)")
    print(f"  - Accuracy           : {acc:.2f}%\n")
    
    if total_errors == 0:
        print("🎉 Zero misclassifications found! Perfect accuracy.")
        return

    # 3. Analyze Error Details
    error_records = []
    
    for idx in errors_idx:
        img_2d = x_test[idx][:, :, 0]
        y_true = int(y_test[idx])
        y_pred = int(y_preds[idx])
        
        conf = float(y_probs[idx][y_pred])
        true_p = float(y_probs[idx][y_true])
        margin = conf - true_p
        
        # Geometry & quality metrics
        cx, cy = compute_center_of_mass(img_2d)
        dx = abs(cx - 16.0)
        dy = abs(cy - 16.0)
        laplacian_var = float(cv2.Laplacian(img_2d, cv2.CV_64F).var())
        
        # Risk assessment
        is_label_noise = (conf >= confidence_threshold)
        is_cropped = (dx > 6.0 or dy > 6.0)
        
        risk_tag = "LABEL_NOISE_CANDIDATE" if is_label_noise else ("CROPPED/OFF_CENTER" if is_cropped else "CONFUSION")
        
        error_records.append({
            "Sample_Idx": idx,
            "True_Label": y_true,
            "Pred_Label": y_pred,
            "Confidence": round(conf, 4),
            "True_Prob": round(true_p, 4),
            "Margin": round(margin, 4),
            "Center_X": round(cx, 2),
            "Center_Y": round(cy, 2),
            "Shift_Dx": round(dx, 2),
            "Shift_Dy": round(dy, 2),
            "Laplacian_Var": round(laplacian_var, 2),
            "Risk_Category": risk_tag
        })
        
    df_errors = pd.DataFrame(error_records)
    # Sort by confidence descending
    df_errors = df_errors.sort_values(by="Confidence", ascending=False)
    
    # 4. Export CSV of Noise Candidates
    csv_path = config.MODELS_DIR / "label_noise_candidates.csv"
    df_errors.to_csv(csv_path, index=False)
    print(f"--> Saved error analysis CSV to: {csv_path}")
    
    # Filter high-confidence label noise candidates
    noise_candidates = df_errors[df_errors["Risk_Category"] == "LABEL_NOISE_CANDIDATE"]
    print(f"--> Found {len(noise_candidates)} high-confidence Label Noise candidates (Confidence >= {confidence_threshold*100:.0f}%)")
    
    # 5. Export Markdown Summary Report
    md_report_path = config.MODELS_DIR / "error_analysis_summary.md"
    generate_error_markdown_summary(md_report_path, df_errors, len(x_test), acc)
    
    # 6. Render Error Mosaic Plot
    plot_error_mosaic(x_test, df_errors, max_mosaic_samples)
    
    return df_errors

def generate_error_markdown_summary(md_path: Path, df_errors: pd.DataFrame, total_samples: int, acc: float):
    total_err = len(df_errors)
    label_noise_count = len(df_errors[df_errors["Risk_Category"] == "LABEL_NOISE_CANDIDATE"])
    cropped_count = len(df_errors[df_errors["Risk_Category"] == "CROPPED/OFF_CENTER"])
    confusion_count = len(df_errors[df_errors["Risk_Category"] == "CONFUSION"])
    
    # Group errors by (True -> Pred) pair
    top_pairs = df_errors.groupby(["True_Label", "Pred_Label"]).size().reset_index(name="Count")
    top_pairs = top_pairs.sort_values(by="Count", ascending=False).head(8)

    md = f"""# 🔍 Reporte de Diagnóstico de Errores y Label Noise

- **Muestras Totales de Prueba:** {total_samples}
- **Exactitud Global:** {acc:.2f}%
- **Total Errores:** {total_err} ({total_err/total_samples*100:.2f}%)

---

## 📊 Categorización de Errores Detectados

| Categoría | Cantidad | % de Errores | Descripción |
| :--- | :--- | :--- | :--- |
| **Label Noise Candidates** | **{label_noise_count}** | **{label_noise_count/total_err*100:.1f}%** | Predicción errónea con confianza $> 70\%$. Alto riesgo de estar mal etiquetada en el dataset original. |
| **Recortes Descentrados** | **{cropped_count}** | **{cropped_count/total_err*100:.1f}%** | Dígito desplazado del centro por $> 6$ píxeles en la segmentación. |
| **Confusión Estándar** | **{confusion_count}** | **{confusion_count/total_err*100:.1f}%** | Ambigüedad visual entre dígitos similares. |

---

## ⚠️ Principales Pares de Confusión (True $\\rightarrow$ Pred)

| Clase Real | Predicción Errónea | Ocurrencias |
| :--- | :--- | :--- |
"""
    for _, row in top_pairs.iterrows():
        md += f"| Dígito '{row['True_Label']}' | Dígito '{row['Pred_Label']}' | **{row['Count']}** |\n"

    md += f"""
---

## 📝 Top 10 Muestras Sospechosas de Label Noise

| Sample Index | Clase Real | Predicho | Confianza Predicción | Varianza Nitidez | Categoría |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in df_errors.head(10).iterrows():
        md += f"| `{row['Sample_Idx']}` | Dígito '{row['True_Label']}' | **Dígito '{row['Pred_Label']}'** | `{row['Confidence']*100:.1f}%` | `{row['Laplacian_Var']}` | `{row['Risk_Category']}` |\n"

    md += """
---

### 🖼️ Mosaico Gráfico de Errores
El gráfico comparativo con las peores muestras ha sido generado en: [`models/error_mosaic.png`](models/error_mosaic.png).
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"--> Saved Markdown error summary to: {md_path}")

def plot_error_mosaic(x_test: np.ndarray, df_errors: pd.DataFrame, max_samples: int = 25):
    try:
        import matplotlib.pyplot as plt
        
        n_samples = min(len(df_errors), max_samples)
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        top_errors = df_errors.head(n_samples)
        
        for idx_plot, (_, row) in enumerate(top_errors.iterrows()):
            sample_idx = int(row["Sample_Idx"])
            img = x_test[sample_idx][:, :, 0]
            true_lbl = int(row["True_Label"])
            pred_lbl = int(row["Pred_Label"])
            conf = float(row["Confidence"])
            
            ax = axes[idx_plot]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            
            title_color = "darkred" if conf > 0.70 else "darkorange"
            ax.set_title(
                f"True:{true_lbl} -> Pred:{pred_lbl}\nConf:{conf*100:.0f}%",
                fontsize=9,
                fontweight="bold",
                color=title_color
            )
            
        # Turn off unused subplots
        for j in range(n_samples, len(axes)):
            axes[j].axis("off")
            
        plt.suptitle("Top Diagnóstico de Errores & Label Noise (TinyML INT8)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        mosaic_path = config.MODELS_DIR / "error_mosaic.png"
        plt.savefig(mosaic_path, dpi=150)
        print(f"--> Saved visual error mosaic to: {mosaic_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate visual mosaic plot: {e}")

if __name__ == "__main__":
    analyze_errors()
