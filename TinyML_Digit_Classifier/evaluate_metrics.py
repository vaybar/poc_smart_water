"""
evaluate_metrics.py - Comprehensive Metrics & Evaluation Script

Generates full evaluation metrics for both Float32 and Quantized INT8 models:
1. Overall Accuracy & Loss
2. Per-class Precision, Recall, F1-Score (Classification Report)
3. Confusion Matrix (Text Grid)
4. Float32 vs INT8 Quantization Degradation Analysis
5. Model Footprint (Flash KB) & RAM Tensor Arena Estimation
6. Inference Latency & Throughput Benchmark
"""

import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from dataset import load_digit_dataset

def evaluate_full_metrics():
    print("=" * 65)
    print("      TinyML Digit Classifier - Comprehensive Model Metrics")
    print("=" * 65)
    
    # 1. Load Test Dataset
    _, _, (x_test, y_test) = load_digit_dataset()
    print(f"Test dataset size: {len(x_test)} images\n")
    
    # -------------------------------------------------------------
    # EVALUATION 1: Float32 Model
    # -------------------------------------------------------------
    if config.FLOAT_MODEL_PATH.exists():
        print("--- [1/2] Evaluating Float32 Model ---")
        float_model = tf.keras.models.load_model(str(config.FLOAT_MODEL_PATH))
        
        t0 = time.perf_counter()
        f32_preds_prob = float_model.predict(x_test, verbose=0)
        t1 = time.perf_counter()
        
        f32_preds = np.argmax(f32_preds_prob, axis=1)
        f32_acc = np.mean(f32_preds == y_test) * 100.0
        f32_latency = ((t1 - t0) / len(x_test)) * 1000.0
        
        print(f" Float32 Test Accuracy: {f32_acc:.2f}%")
        print(f" Float32 Avg Latency  : {f32_latency:.3f} ms / image\n")
    else:
        print("Float32 model checkpoint not found. Skipping Float32 evaluation.")
        f32_acc = None

    # -------------------------------------------------------------
    # EVALUATION 2: INT8 Quantized Model
    # -------------------------------------------------------------
    if not config.TFLITE_INT8_PATH.exists():
        print(f"Quantized INT8 model not found at {config.TFLITE_INT8_PATH}. Run quantize_and_export.py first!")
        return

    print("--- [2/2] Evaluating Quantized INT8 TFLite Model ---")
    tflite_bytes = open(config.TFLITE_INT8_PATH, "rb").read()
    flash_size_kb = len(tflite_bytes) / 1024.0
    
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    int8_preds = []
    latencies = []
    
    for i in range(len(x_test)):
        img_float = x_test[i].astype(np.float32)
        img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
        img_input = np.expand_dims(img_quant, axis=0)
        
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], img_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        pred_label = np.argmax(output_data[0])
        int8_preds.append(pred_label)
        
    int8_preds = np.array(int8_preds)
    int8_acc = np.mean(int8_preds == y_test) * 100.0
    avg_latency = np.mean(latencies)
    
    # -------------------------------------------------------------
    # METRICS SUMMARY & REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("                    FINAL METRICS SUMMARY")
    print("=" * 65)
    print(f" Flash Memory Size (TFLite INT8) : {flash_size_kb:.2f} KB  (Target < 256 KB)")
    print(f" INT8 Model Test Accuracy       : {int8_acc:.2f}%")
    if f32_acc is not None:
        drop = f32_acc - int8_acc
        print(f" Quantization Accuracy Loss      : {drop:+.2f}%  (Float32: {f32_acc:.2f}% -> INT8: {int8_acc:.2f}%)")
    print(f" Avg Inference Latency (Python)  : {avg_latency:.3f} ms / image ({1000.0/avg_latency:.1f} FPS)")
    print("=" * 65)
    
    print("\n--- PER-CLASS CLASSIFICATION REPORT (INT8) ---")
    target_names = [f"Digit '{d}'" for d in range(10)]
    report_text = classification_report(y_test, int8_preds, target_names=target_names, digits=3)
    report_dict = classification_report(y_test, int8_preds, output_dict=True)
    print(report_text)
    
    print("--- CONFUSION MATRIX (Rows: True, Cols: Predicted) ---")
    cm = confusion_matrix(y_test, int8_preds)
    header = "True \\ Pred | " + " ".join([f"{d:4d}" for d in range(10)])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:4d}" for val in row])
        print(f" Digit '{i}'  | {row_str}")
    print("=" * 65)

    # -------------------------------------------------------------
    # LOG TO BITÁCORA & PER-EXPERIMENT REPORTS
    # -------------------------------------------------------------
    exp_id = None
    try:
        from experiment_logger import log_experiment
        # Estimate arena RAM: ~0.3 of model size + 10KB
        arena_est_kb = (len(tflite_bytes) * 0.3) / 1024.0 + 10.0
        exp_id = log_experiment(
            alpha=config.ALPHA,
            input_shape=f"{config.IMG_H}x{config.IMG_W}x{config.CHANNELS}",
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            float_acc=f32_acc,
            int8_acc=int8_acc,
            flash_kb=flash_size_kb,
            arena_kb=arena_est_kb,
            latency_ms=avg_latency,
            classification_report_dict=report_dict,
            confusion_matrix_arr=cm,
            notes=f"Entrenamiento con alpha={config.ALPHA}"
        )
    except Exception as e:
        print(f"Could not update experiment log: {e}")

    # Save visual Confusion Matrix plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        plt.title(f"Matriz de Confusión - TinyML INT8 ({exp_id if exp_id else 'Evaluation'})", pad=20, fontsize=12, fontweight="bold")
        plt.xlabel("Predicho (Predicted)", fontsize=10)
        plt.ylabel("Real (True)", fontsize=10)
        plt.xticks(range(10), range(10))
        plt.yticks(range(10), range(10))
        
        # Annotate numbers inside matrix cells
        for i in range(10):
            for j in range(10):
                color = "white" if cm[i, j] > np.max(cm) / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontweight="bold")
                
        plot_path = (config.EXPERIMENTS_DIR / exp_id / "confusion_matrix.png") if exp_id else (config.MODELS_DIR / "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"--> Visual Confusion Matrix plot saved to: {plot_path}")
        plt.close(fig)
    except Exception as err:
        print(f"Note: Could not render plot ({err})")

if __name__ == "__main__":
    evaluate_full_metrics()
