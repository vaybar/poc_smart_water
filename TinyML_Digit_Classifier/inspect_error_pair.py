"""
inspect_error_pair.py - Visual Inspector for Specific Misclassification Pairs

Allows inspecting and plotting all test set images for a specific error pair
(e.g., True Label = 1, Predicted Label = 0).

Usage:
  python inspect_error_pair.py --true_label 1 --pred_label 0
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config
from dataset import load_digit_dataset

def inspect_pair(true_label: int = 1, pred_label: int = 0):
    print("=" * 65)
    print(f"  Visualizing Error Pair: True '{true_label}' -> Predicted '{pred_label}'")
    print("=" * 65)
    
    # 1. Load Test Dataset
    _, _, (x_test, y_test) = load_digit_dataset()
    print(f"Loaded test dataset: {len(x_test)} images")
    
    # 2. Check TFLite Model
    if not config.TFLITE_INT8_PATH.exists():
        print(f"Error: Model not found at {config.TFLITE_INT8_PATH}")
        return
        
    tflite_bytes = open(config.TFLITE_INT8_PATH, "rb").read()
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    # 3. Find matching samples
    matches = []
    probs_list = []
    
    for idx in range(len(x_test)):
        if y_test[idx] == true_label:
            img_float = x_test[idx].astype(np.float32)
            img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
            img_input = np.expand_dims(img_quant, axis=0)
            
            interpreter.set_tensor(input_details['index'], img_input)
            interpreter.invoke()
            out_raw = interpreter.get_tensor(output_details['index'])[0]
            
            # De-quantize output and compute Softmax
            out_dequant = (out_raw.astype(np.float32) - output_zero_point) * output_scale
            exp_scores = np.exp(out_dequant - np.max(out_dequant))
            prob = exp_scores / np.sum(exp_scores)
            
            predicted = np.argmax(prob)
            if predicted == pred_label:
                matches.append(idx)
                probs_list.append((prob[true_label], prob[pred_label]))
                
    print(f"\n--> Found {len(matches)} matching samples:")
    for idx, (p_true, p_pred) in zip(matches, probs_list):
        print(f"  - Sample #{idx:04d} | Prob(True={true_label}): {p_true*100:.1f}% | Prob(Pred={pred_label}): {p_pred*100:.1f}%")
        
    if not matches:
        print("No misclassified samples found for this pair.")
        return

    # 4. Render and Save Plot
    n_samples = len(matches)
    cols = min(n_samples, 5)
    rows = int(np.ceil(n_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (sample_idx, (p_true, p_pred)) in enumerate(zip(matches, probs_list)):
        ax = axes[i]
        img_2d = x_test[sample_idx][:, :, 0]
        ax.imshow(img_2d, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"Sample #{sample_idx}\nReal:{true_label} ({p_true*100:.1f}%)\nPred:{pred_label} ({p_pred*100:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color="darkred"
        )
        
    for j in range(n_samples, len(axes)):
        axes[j].axis("off")
        
    plt.suptitle(
        f"Misclassifications: True Digit '{true_label}' Predicted as '{pred_label}' (TinyML INT8)",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    
    out_plot_path = config.MODELS_DIR / f"error_pair_True{true_label}_Pred{pred_label}.png"
    plt.savefig(out_plot_path, dpi=150)
    print(f"\n--> Saved error visualization plot to: {out_plot_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect specific misclassification error pairs")
    parser.add_argument("--true_label", type=int, default=1, help="True ground truth digit label (0-9)")
    parser.add_argument("--pred_label", type=int, default=0, help="Predicted digit label (0-9)")
    args = parser.parse_args()
    
    inspect_pair(true_label=args.true_label, pred_label=args.pred_label)
