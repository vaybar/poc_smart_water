"""
test_inference.py - Python Microcontroller Inference Simulator

Simulates TFLite Micro edge inference on single image files or test datasets.
Measures latency, verifies int8 quantization scaling, and displays prediction probabilities.
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

import config
from dataset import load_digit_dataset, preprocess_digit_image

def run_inference_single_image(image_path: str):
    """
    Run INT8 TFLite inference on a single image file.
    """
    if not config.TFLITE_INT8_PATH.exists():
        raise FileNotFoundError(f"Quantized model not found at {config.TFLITE_INT8_PATH}. Run quantize_and_export.py first!")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    proc_img = preprocess_digit_image(img)  # (32, 32, 1) uint8
    
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(config.TFLITE_INT8_PATH))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    # Quantize input: int8 = (float_val / scale) + zero_point
    img_float = proc_img.astype(np.float32)
    img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
    img_input = np.expand_dims(img_quant, axis=0)
    
    # Measure latency
    t0 = time.perf_counter()
    interpreter.set_tensor(input_details['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000.0
    
    # De-quantize output predictions to probabilities
    probabilities = (output_data[0].astype(np.float32) - output_zero_point) * output_scale
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    print("\n" + "=" * 50)
    print(f"  TinyML Digit Inference Result for: {image_path}")
    print("=" * 50)
    print(f" Predicted Digit : {predicted_digit}")
    print(f" Confidence      : {confidence * 100:.2f}%")
    print(f" Inference Time  : {latency_ms:.3f} ms")
    print("-" * 50)
    print(" Class Probabilities:")
    for digit, prob in enumerate(probabilities):
        bar = "#" * int(prob * 30)
        print(f"   Digit {digit}: {prob*100:5.1f}% | {bar}")
    print("=" * 50)
    
    return predicted_digit, confidence

def benchmark_test_set():
    """
    Run INT8 benchmark across the entire test set.
    """
    if not config.TFLITE_INT8_PATH.exists():
        raise FileNotFoundError(f"Quantized model not found at {config.TFLITE_INT8_PATH}")
        
    _, _, (x_test, y_test) = load_digit_dataset()
    
    interpreter = tf.lite.Interpreter(model_path=str(config.TFLITE_INT8_PATH))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    latencies = []
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        img_float = x_test[i].astype(np.float32)
        img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
        img_input = np.expand_dims(img_quant, axis=0)
        
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], img_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        pred = np.argmax(output_data[0])
        if pred == y_test[i]:
            correct += 1
            
    avg_latency = np.mean(latencies)
    accuracy = (correct / total) * 100.0
    
    print("\n" + "=" * 50)
    print("  TinyML INT8 Model Benchmark Summary")
    print("=" * 50)
    print(f" Test Set Accuracy : {accuracy:.2f}% ({correct}/{total})")
    print(f" Avg Latency / Img : {avg_latency:.3f} ms")
    print(f" Throughput        : {1000.0 / avg_latency:.1f} FPS")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TinyML Digit Classifier Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image file")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark on test dataset")
    args = parser.parse_args()
    
    if args.image:
        run_inference_single_image(args.image)
    else:
        benchmark_test_set()
