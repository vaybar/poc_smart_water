"""
quantize_and_export.py - INT8 Post-Training Quantization & C Header Exporter

Converts float32 Keras model to full INT8 TFLite model using representative
calibration data. Evaluates file footprint (< 256 KB Flash), estimates RAM
Tensor Arena consumption, and generates C header file (digit_model_quantized.h)
for immediate inclusion in ESP32-S3 Arduino / ESP-IDF projects.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import config
from dataset import load_digit_dataset, get_representative_dataset_generator

def estimate_tensor_arena_size(interpreter: tf.lite.Interpreter) -> int:
    """
    Estimates maximum RAM required for TFLite Micro Tensor Arena
    by computing cumulative sizes of intermediate tensor activations.
    """
    tensor_details = interpreter.get_tensor_details()
    max_arena = 0
    for detail in tensor_details:
        shape = detail['shape']
        dtype_size = np.dtype(detail['dtype']).itemsize
        num_elements = int(np.prod(shape))
        tensor_bytes = num_elements * dtype_size
        max_arena += tensor_bytes
    # TFLite Micro reuses buffers, so peak memory is roughly 25-40% of total tensor sum + safety margin
    estimated_arena = int(max_arena * 0.35) + 4096
    return estimated_arena

def convert_to_c_header(tflite_bytes: bytes, output_header_path: Path):
    """
    Converts binary TFLite model byte sequence to a C array header file (.h).
    """
    hex_lines = []
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i+12]
        hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
        hex_lines.append("  " + hex_str)
        
    hex_content = ",\n".join(hex_lines)
    c_content = f"""/*
 * Auto-generated Micro-MobileNet Quantized INT8 Model Header for ESP32-S3 / TinyML
 * Model size: {len(tflite_bytes):,} bytes ({len(tflite_bytes)/1024:.2f} KB)
 */

#ifndef DIGIT_MODEL_QUANTIZED_H
#define DIGIT_MODEL_QUANTIZED_H

#ifdef __has_include
  #if __has_include(<pgmspace.h>)
    #include <pgmspace.h>
  #endif
#endif

#ifndef PROGMEM
  #define PROGMEM
#endif

alignas(8) const unsigned char g_digit_model[] PROGMEM = {{
{hex_content}
}};

const unsigned int g_digit_model_len = {len(tflite_bytes)};

#endif // DIGIT_MODEL_QUANTIZED_H
"""
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write(c_content)
    print(f"--> Exported C Header to: {output_header_path}")

def quantize_and_export():
    print("=" * 60)
    print("  Full INT8 Post-Training Quantization & C Header Export")
    print("=" * 60)
    
    if not config.FLOAT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Float32 model not found at {config.FLOAT_MODEL_PATH}. Run train.py first!"
        )
        
    # 1. Load float model
    print(f"Loading Keras float model from: {config.FLOAT_MODEL_PATH}")
    model = tf.keras.models.load_model(str(config.FLOAT_MODEL_PATH))
    
    # 2. Load calibration data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_digit_dataset()
    
    # 3. Setup TFLite Converter for Full INT8 Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset calibration generator
    rep_gen = get_representative_dataset_generator(x_train, num_samples=150)
    converter.representative_dataset = rep_gen
    
    # Enforce INT8 operations for full micro hardware acceleration
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Running INT8 quantization and conversion...")
    tflite_quant_model = converter.convert()
    
    # Save .tflite binary
    with open(config.TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_quant_model)
        
    file_size_bytes = len(tflite_quant_model)
    file_size_kb = file_size_bytes / 1024.0
    print(f"\n--> INT8 TFLite Model Size: {file_size_bytes:,} bytes ({file_size_kb:.2f} KB)")
    
    # Check against memory budget
    if file_size_bytes <= config.MAX_FLASH_BYTES:
        print(f"  [SUCCESS] Model footprint ({file_size_kb:.1f} KB) is well within Flash budget ({config.MAX_FLASH_BYTES/1024:.0f} KB)!")
    else:
        print(f"  [WARNING] Model size ({file_size_kb:.1f} KB) exceeds target {config.MAX_FLASH_BYTES/1024:.0f} KB budget.")

    # 4. Evaluate INT8 accuracy & Tensor Arena
    print("\nEvaluating INT8 Model Accuracy with TFLite Interpreter...")
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        # Convert float image to int8: q = (float / scale) + zero_point
        img_float = x_test[i].astype(np.float32)
        img_quant = (img_float / input_scale + input_zero_point).astype(np.int8)
        img_input = np.expand_dims(img_quant, axis=0)
        
        interpreter.set_tensor(input_details['index'], img_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details['index'])
        pred_label = np.argmax(output_data[0])
        if pred_label == y_test[i]:
            correct += 1
            
    acc_int8 = (correct / total) * 100.0
    print(f"--> INT8 Quantized Accuracy on Test Set: {acc_int8:.2f}% ({correct}/{total})")
    
    arena_ram_est = estimate_tensor_arena_size(interpreter)
    print(f"--> Estimated Tensor Arena RAM needed: ~{arena_ram_est / 1024:.2f} KB (Target < 40 KB)")
    
    # 5. Convert to C Header File
    convert_to_c_header(tflite_quant_model, config.C_HEADER_PATH)
    print("=" * 60)
    print("Quantization & Export complete!")

if __name__ == "__main__":
    quantize_and_export()
