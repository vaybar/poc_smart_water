/*
 * esp32_s3_digit_classifier.ino
 *
 * TinyML Ultra-Lightweight Digit Classifier for ESP32-S3
 * Model Memory Footprint: < 120 KB Flash, < 35 KB RAM Tensor Arena
 *
 * Requirements for Arduino IDE:
 * 1. Install "TensorFlowLite_ESP32" or "tflite-micro" library
 * 2. Select Board: ESP32S3 Dev Module
 * 3. Copy models/digit_model_quantized.h into your sketch directory
 */

#include <Arduino.h>
#include "digit_model_quantized.h"  // Auto-generated C header byte array

// TFLite Micro Headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Memory Allocation Budget for ESP32-S3
// Micro-MobileNet 32x32x1 requires ~28 KB to 35 KB RAM for Tensor Arena
constexpr int kTensorArenaSize = 35 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLite Global Variables
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  tflite::AllOpsResolver resolver;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("\n==============================================");
  Serial.println("   ESP32-S3 TinyML Micro-MobileNet Digit AI   ");
  Serial.println("==============================================");

  // 1. Load Model from PROGMEM Flash Array
  model = tflite::GetModel(g_digit_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version mismatch! Expected %d, got %d\n",
                  TFLITE_SCHEMA_VERSION, model->version());
    return;
  }
  Serial.printf("Model successfully loaded from Flash! (%d bytes)\n", g_digit_model_len);

  // 2. Build Micro Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 3. Allocate Tensors in Tensor Arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed! Increase kTensorArenaSize.");
    return;
  }
  Serial.printf("Tensor Arena allocated: %d bytes RAM\n", kTensorArenaSize);

  // 4. Obtain Input and Output Pointers
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("Input Shape: %d x %d x %d (Type: %d)\n",
                input->dims->data[1], input->dims->data[2], input->dims->data[3], input->type);
  Serial.printf("Output Classes: %d\n", output->dims->data[1]);
  Serial.println("ESP32-S3 Initialization Complete. Ready for Inference!\n");
}

void loop() {
  // Dummy test inference generator for demonstration
  Serial.println("--- Running Digit Classification ---");
  
  // Fill input buffer with dummy grayscale 32x32 image (or stream camera frame)
  int8_t* input_buffer = input->data.int8;
  for (int i = 0; i < input->bytes; i++) {
    input_buffer[i] = (int8_t)(random(0, 255) - 128); // INT8 normalized pixel
  }

  // Measure Inference Time on ESP32-S3 (cycles/microsecond accuracy)
  unsigned long start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long duration_us = micros() - start_time;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Inference invoke failed!");
    delay(2000);
    return;
  }

  // Read Prediction Output
  int8_t* output_buffer = output->data.int8;
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  int max_digit = 0;
  float max_prob = -1.0f;

  for (int digit = 0; digit < 10; digit++) {
    float prob = (output_buffer[digit] - zero_point) * scale;
    if (prob > max_prob) {
      max_prob = prob;
      max_digit = digit;
    }
  }

  Serial.printf("Predicted Digit : %d\n", max_digit);
  Serial.printf("Confidence      : %.2f%%\n", max_prob * 100.0f);
  Serial.printf("Inference Latency: %.2f ms (ESP32-S3 @ 240MHz)\n", duration_us / 1000.0f);
  Serial.println("----------------------------------------------\n");

  delay(3000); // Repeat every 3 seconds
}
