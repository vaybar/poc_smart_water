"""
model_builder.py - Ultra-lightweight Micro-MobileNet Model Construction

Implements a highly scalable Micro-MobileNet architecture specifically optimized
for low-power microcontrollers (ESP32-S3, STM32, Cortex-M).

Key Design Choices for Edge AI (< 256 KB Flash, < 40 KB RAM):
1. Depthwise Separable Convolutions (spatial conv + 1x1 pointwise conv)
2. Parametric Width Multiplier (Alpha) for channel scaling
3. Global Average Pooling (eliminates parameter-heavy FC layers)
4. ReLU / ReLU6 activations for optimal INT8 quantization mapping
5. Single-channel grayscale input (32x32x1) reducing Tensor Arena size by 3x
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config

def depthwise_separable_conv_block(x, pointwise_filters, alpha=1.0, stride=1, block_id=1):
    """
    Depthwise Separable Convolution block:
    - Depthwise 3x3 Conv (spatial filtering per channel)
    - Batch Normalization + ReLU6
    - Pointwise 1x1 Conv (linear channel combination)
    - Batch Normalization + ReLU6
    """
    filters = max(8, int(pointwise_filters * alpha))
    
    # 1. Depthwise Convolution
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False,
        name=f"conv_dw_{block_id}"
    )(x)
    x = layers.BatchNormalization(name=f"conv_dw_{block_id}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"conv_dw_{block_id}_relu")(x)

    # 2. Pointwise Convolution (1x1)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"conv_pw_{block_id}"
    )(x)
    x = layers.BatchNormalization(name=f"conv_pw_{block_id}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"conv_pw_{block_id}_relu")(x)
    
    return x

def build_micro_mobilenet(
    input_shape=config.INPUT_SHAPE,
    num_classes=config.NUM_CLASSES,
    alpha=config.ALPHA,
    dropout_rate=0.2
) -> tf.keras.Model:
    """
    Builds and returns a Micro-MobileNet Keras model optimized for TinyML.
    
    Parameters:
        input_shape: Tuple (H, W, C), default (32, 32, 1)
        num_classes: Int, default 10
        alpha: Float width multiplier (e.g. 0.10, 0.25, 0.35, 0.50)
        dropout_rate: Float dropout before classification layer
        
    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Normalization layer: scale [0, 255] to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    
    # Initial Standard Conv2D (stride 2 to downsample early and save RAM)
    init_filters = max(8, int(16 * alpha))
    x = layers.Conv2D(
        filters=init_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv1"
    )(x)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(max_value=6.0, name="conv1_relu")(x)
    
    # Stack of Depthwise Separable Blocks
    # Block 1: 16x16 -> 16x16
    x = depthwise_separable_conv_block(x, pointwise_filters=16, alpha=alpha, stride=1, block_id=1)
    
    # Block 2: 16x16 -> 8x8
    x = depthwise_separable_conv_block(x, pointwise_filters=32, alpha=alpha, stride=2, block_id=2)
    
    # Block 3: 8x8 -> 8x8
    x = depthwise_separable_conv_block(x, pointwise_filters=32, alpha=alpha, stride=1, block_id=3)
    
    # Block 4: 8x8 -> 4x4
    x = depthwise_separable_conv_block(x, pointwise_filters=64, alpha=alpha, stride=2, block_id=4)
    
    # Block 5: 4x4 -> 4x4
    x = depthwise_separable_conv_block(x, pointwise_filters=64, alpha=alpha, stride=1, block_id=5)
    
    # Global Average Pooling (removes need for huge dense layers)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
        
    # Final Dense Classifier
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f"MicroMobileNet_alpha{alpha:.2f}")
    return model

if __name__ == "__main__":
    m = build_micro_mobilenet(alpha=0.25)
    m.summary()
    print(f"Total params: {m.count_params():,}")
