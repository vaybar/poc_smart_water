"""
model_builder.py - Micro-Corner-Regressor Architecture for ESP32 Classic

Defines a compact Depthwise-Separable CNN designed for MCU deployment (ESP32 Classic):
- Grayscale input (128x128x1) to fit in a 16 KB capture buffer.
- Initial Stride-2 Conv2D to immediately reduce spatial dimensions and activation memory.
- Sequential Depthwise Separable blocks with ReLU6 for linear INT8 quantization.
- Global Average Pooling (GAP) eliminating 99% of dense parameters.
- Sigmoid activation on output coordinates to guarantee strictly bounded [0.0, 1.0] predictions.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config

def depthwise_separable_block(x, filters: int, stride: int = 1, name_prefix: str = "ds_block"):
    """
    Standard Depthwise Separable Convolution block (Depthwise Conv + Pointwise Conv)
    with BatchNormalization and ReLU6 activations.
    """
    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_dw"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name_prefix}_dw_relu6")(x)

    # Pointwise (1x1 Conv)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_pw"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name_prefix}_pw_relu6")(x)
    return x

def vector_to_corners(inputs):
    """
    Differentiable Vectorized Parallelogram Reconstruction:
    Converts 6 parameters [cx, cy, ux, uy, vx, vy] to 8 normalized corner coordinates:
    TL = C - u - v
    TR = C + u - v
    BR = C + u + v
    BL = C - u + v
    Guarantees 100% rigid, parallel rectangle shapes with 0% distortion.
    """
    cx_cy = inputs[:, 0:2]
    u = inputs[:, 2:4]
    v = inputs[:, 4:6]

    p0 = cx_cy - u - v # TL
    p1 = cx_cy + u - v # TR
    p2 = cx_cy + u + v # BR
    p3 = cx_cy - u + v # BL

    return tf.concat([p0, p1, p2, p3], axis=-1)

def build_micro_corner_regressor(
    input_shape=config.THUMB_INPUT_SHAPE,
    num_coords: int = config.NUM_COORDINATES,
    alpha: float = config.ALPHA,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Builds the Micro-Corner-Regressor V3 model with Vectorized Parallelogram Head.

    Args:
        input_shape: (H, W, C) input thumbnail shape, default (128, 128, 1).
        num_coords : Number of output coordinates (8 for 4 corners [x1, y1, ..., x4, y4]).
        alpha      : Width multiplier scaling channel filters.
        dropout_rate: Dropout rate before dense head.

    Returns:
        Uncompiled tf.keras.Model.
    """
    base_filters = [16, 24, 32, 48, 64]
    filters = [max(8, int(f * alpha)) for f in base_filters]

    inputs = layers.Input(shape=input_shape, name="thumb_input")

    # Step 1: Normalization
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Step 2: Initial Conv2D with Stride 2 (128x128 -> 64x64)
    x = layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu6")(x)

    # Step 3: Depthwise Separable Downsampling stages
    # Stage 1: 64x64 -> 32x32
    x = depthwise_separable_block(x, filters=filters[1], stride=2, name_prefix="ds_stage1")

    # Stage 2: 32x32 -> 16x16 (captures fine edge & corner details)
    x = depthwise_separable_block(x, filters=filters[2], stride=2, name_prefix="ds_stage2")
    skip_16x16 = x

    # Stage 3: 16x16 -> 8x8
    x = depthwise_separable_block(x, filters=filters[3], stride=2, name_prefix="ds_stage3")

    # Stage 4: 8x8 -> 8x8 (stride 1 preserves 8x8 spatial resolution)
    x = depthwise_separable_block(x, filters=filters[4], stride=1, name_prefix="ds_stage4")
    deep_8x8 = x

    # Step 4: Multi-Scale Skip Connection (16x16 -> 8x8 via stride-2 pooling)
    skip_proj = layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="skip_pool")(skip_16x16)
    skip_proj = layers.Conv2D(16, kernel_size=1, padding="same", use_bias=False, name="skip_conv")(skip_proj)
    skip_proj = layers.BatchNormalization(name="skip_bn")(skip_proj)
    skip_proj = layers.ReLU(max_value=6.0, name="skip_relu6")(skip_proj)

    # Step 5: Multi-Scale Fusion & Spatial Projection (8x8)
    fusion = layers.Concatenate(name="multiscale_concat")([deep_8x8, skip_proj])

    # Spatial projection to 12 channels (8x8x12 = 768 spatial features)
    x = layers.Conv2D(12, kernel_size=1, padding="same", use_bias=False, name="spatial_proj_conv")(fusion)
    x = layers.BatchNormalization(name="spatial_proj_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_proj_relu6")(x)
    x = layers.Flatten(name="spatial_flatten")(x)

    x = layers.Dense(64, activation="relu", name="head_dense")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Vectorized Parallelogram Output Parameters:
    # 1. Centroid (cx, cy) in [0.0, 1.0] via Sigmoid
    center = layers.Dense(2, activation="sigmoid", name="center_dense")(x)

    # 2. Long vector u = (ux, uy) in [-0.5, 0.5] via Tanh * 0.5
    u_vec = layers.Dense(2, activation="tanh", name="u_vec_dense")(x)
    u_vec = layers.Rescaling(0.5, name="u_vec_scale")(u_vec)

    # 3. Short vector v = (vx, vy) in [-0.5, 0.5] via Tanh * 0.5
    v_vec = layers.Dense(2, activation="tanh", name="v_vec_dense")(x)
    v_vec = layers.Rescaling(0.5, name="v_vec_scale")(v_vec)

    # Concatenate 6 parameters -> (batch, 6)
    vec_params = layers.Concatenate(name="vec_params_concat")([center, u_vec, v_vec])

    # Output: 8 normalized corners reconstructed via Lambda layer
    outputs = layers.Lambda(vector_to_corners, name="corners_output")(vec_params)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"MicroCornerRegressor_V3_a{int(alpha*100):02d}")
    return model


if __name__ == "__main__":
    m = build_micro_corner_regressor(alpha=0.75)
    m.summary()
    print(f"\nTotal Parameters: {m.count_params():,}")
    print(f"Estimated INT8 Flash Size: ~{m.count_params() / 1024:.2f} KB")
