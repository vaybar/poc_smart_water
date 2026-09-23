"""
model_builder.py - Micro-Corner-Regressor V5 Architecture for ESP32 Classic

V5 changes from V4:
- CoordConv: Injects normalized (x, y) coordinate channels into the 8×8 feature map,
  preserving spatial position information that Flatten destroys.
- Center+Offsets output: Predicts center (cx, cy) + 4 relative offsets instead of
  8 independent absolute coordinates. This couples corner predictions and prevents
  the systematic quad inflation observed in EXP_007 diagnostics.
- Reconstruction layer converts center+offsets back to 8 absolute coordinates,
  maintaining full backward compatibility with existing loss functions and evaluation.

Architecture: V2 backbone (unchanged) → CoordConv → Conv(1×1) → Flatten → Dense → Reconstruct
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config


# ---------------------------------------------------------------------------
#  CoordConv Layer — injects spatial position into feature maps
# ---------------------------------------------------------------------------
class CoordConv2D(layers.Layer):
    """
    Adds normalized (x, y) coordinate channels to the input feature map.

    Given input of shape (batch, H, W, C), outputs (batch, H, W, C+2) where
    the two extra channels contain x ∈ [0,1] and y ∈ [0,1] grids.

    This injects explicit spatial position information, enabling the model to
    learn position-dependent features without losing spatial reference after Flatten.

    Reference: Liu et al., "An Intriguing Failing of Convolutional Neural Networks
    and the CoordConv Solution", NeurIPS 2018.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        h, w = int(input_shape[1]), int(input_shape[2])
        x_coords = tf.linspace(0.0, 1.0, w)
        y_coords = tf.linspace(0.0, 1.0, h)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        # Store as (1, H, W, 2) non-trainable constant
        self.coord_grid = self.add_weight(
            name="coord_grid",
            shape=(1, h, w, 2),
            initializer=tf.keras.initializers.Constant(
                tf.stack([x_grid, y_grid], axis=-1)[tf.newaxis, ...].numpy()
            ),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        coords = tf.tile(self.coord_grid, [batch_size, 1, 1, 1])
        return tf.concat([inputs, coords], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] + 2)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
#  Center + Offset Reconstruction Layer
# ---------------------------------------------------------------------------
class CenterOffsetReconstruction(layers.Layer):
    """
    Reconstructs 8 absolute corner coordinates from center + offset representation.

    Input:  (batch, 10) raw logits — [raw_cx, raw_cy, raw_dx1, raw_dy1, ..., raw_dx4, raw_dy4]
    Output: (batch, 8) absolute coords — [x1, y1, x2, y2, x3, y3, x4, y4] in [0, 1]

    Internal activations:
    - Center (cx, cy): sigmoid → [0, 1]
    - Offsets (dx_i, dy_i): tanh × max_offset → [-max_offset, +max_offset]

    The gradient from each corner flows back to BOTH the center AND its offset,
    causing the center to naturally learn the quad centroid while offsets learn shape.
    """
    def __init__(self, max_offset: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.max_offset = max_offset

    def call(self, x):
        center = tf.sigmoid(x[:, :2])                    # (batch, 2) in [0, 1]
        offsets = tf.tanh(x[:, 2:]) * self.max_offset     # (batch, 8) in [-0.35, 0.35]

        cx = center[:, 0:1]
        cy = center[:, 1:2]

        corners = tf.concat([
            cx + offsets[:, 0:1], cy + offsets[:, 1:2],   # TL (x1, y1)
            cx + offsets[:, 2:3], cy + offsets[:, 3:4],   # TR (x2, y2)
            cx + offsets[:, 4:5], cy + offsets[:, 5:6],   # BR (x3, y3)
            cx + offsets[:, 6:7], cy + offsets[:, 7:8],   # BL (x4, y4)
        ], axis=1)

        return tf.clip_by_value(corners, 0.0, 1.0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 8)

    def get_config(self):
        base = super().get_config()
        base.update({"max_offset": self.max_offset})
        return base


# ---------------------------------------------------------------------------
#  Depthwise Separable Block (unchanged from V4)
# ---------------------------------------------------------------------------
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


def build_micro_corner_regressor(
    input_shape=config.THUMB_INPUT_SHAPE,
    num_coords: int = config.NUM_COORDINATES,
    alpha: float = config.ALPHA,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Builds the Micro-Corner-Regressor V5 model with CoordConv + Center+Offsets.

    Backbone: V2 architecture with multi-scale skip connections (best from EXP_005).
    Head: CoordConv → Conv(1×1) → Flatten → Dense → CenterOffsetReconstruction
          Output is 8 absolute corner coordinates [x1,y1,...,x4,y4] in [0,1].

    Key improvements over V4:
    - CoordConv injects (x,y) position into the 8×8 feature map before Flatten
    - Center+Offsets couples all 4 corners through a shared center prediction
    - Reduced Flatten size (512 vs 768) due to channel reduction

    Args:
        input_shape: (H, W, C) input thumbnail shape, default (128, 128, 1).
        num_coords : Number of output coordinates (8 for 4 corners, reconstructed internally).
        alpha      : Width multiplier scaling channel filters.
        dropout_rate: Dropout rate before dense head.

    Returns:
        Uncompiled tf.keras.Model with output shape (batch, 8).
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

    # Step 5: Multi-Scale Fusion (8x8)
    fusion = layers.Concatenate(name="multiscale_concat")([deep_8x8, skip_proj])

    # Spatial projection to 12 channels
    x = layers.Conv2D(12, kernel_size=1, padding="same", use_bias=False, name="spatial_proj_conv")(fusion)
    x = layers.BatchNormalization(name="spatial_proj_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_proj_relu6")(x)

    # ---- V5 NEW: CoordConv + Center-Offset Head ----

    # Step 6: CoordConv — inject (x, y) position channels → 8×8×14
    x = CoordConv2D(name="coord_conv")(x)

    # Step 7: Mix coordinate info with features → 8×8×8
    x = layers.Conv2D(8, kernel_size=1, padding="same", use_bias=False, name="coord_mix_conv")(x)
    x = layers.BatchNormalization(name="coord_mix_bn")(x)
    x = layers.ReLU(max_value=6.0, name="coord_mix_relu6")(x)

    # Step 8: Flatten spatial features → 512
    x = layers.Flatten(name="spatial_flatten")(x)

    # Step 9: Dense regression head
    x = layers.Dense(64, activation="relu", name="head_dense")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Step 10: Raw output — 10 logits (2 center + 8 offsets)
    raw = layers.Dense(10, name="raw_center_offsets")(x)

    # Step 11: Reconstruct 8 absolute corner coordinates from center + offsets
    # Output shape: (batch, 8) in [0, 1] — fully compatible with existing losses
    outputs = CenterOffsetReconstruction(
        max_offset=config.MAX_QUAD_OFFSET,
        name="corners_output"
    )(raw)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"MicroCornerRegressor_V5_a{int(alpha*100):02d}"
    )
    return model


if __name__ == "__main__":
    m = build_micro_corner_regressor(alpha=0.75)
    m.summary()
    print(f"\nTotal Parameters: {m.count_params():,}")
    print(f"Estimated INT8 Flash Size: ~{m.count_params() / 1024:.2f} KB")
