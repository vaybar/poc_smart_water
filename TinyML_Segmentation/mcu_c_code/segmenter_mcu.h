/**
 * segmenter_mcu.h - MCU-Native Water Meter Segmentation for ESP32 Classic
 * 
 * Provides pure C routines (zero OpenCV dependencies) for:
 * 1. Downsampling camera buffer (e.g. QVGA/VGA) to 128x128 Grayscale.
 * 2. Bilinear Quadrilateral Rectification of tilted counter into a 64x(N*32) strip.
 * 3. Fixed slot slicing into 64x32 digit tiles for TinyML_Digit_Classifier.
 */

#ifndef SEGMENTER_MCU_H_
#define SEGMENTER_MCU_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SEG_THUMB_W         128
#define SEG_THUMB_H         128
#define SEG_DIGIT_W         32
#define SEG_DIGIT_H         64
#define SEG_DEFAULT_DIGITS  6

typedef struct {
    float x;
    float y;
} Point2D;

typedef struct {
    Point2D p0; // Top-Left
    Point2D p1; // Top-Right
    Point2D p2; // Bottom-Right
    Point2D p3; // Bottom-Left
} CounterQuad;

/**
 * Downsamples a grayscale camera framebuffer into a 128x128 thumbnail
 * for the Micro-Corner-Regressor model.
 */
void downsample_to_128x128_gray(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst_thumb128
);

/**
 * Bilinear quadrilateral inverse mapping.
 * Warps the tilted counter box defined by 'corners' into a flat, horizontal
 * rectified strip of size: target_w x target_h (typically [N*32] x 64).
 */
void bilinear_warp_counter_strip(
    const uint8_t* src_gray,
    int src_w,
    int src_h,
    const CounterQuad* corners,
    uint8_t* dst_strip,
    int target_w,
    int target_h
);

/**
 * Slices a single digit tile (64x32 uint8) from the rectified strip.
 * digit_index is in range [0, num_digits - 1].
 */
void slice_digit_tile(
    const uint8_t* strip,
    int strip_w,
    int strip_h,
    int digit_index,
    uint8_t* dst_digit_tile64x32
);

#ifdef __cplusplus
}
#endif

#endif // SEGMENTER_MCU_H_
