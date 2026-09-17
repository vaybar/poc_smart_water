/**
 * segmenter_mcu.c - Implementation of MCU-Native Vision Operations for ESP32
 */

#include "segmenter_mcu.h"
#include <math.h>
#include <string.h>

void downsample_to_128x128_gray(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst_thumb128
) {
    if (!src || !dst_thumb128 || src_w <= 0 || src_h <= 0) return;

    // Fast nearest-neighbor or 2x2 box downsampling
    float step_x = (float)src_w / (float)SEG_THUMB_W;
    float step_y = (float)src_h / (float)SEG_THUMB_H;

    for (int y = 0; y < SEG_THUMB_H; y++) {
        int src_y = (int)(y * step_y);
        if (src_y >= src_h) src_y = src_h - 1;
        const uint8_t* row_ptr = &src[src_y * src_w];
        uint8_t* dst_ptr = &dst_thumb128[y * SEG_THUMB_W];

        for (int x = 0; x < SEG_THUMB_W; x++) {
            int src_x = (int)(x * step_x);
            if (src_x >= src_w) src_x = src_w - 1;
            dst_ptr[x] = row_ptr[src_x];
        }
    }
}

static inline uint8_t bilinear_sample(
    const uint8_t* src,
    int w,
    int h,
    float fx,
    float fy
) {
    if (fx < 0.0f) fx = 0.0f;
    if (fy < 0.0f) fy = 0.0f;
    if (fx >= (float)(w - 1)) fx = (float)(w - 1);
    if (fy >= (float)(h - 1)) fy = (float)(h - 1);

    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= w) x1 = w - 1;
    if (y1 >= h) y1 = h - 1;

    float dx = fx - (float)x0;
    float dy = fy - (float)y0;

    float p00 = (float)src[y0 * w + x0];
    float p10 = (float)src[y0 * w + x1];
    float p01 = (float)src[y1 * w + x0];
    float p11 = (float)src[y1 * w + x1];

    float val = (1.0f - dx) * (1.0f - dy) * p00 +
                dx * (1.0f - dy) * p10 +
                (1.0f - dx) * dy * p01 +
                dx * dy * p11;

    if (val < 0.0f) val = 0.0f;
    if (val > 255.0f) val = 255.0f;
    return (uint8_t)(val + 0.5f);
}

void bilinear_warp_counter_strip(
    const uint8_t* src_gray,
    int src_w,
    int src_h,
    const CounterQuad* corners,
    uint8_t* dst_strip,
    int target_w,
    int target_h
) {
    if (!src_gray || !corners || !dst_strip || target_w <= 1 || target_h <= 1) return;

    float inv_w = 1.0f / (float)(target_w - 1);
    float inv_h = 1.0f / (float)(target_h - 1);

    float p0x = corners->p0.x, p0y = corners->p0.y;
    float p1x = corners->p1.x, p1y = corners->p1.y;
    float p2x = corners->p2.x, p2y = corners->p2.y;
    float p3x = corners->p3.x, p3y = corners->p3.y;

    for (int v = 0; v < target_h; v++) {
        float t = (float)v * inv_h;
        float omt = 1.0f - t;

        // Precompute edge interpolations for this row
        float left_x  = omt * p0x + t * p3x;
        float left_y  = omt * p0y + t * p3y;
        float right_x = omt * p1x + t * p2x;
        float right_y = omt * p1y + t * p2y;

        uint8_t* out_row = &dst_strip[v * target_w];

        for (int u = 0; u < target_w; u++) {
            float s = (float)u * inv_w;
            float map_x = (1.0f - s) * left_x + s * right_x;
            float map_y = (1.0f - s) * left_y + s * right_y;

            out_row[u] = bilinear_sample(src_gray, src_w, src_h, map_x, map_y);
        }
    }
}

void slice_digit_tile(
    const uint8_t* strip,
    int strip_w,
    int strip_h,
    int digit_index,
    uint8_t* dst_digit_tile64x32
) {
    if (!strip || !dst_digit_tile64x32) return;

    int x_start = digit_index * SEG_DIGIT_W;
    if (x_start + SEG_DIGIT_W > strip_w) return;

    for (int row = 0; row < SEG_DIGIT_H && row < strip_h; row++) {
        const uint8_t* src_row = &strip[row * strip_w + x_start];
        uint8_t* dst_row = &dst_digit_tile64x32[row * SEG_DIGIT_W];
        memcpy(dst_row, src_row, SEG_DIGIT_W);
    }
}
