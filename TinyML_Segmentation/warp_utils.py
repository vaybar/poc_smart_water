"""
warp_utils.py - Lightweight Bilinear Quadrilateral Rectification & Slot Slicer

Implements pure-Python/NumPy perspective mapping without OpenCV dependencies,
matching 1:1 the C algorithm designed for the ESP32 Classic firmware.

Key Features:
1. Bilinear Quadrilateral Mapping: Maps source 4 corners (TL, TR, BR, BL) into a
   horizontal rectified strip (64 x [N*32]) using bilinear surface interpolation.
2. Fast Bilinear Texture Sampling: Sub-pixel interpolation for high visual fidelity.
3. Deterministic Slot Slicing: Divides the rectified strip into N individual (64, 32, 1)
   grayscale digit tiles, perfectly matching TinyML_Digit_Classifier's input contract.
"""

import numpy as np

def bilinear_sample_pixel(src: np.ndarray, x: float, y: float) -> int:
    """
    Samples a pixel from a 2D grayscale image at continuous coordinates (x, y)
    using bilinear interpolation. Matches the scalar C function for ESP32.
    """
    h, w = src.shape[:2]
    if x < 0.0 or x >= w - 1 or y < 0.0 or y >= h - 1:
        # Clamp to edge
        clamped_x = max(0, min(w - 1, int(round(x))))
        clamped_y = max(0, min(h - 1, int(round(y))))
        return int(src[clamped_y, clamped_x])

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    val = (
        (1.0 - dx) * (1.0 - dy) * float(src[y0, x0]) +
        dx * (1.0 - dy) * float(src[y0, x1]) +
        (1.0 - dx) * dy * float(src[y1, x0]) +
        dx * dy * float(src[y1, x1])
    )
    return max(0, min(255, int(round(val))))

def rectify_counter_strip(
    src_gray: np.ndarray,
    corners: np.ndarray,
    target_h: int = 64,
    target_w: int = 192,
) -> np.ndarray:
    """
    Rectifies an arbitrarily rotated counter quadrilateral into a horizontal strip.

    Args:
        src_gray: Grayscale input image (H, W) uint8.
        corners : (4, 2) array of coordinates in pixel space:
                  [0] Top-Left (TL)
                  [1] Top-Right (TR)
                  [2] Bottom-Right (BR)
                  [3] Bottom-Left (BL)
        target_h: Target height of rectified strip (default 64 px).
        target_w: Target width of rectified strip (e.g. 6 digits * 32 = 192 px).

    Returns:
        (target_h, target_w) uint8 numpy array with the horizontal strip.
    """
    if len(src_gray.shape) == 3:
        src_gray = src_gray[:, :, 0]

    # Normalized grid coordinates in target strip [0, 1]
    u = np.linspace(0.0, 1.0, target_w, endpoint=True)
    v = np.linspace(0.0, 1.0, target_h, endpoint=True)
    grid_u, grid_v = np.meshgrid(u, v)

    p0 = corners[0] # TL
    p1 = corners[1] # TR
    p2 = corners[2] # BR
    p3 = corners[3] # BL

    # Bilinear Quadrilateral Mapping surface:
    # P(s, t) = (1-s)(1-t)*P0 + s*(1-t)*P1 + s*t*P2 + (1-s)*t*P3
    s = grid_u
    t = grid_v

    map_x = (1.0 - s) * (1.0 - t) * p0[0] + s * (1.0 - t) * p1[0] + s * t * p2[0] + (1.0 - s) * t * p3[0]
    map_y = (1.0 - s) * (1.0 - t) * p0[1] + s * (1.0 - t) * p1[1] + s * t * p2[1] + (1.0 - s) * t * p3[1]

    # Vectorized bilinear sampling
    h_src, w_src = src_gray.shape
    map_x_clamped = np.clip(map_x, 0, w_src - 1)
    map_y_clamped = np.clip(map_y, 0, h_src - 1)

    x0 = np.floor(map_x_clamped).astype(np.int32)
    y0 = np.floor(map_y_clamped).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w_src - 1)
    y1 = np.clip(y0 + 1, 0, h_src - 1)

    dx = (map_x_clamped - x0).astype(np.float32)
    dy = (map_y_clamped - y0).astype(np.float32)

    ia = src_gray[y0, x0].astype(np.float32)
    ib = src_gray[y0, x1].astype(np.float32)
    ic = src_gray[y1, x0].astype(np.float32)
    id = src_gray[y1, x1].astype(np.float32)

    strip = (
        (1.0 - dx) * (1.0 - dy) * ia +
        dx * (1.0 - dy) * ib +
        (1.0 - dx) * dy * ic +
        dx * dy * id
    )

    return np.clip(np.round(strip), 0, 255).astype(np.uint8)

def slice_digits_from_strip(
    strip: np.ndarray,
    num_digits: int = 6,
    digit_w: int = 32,
    digit_h: int = 64,
) -> list[np.ndarray]:
    """
    Slices the horizontal rectified strip into num_digits individual digit tiles.

    Returns:
        List of num_digits numpy arrays, each of shape (64, 32, 1) and dtype uint8.
    """
    tiles = []
    for i in range(num_digits):
        x_start = i * digit_w
        x_end = x_start + digit_w
        tile = strip[0:digit_h, x_start:x_end]
        tile_3d = np.expand_dims(tile, axis=-1) # (64, 32, 1)
        tiles.append(tile_3d.astype(np.uint8))
    return tiles
