# sobel_edges.py
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import mlx.core as mx

def load_grayscale(path: Path) -> mx.array:
    # Read with PIL and convert to single-channel grayscale [0..1] float32
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W)
    x = mx.array(arr)                                        # MLX array
    x = x[None, :, :, None]                                  # (N=1, H, W, C=1) NHWC for MLX
    return x

def save_gray_uint8(path: Path, x: mx.array) -> None:
    # x expected shape (1, H, W, 1) or (H, W)
    if len(x.shape) == 4:
        x = x[0, :, :, 0]
    x = mx.clip(x, 0.0, 1.0)
    mx.eval(x)                                               # ensure computed
    out = (np.array(x) * 255.0).astype(np.uint8)             # MLX -> NumPy -> uint8
    Image.fromarray(out, mode="L").save(path)

def sobel_kernels() -> mx.array:
    # Sobel kernels for horizontal (Gx) and vertical (Gy) gradients
    kx = mx.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=mx.float32)            # (3,3)
    ky = mx.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=mx.float32)          # (3,3)
    # Stack as C_out=2 filters, add input-channel dim: (2, KH, KW, C_in=1)
    w = mx.stack((kx, ky), axis=0)[:, :, :, None]
    return w

def gaussian_kernel_5x5(sigma=1.0) -> mx.array:
    # Optional: light smoothing to reduce noise before Sobel (separable 5x5)
    # Build a 1D Gaussian and outer-product it
    import math
    coords = mx.arange(-2, 3, dtype=mx.float32)
    g1 = mx.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1 = g1 / mx.sum(g1)
    g2d = (g1[:, None] * g1[None, :]).astype(mx.float32)     # (5,5)
    # make depthwise single-channel weight: (C_out=1, KH, KW, C_in=1)
    return g2d[None, :, :, None]

def sobel_edges(x: mx.array, smooth: bool = False, thr: float | None = None) -> mx.array:
    # x: (1,H,W,1) float32 in [0,1]
    if smooth:
        g = gaussian_kernel_5x5(1.0)
        x = mx.conv2d(x, g, padding=2)                        # keep (H,W) same
    w = sobel_kernels()
    # MLX conv2d expects input NHWC and weight (C_out, KH, KW, C_in). Use padding=1 for 3x3 SAME. 
    grads = mx.conv2d(x, w, padding=1)                        # (1,H,W,2)
    gx, gy = grads[..., 0:1], grads[..., 1:2]
    mag = mx.sqrt(gx * gx + gy * gy)                          # (1,H,W,1)

    # Normalize to [0,1] for viewing
    eps = mx.array(1e-6, dtype=mx.float32)
    mmin = mx.min(mag); mmax = mx.max(mag)
    mag = (mag - mmin) / (mmax - mmin + eps)

    if thr is not None:  # simple binary edges
        mag = (mag > thr).astype(mx.float32)
    return mag

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=Path("edges.png"))
    p.add_argument("--smooth", action="store_true", help="Gaussian blur before Sobel")
    p.add_argument("--threshold", type=float, default=None, help="Optional binary threshold in [0,1]")
    p.add_argument("--gpu", action="store_true", help="Run on GPU")
    args = p.parse_args()

    if args.gpu:
        mx.set_default_device(mx.gpu)

    x = load_grayscale(args.input)
    y = sobel_edges(x, smooth=args.smooth, thr=args.threshold)
    save_gray_uint8(args.out, y)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
