import argparse, time
from typing import Tuple
import numpy as np
import cv2

import mlx.core as mx

# -------------------------------
# Utilities (weights + helpers)
# -------------------------------

def srgb_to_linear(x: mx.array) -> mx.array:
    # x: NHWC in [0,1], sRGB → linear
    a = 0.055
    return mx.where(x <= 0.04045, x/12.92, ((x + a)/(1+a)) ** 2.4)

def gaussian_depthwise_rgb(sigma: float, radius: int = 2) -> mx.array:
    """Depthwise 5x5 Gaussian applied independently to R,G,B.
       Returns weight of shape (C_out=3, KH=5, KW=5, C_in=3) with a diagonal kernel."""
    xs = mx.arange(-radius, radius+1, dtype=mx.float32)
    g1 = mx.exp(-(xs**2)/(2*sigma*sigma))
    g1 = g1 / mx.sum(g1)
    g2 = (g1[:, None] * g1[None, :]).astype(mx.float32)          # (5,5)
    g2_np = np.array(g2)

    w = np.zeros((3, 5, 5, 3), dtype=np.float32)                 # diagonal “depthwise”
    for c in range(3):
        w[c, :, :, c] = g2_np
    return mx.array(w)

def scharr_filters_rgb() -> mx.array:
    """6 filters: (Gx_R,Gy_R,Gx_G,Gy_G,Gx_B,Gy_B) stacked along C_out.
       Shape: (6, 3, 3, 3) for NHWC conv2d."""
    kx = np.array([[ 3, 0, -3],
                   [10, 0,-10],
                   [ 3, 0, -3]], dtype=np.float32)
    ky = kx.T
    w = np.zeros((6, 3, 3, 3), dtype=np.float32)
    for c in range(3):
        w[2*c,   :, :, c] = kx
        w[2*c+1, :, :, c] = ky
    return mx.array(w)

def sobel_filters_rgb() -> mx.array:
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    w = np.zeros((6, 3, 3, 3), dtype=np.float32)
    for c in range(3):
        w[2*c,   :, :, c] = kx
        w[2*c+1, :, :, c] = ky
    return mx.array(w)

def edges_color_magnitude(frame_rgb_srgb: np.ndarray,
                          W_grad: mx.array,
                          sigma: float|None) -> Tuple[np.ndarray, np.ndarray]:
    """
    frame_rgb_srgb: HxWx3 uint8 (sRGB)
    returns (mag_np [0..1], rgb_for_overlay uint8)
    """
    H, W, _ = frame_rgb_srgb.shape
    # to MLX NHWC float
    x = mx.array(frame_rgb_srgb.astype(np.float32) / 255.0)[None, ...]  # (1,H,W,3)
    x = srgb_to_linear(x)

    if sigma and sigma > 0:
        W_blur = gaussian_depthwise_rgb(float(sigma))
        x = mx.conv2d(x, W_blur, padding=2)                              # (1,H,W,3)

    grads = mx.conv2d(x, W_grad, padding=1)                              # (1,H,W,6)
    grads = grads.reshape((1, H, W, 3, 2))
    gx = grads[..., 0]
    gy = grads[..., 1]
    mag = mx.sqrt(mx.sum(gx*gx + gy*gy, axis=-1, keepdims=True))         # (1,H,W,1)

    # normalize per-frame for display
    eps = mx.array(1e-6, mx.float32)
    mag = (mag - mx.min(mag)) / (mx.max(mag) - mx.min(mag) + eps)

    mx.eval(mag)               # compute before converting to NumPy
    mag_np = np.array(mag[0, :, :, 0])   # HxW
    return mag_np, frame_rgb_srgb

def make_overlay(base_rgb_srgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """White overlay on edges; inputs HxWx3 uint8, HxW float 0..1"""
    out = base_rgb_srgb.copy()
    m = mask01 >= 0.5
    out[m] = (255, 255, 255)
    return out

# -------------------------------
# Main (webcam loop)
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Use MLX GPU device")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=960, help="Resize width for processing")
    ap.add_argument("--scharr", action="store_true", help="Use Scharr (default Sobel if unset)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (0 = off)")
    ap.add_argument("--mode", choices=["mag", "mask", "overlay"], default="overlay")
    ap.add_argument("--thr_pct", type=float, default=87.0, help="Percentile for binary mask [0..100]")
    args = ap.parse_args()

    if args.gpu:
        mx.set_default_device(mx.gpu)

    # Build gradient weights once (lives on selected device)
    W_grad = scharr_filters_rgb() if args.scharr else sobel_filters_rgb()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions in System Settings.")

    win = "MLX Edges (q=quit, m=toggle mode)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_t = time.perf_counter()
    mode = args.mode  # 'mag' | 'mask' | 'overlay'
    sigma = args.sigma

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Optional resize for speed
        if args.width and frame_bgr.shape[1] != args.width:
            h = int(frame_bgr.shape[0] * (args.width / frame_bgr.shape[1]))
            frame_bgr = cv2.resize(frame_bgr, (args.width, h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB for processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MLX math
        mag_np, base_rgb = edges_color_magnitude(frame_rgb, W_grad, sigma)

        # Threshold (binary) via percentile
        thr = np.percentile(mag_np, np.clip(args.thr_pct, 0, 100))
        mask01 = (mag_np > thr).astype(np.float32)

        # Choose visualization
        if mode == "mag":
            vis = (mag_np * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif mode == "mask":
            vis = (mask01 * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:  # overlay
            vis = make_overlay(base_rgb, mask01)

        # Back to BGR for imshow
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        # FPS counter
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, (now - last_t))
        last_t = now
        cv2.putText(vis_bgr, f"{mode} | {'GPU' if args.gpu else 'CPU'} | {fps:5.1f} FPS",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = {"mag":"mask","mask":"overlay","overlay":"mag"}[mode]
        elif key == ord('['):
            args.thr_pct = max(0.0, args.thr_pct - 1.0)
        elif key == ord(']'):
            args.thr_pct = min(100.0, args.thr_pct + 1.0)
        elif key == ord('-'):
            sigma = max(0.0, sigma - 0.2)
        elif key == ord('=') or key == ord('+'):
            sigma = min(5.0, sigma + 0.2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
