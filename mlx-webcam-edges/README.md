# MLX Edge Detector

Real-time, GPU-accelerated edge detection on macOS using **Apple MLX**.
This repo includes:

* **Single-image** Sobel/Scharr filter pipeline (`sobel_edges.py`)
* **Webcam** live demo with overlay/threshold controls (`webcam_edges_mlx.py`)

> Great for learning MLX arrays, `conv2d`, and lazy evaluation—without training any models.

---

## Features

* Sobel or Scharr 3×3 filters (Scharr recommended for crisper gradients)
* Color-aware gradients (per-channel + L2 combine) in **linear light** (sRGB → linear)
* Optional Gaussian blur pre-filter (depthwise per channel)
* Toggle **MLX GPU** path (`--gpu`) to demonstrate acceleration
* Live controls for view mode, threshold, and smoothing in the webcam demo

---

## Requirements

* macOS 13.5+ on **Apple Silicon (M-series)**
* Python 3.9+ (examples use 3.12)
* Camera permission for your terminal/IDE (for the webcam demo)

---

## Setup (with `uv`)

```bash
# clone your repo or create a new folder, then:
uv init --python 3.12
uv add mlx pillow numpy opencv-python
```

Repo layout (example):

```
.
├── README.md
├── sobel_edges.py
└── webcam_edges_mlx.py
```

---

## Single-Image Usage

```bash
uv run python sobel_edges.py input.jpg \
  --gpu \
  --smooth \
  --threshold 0.30 \
  --out edges.png
```

**Common flags**

* `--gpu` use the MLX GPU device (recommended on M-series)
* `--smooth` small Gaussian blur before gradients
* `--threshold 0..1` hard binary threshold (omit to save normalized magnitude)
* `--out` output file (PNG)

---

## Webcam (Real-Time) Usage

```bash
uv run python webcam_edges_mlx.py \
  --gpu \
  --scharr \
  --width 960 \
  --sigma 1.0 \
  --mode overlay \
  --thr_pct 87
```

**Flags**

* `--camera N` camera index (default `0`)
* `--width` resize width for processing (smaller → faster)
* `--scharr` use Scharr (omit for Sobel)
* `--sigma` Gaussian blur sigma (`0` disables)
* `--mode` `mag` | `mask` | `overlay`
* `--thr_pct` percentile for the binary mask (0–100)

**Keyboard controls**

* `q` quit
* `m` cycle view (magnitude → mask → overlay)
* `[` / `]` decrease / increase threshold percentile
* `-` / `+` decrease / increase

