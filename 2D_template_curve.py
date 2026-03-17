# -*- coding: utf-8 -*-
"""
Build smooth 2D template curve from spectra_dax.npy + xy_21x5.npy
using CuPy + dual GPU parallel accumulation.

Coordinate definition:
- patch shape: N x H x W
- x from 0.5 to W+0.5, pixel centers are integers
- y from 0.5 to H+0.5, pixel centers are integers, increasing downward
- board center = ((W+1)/2, (H+1)/2)

For each patch:
- align its own (x_i, y_i) to board center
- evaluate on fine grid (10x10 subcells per pixel)
- average only over patches that actually cover each fine cell
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import multiprocessing as mp
import cupy as cp


# ---------------- file selection ----------------
def select_file(title="Select file", ext=".npy"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[(f"{ext} file", ext), ("All files", "*.*")]
    )
    root.destroy()
    return path


# ---------------- progress ----------------
def print_progress(i, total, step=1000, prefix=""):
    if (i + 1) % step == 0 or (i + 1) == total:
        print(f"{prefix}Processed {i+1}/{total}", flush=True)


# ---------------- GPU worker ----------------
def gpu_accumulate_worker(
    gpu_id,
    spectra_path,
    xy_path,
    start_idx,
    end_idx,
    H,
    W,
    upscale
):
    """
    Each worker:
    - loads only its slice
    - creates fine grid on its own GPU
    - accumulates template_sum and template_count
    - returns CPU numpy arrays
    """
    cp.cuda.Device(gpu_id).use()

    # load slice on CPU first
    spectra_all = np.load(spectra_path, mmap_mode="r")
    xy = np.load(xy_path, mmap_mode="r")

    spectra_sub = np.asarray(spectra_all[start_idx:end_idx], dtype=np.float32)  # float32 compute
    xy_sub = np.asarray(xy[start_idx:end_idx], dtype=np.float32)

    n_sub = spectra_sub.shape[0]

    x_center = np.float32((W + 1) / 2.0)
    y_center = np.float32((H + 1) / 2.0)

    step = np.float32(1.0 / upscale)

    # fine-grid centers: 0.55, 0.65, ..., W+0.45
    x_fine = cp.arange(
        np.float32(0.5 + step / 2),
        np.float32(W + 0.5),
        step,
        dtype=cp.float32
    )
    y_fine = cp.arange(
        np.float32(0.5 + step / 2),
        np.float32(H + 0.5),
        step,
        dtype=cp.float32
    )

    X_board, Y_board = cp.meshgrid(x_fine, y_fine)   # shape: (H*upscale, W*upscale)

    out_h = int(H * upscale)
    out_w = int(W * upscale)

    template_sum = cp.zeros((out_h, out_w), dtype=cp.float32)
    template_count = cp.zeros((out_h, out_w), dtype=cp.float32)

    # pixel-center coordinates in patch local coordinates: 1..W and 1..H
    # but for bilinear interpolation we use cell index based on center coordinates
    # x_patch and y_patch are in the same coordinate system as xy_sub

    for i in range(n_sub):
        patch = cp.asarray(spectra_sub[i], dtype=cp.float32)   # H x W
        x_i = cp.float32(xy_sub[i, 0])
        y_i = cp.float32(xy_sub[i, 1])

        # map board fine-grid center to patch coordinate system
        X_patch = X_board - x_center + x_i
        Y_patch = Y_board - y_center + y_i

        # valid coverage inside patch support [0.5, W+0.5] and [0.5, H+0.5]
        valid = (
            (X_patch >= cp.float32(0.5)) & (X_patch <= cp.float32(W + 0.5)) &
            (Y_patch >= cp.float32(0.5)) & (Y_patch <= cp.float32(H + 0.5))
        )

        if not bool(cp.any(valid)):
            continue

        # convert coordinate to pixel-index space
        # center 1 corresponds to array index 0, center W corresponds to array index W-1
        # so array-index coordinate = coord - 1
        X_idx = X_patch - cp.float32(1.0)
        Y_idx = Y_patch - cp.float32(1.0)

        # bilinear interpolation
        x0 = cp.floor(X_idx).astype(cp.int32)
        y0 = cp.floor(Y_idx).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # clip for safe gather
        x0c = cp.clip(x0, 0, W - 1)
        x1c = cp.clip(x1, 0, W - 1)
        y0c = cp.clip(y0, 0, H - 1)
        y1c = cp.clip(y1, 0, H - 1)

        wx = X_idx - x0.astype(cp.float32)
        wy = Y_idx - y0.astype(cp.float32)

        Ia = patch[y0c, x0c]
        Ib = patch[y0c, x1c]
        Ic = patch[y1c, x0c]
        Id = patch[y1c, x1c]

        vals = (
            (cp.float32(1.0) - wx) * (cp.float32(1.0) - wy) * Ia +
            wx * (cp.float32(1.0) - wy) * Ib +
            (cp.float32(1.0) - wx) * wy * Ic +
            wx * wy * Id
        )

        vals = cp.where(valid, vals, cp.float32(0.0))

        template_sum += vals
        template_count += valid.astype(cp.float32)

        if (i + 1) % 2000 == 0 or (i + 1) == n_sub:
            print(f"[GPU {gpu_id}] Processed {i+1}/{n_sub}", flush=True)

    return cp.asnumpy(template_sum), cp.asnumpy(template_count)


# ---------------- main ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    spectra_path = select_file("Select spectra_dax.npy", ".npy")
    xy_path = select_file("Select xy_21x5.npy", ".npy")

    if not spectra_path or not xy_path:
        raise RuntimeError("File selection cancelled.")

    # CPU side metadata
    spectra_all = np.load(spectra_path, mmap_mode="r")
    xy = np.load(xy_path, mmap_mode="r").astype(np.float32)

    if spectra_all.ndim != 3:
        raise RuntimeError(f"spectra_dax.npy must be 3D, got shape {spectra_all.shape}")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise RuntimeError(f"xy_21x5.npy must be (N,2), got shape {xy.shape}")
    if spectra_all.shape[0] != xy.shape[0]:
        raise RuntimeError(
            f"Length mismatch: spectra has {spectra_all.shape[0]} patches, xy has {xy.shape[0]} rows"
        )

    N, H, W = spectra_all.shape
    if W != 21:
        print(f"Warning: expected W=21, got W={W}", flush=True)

    print(f"Loaded spectra shape = {spectra_all.shape}, dtype = {spectra_all.dtype}", flush=True)
    print(f"Loaded xy shape      = {xy.shape}, dtype = {xy.dtype}", flush=True)

    upscale = 10
    x_center = (W + 1) / 2.0
    y_center = (H + 1) / 2.0
    print(f"Board center = ({x_center:.3f}, {y_center:.3f})", flush=True)

    # split equally to two GPUs
    mid = N // 2
    jobs = [
        (0, spectra_path, xy_path, 0, mid, H, W, upscale),
        (1, spectra_path, xy_path, mid, N, H, W, upscale),
    ]

    with mp.Pool(processes=2) as pool:
        results = pool.starmap(gpu_accumulate_worker, jobs)

    # merge CPU-side partial sums
    template_sum = np.zeros((H * upscale, W * upscale), dtype=np.float32)
    template_count = np.zeros((H * upscale, W * upscale), dtype=np.float32)

    for s, c in results:
        template_sum += s
        template_count += c

    template_curve = np.full_like(template_sum, np.nan, dtype=np.float32)
    valid = template_count > 0
    template_curve[valid] = template_sum[valid] / template_count[valid]

    # fine-grid coordinates for plotting
    step = 1.0 / upscale
    x_fine = np.arange(0.5 + step / 2, W + 0.5, step, dtype=np.float32)
    y_fine = np.arange(0.5 + step / 2, H + 0.5, step, dtype=np.float32)

    # plot with geometric aspect proportional to coordinate lengths
    fig_w = 8.0
    fig_h = fig_w * (H / W)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(
        template_curve,
        cmap="viridis",
        origin="upper",                  # y increases downward in your definition
        extent=[0.5, W + 0.5, H + 0.5, 0.5],
        aspect="equal",
        interpolation="nearest"
    )
    plt.colorbar(im, label="Average intensity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2D smooth template curve ({W}x{H}, {upscale}x{upscale} subcells)")
    plt.tight_layout()

    base, _ = os.path.splitext(spectra_path)
    out_png = base + f"_2D_template_curve_heatmap_{W}x{H}_10x_GPU.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved heatmap: {out_png}", flush=True)