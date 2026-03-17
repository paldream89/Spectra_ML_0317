# -*- coding: utf-8 -*-
"""
Estimate x_fit by matching each 5x21 spectra patch to template_curve via
normalized cross-correlation while fixing y at y_ml_oof.

Inputs selected via file dialogs:
    *_spectra*.npy        shape: (N, 5, 21)
    *_xy*.npy             shape: (N, 2), xy[:,0]=x_true
    *_y_ml_oof*.npy       shape: (N,)
    *template_curve*.npy  shape: (50, 210)

Outputs:
    *_x_fit_oof.npy
    *_x_fit_vs_x_true.png
    *_x_fit_results.npz
"""

import os
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt


# =========================
# file selector (same style)
# =========================
def select_file(initial_dir=".", title="Select file", ext=".npy"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title=title,
        filetypes=[(f"{ext} file", ext), ("All files", "*.*")]
    )
    root.destroy()
    return path


def log(msg):
    print(msg, flush=True)


def bilinear_sample_batch(template, x_coords, y_coords):
    """
    template: (50,210)
    x_coords: (B,5,21)
    y_coords: (B,5,21)

    Coordinate definition:
      x centers are 0.5..21.5 over 210 bins (step=0.1)
      y centers are 0.5..5.5  over  50 bins (step=0.1)
    """
    # convert physical coords -> float indices
    fx = (x_coords - 0.5) / 0.1
    fy = (y_coords - 0.5) / 0.1

    # valid interpolation range is [0, 209] x [0, 49]
    fx = np.clip(fx, 0.0, 209.0)
    fy = np.clip(fy, 0.0, 49.0)

    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, 209)
    y1 = np.clip(y0 + 1, 0, 49)

    wx = fx - x0
    wy = fy - y0

    # advanced indexing produces shape (B,5,21)
    Ia = template[y0, x0]
    Ib = template[y0, x1]
    Ic = template[y1, x0]
    Id = template[y1, x1]

    out = (
        Ia * (1.0 - wx) * (1.0 - wy)
        + Ib * wx * (1.0 - wy)
        + Ic * (1.0 - wx) * wy
        + Id * wx * wy
    )
    return out.astype(np.float32)


def ncc_score_batch(a, b, eps=1e-8):
    """Normalized cross-correlation over last two dims for arrays (B,5,21)."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)

    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1) + eps)
    return num / den


def find_best_xfit(
    spectra,
    y_ml_oof,
    template_curve,
    x_min=10.5,
    x_max=11.5,
    n_steps=201,
    batch_size=4096,
):
    """
    For each sample i:
      - fixed anchor point in patch: (x_fit, y_ml_oof[i])
      - align this anchor to template center (11, 3)
      - vary x_fit in [x_min, x_max], choose best NCC score
    """
    N = spectra.shape[0]

    x_candidates = np.linspace(x_min, x_max, n_steps, dtype=np.float32)
    x_fit_best = np.empty(N, dtype=np.float32)
    score_best = np.full(N, -np.inf, dtype=np.float32)

    # patch coordinate centers (integers)
    patch_x_centers = np.arange(1, 22, dtype=np.float32)[None, None, :]  # (1,1,21)
    patch_y_centers = np.arange(1, 6, dtype=np.float32)[None, :, None]   # (1,5,1)

    # promote to float32 once
    spectra = np.asarray(spectra, dtype=np.float32)
    y_ml_oof = np.asarray(y_ml_oof, dtype=np.float32)
    template_curve = np.asarray(template_curve, dtype=np.float32)

    t0 = time.time()
    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        bsz = ed - st

        patch_batch = spectra[st:ed]             # (B,5,21)
        y_batch = y_ml_oof[st:ed][:, None, None] # (B,1,1)

        # y mapping does not depend on x_fit candidate
        # template_y = patch_y - y_fit + 3
        y_coords = patch_y_centers - y_batch + 3.0
        y_coords = np.broadcast_to(y_coords, (bsz, 5, 21))

        best_local_score = np.full(bsz, -np.inf, dtype=np.float32)
        best_local_x = np.full(bsz, x_min, dtype=np.float32)

        for x_fit in x_candidates:
            # template_x = patch_x - x_fit + 11
            x_coords = patch_x_centers - x_fit + 11.0
            x_coords = np.broadcast_to(x_coords, (bsz, 5, 21))

            tmpl_patch = bilinear_sample_batch(template_curve, x_coords, y_coords)
            scores = ncc_score_batch(patch_batch, tmpl_patch)

            improved = scores > best_local_score
            best_local_score[improved] = scores[improved]
            best_local_x[improved] = x_fit

        x_fit_best[st:ed] = best_local_x
        score_best[st:ed] = best_local_score

        elapsed = time.time() - t0
        log(f"Processed {ed}/{N} ({100.0*ed/N:.2f}%), elapsed={elapsed:.1f}s")

    return x_fit_best, score_best, x_candidates


def main():
    log("Select spectra file: shape (N, 5, 21)")
    spectra_path = select_file(initial_dir=".", title="Select spectra npy", ext=".npy")
    if not spectra_path:
        log("No spectra file selected. Exit.")
        return

    base_dir = os.path.dirname(spectra_path)
    log("Select xy file: shape (N, 2)")
    xy_path = select_file(initial_dir=base_dir, title="Select xy npy", ext=".npy")
    if not xy_path:
        log("No xy file selected. Exit.")
        return

    log("Select y_ml_oof file: shape (N,)")
    yml_path = select_file(initial_dir=base_dir, title="Select y_ml_oof npy", ext=".npy")
    if not yml_path:
        log("No y_ml_oof file selected. Exit.")
        return

    log("Select template_curve file: shape (50, 210)")
    template_path = select_file(initial_dir=base_dir, title="Select template_curve npy", ext=".npy")
    if not template_path:
        log("No template_curve file selected. Exit.")
        return

    log("Loading files...")
    spectra = np.load(spectra_path)
    xy = np.load(xy_path)
    y_ml_oof = np.load(yml_path)
    template_curve = np.load(template_path)

    if spectra.ndim != 3 or spectra.shape[1:] != (5, 21):
        raise ValueError(f"spectra shape must be (N,5,21), got {spectra.shape}")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy shape must be (N,2), got {xy.shape}")
    if y_ml_oof.ndim != 1:
        raise ValueError(f"y_ml_oof shape must be (N,), got {y_ml_oof.shape}")
    if template_curve.shape != (50, 210):
        raise ValueError(f"template_curve shape must be (50,210), got {template_curve.shape}")

    N = spectra.shape[0]
    if xy.shape[0] != N or y_ml_oof.shape[0] != N:
        raise ValueError("N mismatch among spectra / xy / y_ml_oof")

    x_true = xy[:, 0].astype(np.float32)

    log(
        f"Start fitting on N={N}, spectra={spectra.shape}, xy={xy.shape}, "
        f"y_ml_oof={y_ml_oof.shape}, template={template_curve.shape}"
    )

    x_fit_best, score_best, x_candidates = find_best_xfit(
        spectra=spectra,
        y_ml_oof=y_ml_oof,
        template_curve=template_curve,
        x_min=10.5,
        x_max=11.5,
        n_steps=201,
        batch_size=4096,
    )

    # save outputs
    stem = os.path.splitext(os.path.basename(spectra_path))[0]
    out_xfit_path = os.path.join(base_dir, f"{stem}_x_fit_oof.npy")
    out_npz_path = os.path.join(base_dir, f"{stem}_x_fit_results.npz")
    out_fig_path = os.path.join(base_dir, f"{stem}_x_fit_vs_x_true.png")

    np.save(out_xfit_path, x_fit_best)
    np.savez(
        out_npz_path,
        x_true=x_true,
        x_fit=x_fit_best,
        score_best=score_best,
        x_candidates=x_candidates,
        y_ml_oof=y_ml_oof,
    )

    # scatter plot
    plt.figure(figsize=(7.0, 6.0), dpi=150)
    plt.scatter(x_true, x_fit_best, s=1, alpha=0.2)
    plt.plot([10.5, 11.5], [10.5, 11.5], "r--", lw=1.2, label="y=x")
    plt.xlim(10.5, 11.5)
    plt.ylim(10.5, 11.5)
    plt.xlabel("x_true")
    plt.ylabel("x_fit (best NCC)")
    plt.title("x_fit vs x_true")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_fig_path)
    plt.close()

    mae = np.mean(np.abs(x_fit_best - x_true))
    rmse = np.sqrt(np.mean((x_fit_best - x_true) ** 2))

    log(f"Saved: {out_xfit_path}")
    log(f"Saved: {out_npz_path}")
    log(f"Saved: {out_fig_path}")
    log(f"Done. MAE={mae:.6f}, RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
