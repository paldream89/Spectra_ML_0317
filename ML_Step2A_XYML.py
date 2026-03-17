# -*- coding: utf-8 -*-
"""
ML for y using full 21x5 patch (no dimensionality reduction)

Input:
    *_spectra_dax.npy   : (M, H, W), usually uint16
    *_xy_21x5.npy       : (M, 2), float32
                          xy[:,0] = x_true
                          xy[:,1] = y_true

Output:
    *_y_ml_oof.npy
    *_y_err_ml_oof.npy
    *_y_ml_oof_results.npz
    *_y_err_ml_oof_hist.png
    *_y_ml_final_model.pth

Method:
    - full 2D CNN on 21x5 patch
    - 5-fold out-of-fold prediction
    - dual GPU via DataParallel
"""

import os
import time
import math
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold
from scipy.stats import norm

import multiprocessing as mp
import cupy as cp


# =========================
# file selector
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


# =========================
# gaussian fit for error
# =========================
def compute_fwhm_gaussian(data):
    data = data[np.isfinite(data)]
    mu, sigma = norm.fit(data)
    fwhm = 2.355 * sigma
    return mu, sigma, fwhm


# =========================
# dataset
# =========================
class PatchYDataset(Dataset):
    def __init__(self, spectra_dax, y_true, normalize_mode="sum"):
        """
        spectra_dax: (M, H, W), uint16 or float32
        y_true:      (M,), float32
        """
        self.X = np.asarray(spectra_dax, dtype=np.float32)
        self.y = np.asarray(y_true, dtype=np.float32)

        if normalize_mode == "sum":
            denom = self.X.sum(axis=(1, 2), keepdims=True)
            denom = np.clip(denom, 1e-6, None)
            self.X = self.X / denom
        elif normalize_mode == "max":
            denom = self.X.max(axis=(1, 2), keepdims=True)
            denom = np.clip(denom, 1e-6, None)
            self.X = self.X / denom
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Conv2d input: (C,H,W)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)   # (1,H,W)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# =========================
# model
# =========================
class YRegressor2DCNN(nn.Module):
    def __init__(self, H=5, W=21):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * H * W, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(1)


# =========================
# train one fold
# =========================
def train_one_fold_ycnn(
    train_dataset,
    valid_dataset,
    H,
    W,
    gpu_ids=(0, 1),
    epochs=20,
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    fold_name="fold"
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YRegressor2DCNN(H=H, W=W)

    if torch.cuda.is_available() and len(gpu_ids) >= 2 and torch.cuda.device_count() >= 2:
        log(f"[{fold_name}] Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=list(gpu_ids))
    else:
        log(f"[{fold_name}] Using single GPU/CPU")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    best_valid_loss = np.inf
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_loss_sum += loss.item() * bs
            train_n += bs

        model.eval()
        valid_loss_sum = 0.0
        valid_n = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss = criterion(pred, yb)

                bs = xb.shape[0]
                valid_loss_sum += loss.item() * bs
                valid_n += bs

        train_loss = train_loss_sum / max(train_n, 1)
        valid_loss = valid_loss_sum / max(valid_n, 1)

        log(f"[{fold_name}] epoch {ep+1:02d}/{epochs}, train={train_loss:.8f}, valid={valid_loss:.8f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if isinstance(model, nn.DataParallel):
                best_state = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best state
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)

    # predict on validation set
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in valid_loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)
    return preds, model


# =========================
# OOF training
# =========================
def y_oof_prediction_2dcnn(
    spectra_dax,
    y_true,
    gpu_ids=(0, 1),
    n_splits=5,
    epochs=20,
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    normalize_mode="sum",
    random_state=0
):
    M, H, W = spectra_dax.shape

    dataset_all = PatchYDataset(
        spectra_dax=spectra_dax,
        y_true=y_true,
        normalize_mode=normalize_mode
    )

    y_oof = np.zeros(M, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(np.arange(M)), start=1):
        log(f"\n========== Fold {fold_id}/{n_splits} ==========")

        train_dataset = Subset(dataset_all, tr_idx)
        valid_dataset = Subset(dataset_all, va_idx)

        pred_valid, _ = train_one_fold_ycnn(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            H=H,
            W=W,
            gpu_ids=gpu_ids,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            fold_name=f"fold{fold_id}"
        )

        y_oof[va_idx] = pred_valid

    # final model on all data
    log("\nTraining final y model on all data ...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_model = YRegressor2DCNN(H=H, W=W)

    if torch.cuda.is_available() and len(gpu_ids) >= 2 and torch.cuda.device_count() >= 2:
        final_model = nn.DataParallel(final_model, device_ids=list(gpu_ids))

    final_model = final_model.to(device)

    loader_all = DataLoader(
        dataset_all,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    final_model.train()
    for ep in range(epochs):
        loss_sum = 0.0
        n_sum = 0

        for xb, yb in loader_all:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = final_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            loss_sum += loss.item() * bs
            n_sum += bs

        log(f"[final] epoch {ep+1:02d}/{epochs}, train={loss_sum / max(n_sum,1):.8f}")

    return y_oof, final_model


# =========================
# plotting
# =========================
def plot_hist(y_err, out_png, bins=120):
    plt.figure(figsize=(6.8, 5.0))
    plt.hist(y_err[np.isfinite(y_err)], bins=bins)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("y error (pixel)")
    plt.ylabel("Counts")
    plt.title("y error histogram (ML OOF)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show(block=False)
    plt.close()

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


# =========================
# main
# =========================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    log(f"CUDA available, GPU count = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # --------------- parameters ---------------
    gpu_ids = (0, 1)

    n_splits = 5
    epochs = 20
    batch_size = 1024
    lr = 1e-3
    weight_decay = 1e-5
    normalize_mode = "sum"   # "sum", "max", "none"

    # --------------- select files ---------------
    spectra_path = select_file(title="Select spectra_dax.npy", ext=".npy")
    xy_path = select_file(title="Select xy_21x5.npy", ext=".npy")

    if not spectra_path or not xy_path:
        raise RuntimeError("File selection cancelled.")

    # --------------- load data ---------------
    t0 = time.time()

    log("Loading spectra...")
    spectra_dax = np.load(spectra_path, mmap_mode="r")  # (M,H,W)

    log("Loading xy...")
    xy = np.load(xy_path)  # (M,2)

    if spectra_dax.ndim != 3:
        raise RuntimeError(f"spectra should have shape (M,H,W), got {spectra_dax.shape}")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise RuntimeError(f"xy should have shape (M,2), got {xy.shape}")
    if spectra_dax.shape[0] != xy.shape[0]:
        raise RuntimeError("Length mismatch between spectra and xy")

    M, H, W = spectra_dax.shape
    y_true = xy[:, 1].astype(np.float32, copy=False)
    # y_true -= 1.0

    log(f"spectra shape = {spectra_dax.shape}, dtype = {spectra_dax.dtype}")
    log(f"xy shape      = {xy.shape}, dtype = {xy.dtype}")

    # load into RAM once; for 370k * 5 * 21 float32 this is manageable
    spectra_all = np.asarray(spectra_dax, dtype=np.float32)

    # --------------- OOF ML ---------------
    log("Start 2D CNN OOF prediction for y ...")
    y_ml_oof, final_model = y_oof_prediction_2dcnn(
        spectra_dax=spectra_all,
        y_true=y_true,
        gpu_ids=gpu_ids,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        normalize_mode=normalize_mode,
        random_state=0
    )

    # --------------- error ---------------
    y_err_ml = y_true - y_ml_oof

    mean_ml = np.mean(y_err_ml)
    std_ml = np.std(y_err_ml)
    mae_ml = np.mean(np.abs(y_err_ml))

    mu_g, sigma_g, fwhm_g = compute_fwhm_gaussian(y_err_ml)

    log("\nML OOF results:")
    log(f"mean = {mean_ml:.8f}")
    log(f"std  = {std_ml:.8f}")
    log(f"MAE  = {mae_ml:.8f}")

    log("\nGaussian fit results:")
    log(f"mu    = {mu_g:.8f}")
    log(f"sigma = {sigma_g:.8f}")
    log(f"FWHM  = {fwhm_g:.8f}")

    # --------------- save ---------------
    base, _ = os.path.splitext(spectra_path)

    out_y = base + f"_y_ml_oof_H{H}.npy"
    out_err = base + f"_y_err_ml_oof_H{H}.npy"
    out_hist = base + f"_y_err_ml_oof_hist_H{H}.png"
    out_model = base + f"_y_ml_final_model_H{H}.pth"
    out_npz = base + f"_y_ml_oof_results_H{H}.npz"

    np.save(out_y, y_ml_oof.astype(np.float32))
    # np.save(out_err, y_err_ml.astype(np.float32))

    # np.savez(
    #     out_npz,
    #     y_true=y_true.astype(np.float32),
    #     y_ml_oof=y_ml_oof.astype(np.float32),
    #     y_err_ml=y_err_ml.astype(np.float32),
    #     mean=np.float32(mean_ml),
    #     std=np.float32(std_ml),
    #     mae=np.float32(mae_ml),
    #     mu_gauss=np.float32(mu_g),
    #     sigma_gauss=np.float32(sigma_g),
    #     fwhm_gauss=np.float32(fwhm_g),
    # )

    if isinstance(final_model, nn.DataParallel):
        torch.save(final_model.module.state_dict(), out_model)
    else:
        torch.save(final_model.state_dict(), out_model)

    plot_hist(y_err_ml, out_hist, bins=120)

    log("\nSaved:")
    log(out_y)
    log(out_err)
    log(out_npz)
    log(out_hist)
    log(out_model)
    log(f"Total elapsed = {time.time() - t0:.2f} s")
    
    #===================2D template curve===============
    mp.set_start_method("spawn", force=True)

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
    
    #----------------- save template_curve -----------------
    base, _ = os.path.splitext(spectra_path)
    out_npy = base + "_TC.npy"   # <spectra_path>_TC.npy
    np.save(out_npy, template_curve.astype(np.float32))
    print(f"Saved template_curve as npy: {out_npy}", flush=True)

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