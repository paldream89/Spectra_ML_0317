# -*- coding: utf-8 -*-
"""
Step2B: x fitting with NCC + 2D CNN refinement (OOF)

Key point for this version:
- spectra patch and template patch can have different absolute intensity scale.
- therefore, both NCC and CNN input use per-sample normalization to focus on shape.
"""

import os
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


# =========================
# file selector
# =========================
def select_file(initial_dir=".", title="Select file", ext=".npy"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title=title,
        filetypes=[(f"{ext} file", ext), ("All files", "*.*")],
    )
    root.destroy()
    return path


def log(msg):
    print(msg, flush=True)


def normalize_patch_batch(x, eps=1e-8):
    """Per-sample z-score normalization, shape-preserving for (B,5,21)."""
    xm = x.mean(axis=(1, 2), keepdims=True)
    xs = x.std(axis=(1, 2), keepdims=True)
    xs = np.clip(xs, eps, None)
    return (x - xm) / xs


# =========================
# NCC coarse x fitting
# =========================
def bilinear_sample_batch(template, x_coords, y_coords):
    """Sample template(50,210) at physical coords x/y with bilinear interpolation."""
    fx = (x_coords - 0.5) / 0.1
    fy = (y_coords - 0.5) / 0.1

    fx = np.clip(fx, 0.0, 209.0)
    fy = np.clip(fy, 0.0, 49.0)

    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, 209)
    y1 = np.clip(y0 + 1, 0, 49)

    wx = fx - x0
    wy = fy - y0

    Ia = template[y0, x0]
    Ib = template[y0, x1]
    Ic = template[y1, x0]
    Id = template[y1, x1]

    return (
        Ia * (1.0 - wx) * (1.0 - wy)
        + Ib * wx * (1.0 - wy)
        + Ic * (1.0 - wx) * wy
        + Id * wx * wy
    ).astype(np.float32)


def ncc_score_batch(a, b, eps=1e-8):
    """NCC after per-sample z-score, robust to multiplicative/additive intensity differences."""
    a = normalize_patch_batch(a, eps=eps).reshape(a.shape[0], -1)
    b = normalize_patch_batch(b, eps=eps).reshape(b.shape[0], -1)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1) + eps)
    return num / den


def find_best_xfit_ncc(spectra, y_ml_oof, template_curve, n_steps=101, batch_size=4096):
    """Coarse x_fit by maximizing NCC for x in [10.5, 11.5]."""
    N = spectra.shape[0]
    x_candidates = np.linspace(10.5, 11.5, n_steps, dtype=np.float32)

    x_fit_best = np.empty(N, dtype=np.float32)
    score_best = np.full(N, -np.inf, dtype=np.float32)

    patch_x = np.arange(1, 22, dtype=np.float32)[None, None, :]  # (1,1,21)
    patch_y = np.arange(1, 6, dtype=np.float32)[None, :, None]   # (1,5,1)

    spectra = np.asarray(spectra, dtype=np.float32)
    y_ml_oof = np.asarray(y_ml_oof, dtype=np.float32)
    template_curve = np.asarray(template_curve, dtype=np.float32)

    t0 = time.time()
    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        bsz = ed - st

        patch_batch = spectra[st:ed]
        y_batch = y_ml_oof[st:ed][:, None, None]
        y_coords = patch_y - y_batch + 3.0
        y_coords = np.broadcast_to(y_coords, (bsz, 5, 21))

        local_best_s = np.full(bsz, -np.inf, dtype=np.float32)
        local_best_x = np.full(bsz, 11.0, dtype=np.float32)

        for x_fit in x_candidates:
            x_coords = patch_x - x_fit + 11.0
            x_coords = np.broadcast_to(x_coords, (bsz, 5, 21))
            tmpl_patch = bilinear_sample_batch(template_curve, x_coords, y_coords)
            s = ncc_score_batch(patch_batch, tmpl_patch)
            mask = s > local_best_s
            local_best_s[mask] = s[mask]
            local_best_x[mask] = x_fit

        x_fit_best[st:ed] = local_best_x
        score_best[st:ed] = local_best_s
        log(f"[NCC] {ed}/{N} ({100.0 * ed / N:.1f}%), elapsed={time.time() - t0:.1f}s")

    return x_fit_best, score_best


# =========================
# CNN refinement model
# =========================
class XRefineDataset(Dataset):
    def __init__(self, spectra, y_ml_oof, x_ncc, ncc_score, x_true):
        self.spectra = np.asarray(spectra, dtype=np.float32)
        self.y = np.asarray(y_ml_oof, dtype=np.float32)
        self.x_ncc = np.asarray(x_ncc, dtype=np.float32)
        self.ncc_score = np.asarray(ncc_score, dtype=np.float32)
        self.x_true = np.asarray(x_true, dtype=np.float32)

        # Make CNN input scale-invariant to absolute intensity.
        self.spectra = normalize_patch_batch(self.spectra)

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.spectra[idx]).unsqueeze(0)  # (1,5,21)
        aux = torch.tensor([self.y[idx], self.x_ncc[idx], self.ncc_score[idx]], dtype=torch.float32)
        target = torch.tensor(self.x_true[idx], dtype=torch.float32)
        return img, aux, target


class XRefineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.img_head = nn.Sequential(
            nn.Linear(64 * 5 * 21, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, img, aux):
        f_img = self.img_head(self.features(img))
        f_aux = self.aux_head(aux)
        z = torch.cat([f_img, f_aux], dim=1)
        return self.out(z).squeeze(1)


def train_predict_oof_cnn(dataset, n_splits=5, epochs=12, batch_size=4096, lr=1e-3, weight_decay=1e-5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(dataset), dtype=np.float32)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_dp = torch.cuda.is_available() and torch.cuda.device_count() >= 2

    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(dataset))), start=1):
        log(f"[CNN] Fold {fold}/{n_splits}, train={len(tr_idx)}, valid={len(va_idx)}")
        tr_ds = Subset(dataset, tr_idx)
        va_ds = Subset(dataset, va_idx)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = XRefineCNN()
        if use_dp:
            model = nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(dev)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        cri = nn.SmoothL1Loss()

        best_state = None
        best_loss = np.inf

        for ep in range(epochs):
            model.train()
            tr_loss_sum, tr_n = 0.0, 0
            for img, aux, tgt in tr_loader:
                img = img.to(dev, non_blocking=True)
                aux = aux.to(dev, non_blocking=True)
                tgt = tgt.to(dev, non_blocking=True)

                opt.zero_grad()
                pred = model(img, aux)
                loss = cri(pred, tgt)
                loss.backward()
                opt.step()

                bs = img.shape[0]
                tr_loss_sum += loss.item() * bs
                tr_n += bs

            model.eval()
            va_loss_sum, va_n = 0.0, 0
            with torch.no_grad():
                for img, aux, tgt in va_loader:
                    img = img.to(dev, non_blocking=True)
                    aux = aux.to(dev, non_blocking=True)
                    tgt = tgt.to(dev, non_blocking=True)
                    pred = model(img, aux)
                    loss = cri(pred, tgt)
                    bs = img.shape[0]
                    va_loss_sum += loss.item() * bs
                    va_n += bs

            tr_loss = tr_loss_sum / max(tr_n, 1)
            va_loss = va_loss_sum / max(va_n, 1)
            log(f"[CNN][fold{fold}] ep {ep + 1:02d}/{epochs} train={tr_loss:.6f} valid={va_loss:.6f}")

            if va_loss < best_loss:
                best_loss = va_loss
                if isinstance(model, nn.DataParallel):
                    best_state = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        core = XRefineCNN().to(dev)
        core.load_state_dict(best_state)
        core.eval()

        preds = []
        with torch.no_grad():
            for img, aux, _ in va_loader:
                img = img.to(dev, non_blocking=True)
                aux = aux.to(dev, non_blocking=True)
                p = core(img, aux).detach().cpu().numpy()
                preds.append(p)

        preds = np.concatenate(preds, axis=0)
        oof[va_idx] = preds.astype(np.float32)

    return oof


def main():
    log("Select spectra file: shape (N, 5, 21)")
    spectra_path = select_file(initial_dir=".", title="Select spectra npy", ext=".npy")
    if not spectra_path:
        log("No spectra file selected. Exit.")
        return

    base_dir = os.path.dirname(spectra_path)
    xy_path = select_file(initial_dir=base_dir, title="Select xy npy", ext=".npy")
    yml_path = select_file(initial_dir=base_dir, title="Select y_ml_oof npy", ext=".npy")
    template_path = select_file(initial_dir=base_dir, title="Select template_curve npy", ext=".npy")
    if not (xy_path and yml_path and template_path):
        log("Missing one or more required files. Exit.")
        return

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
    if not (spectra.shape[0] == xy.shape[0] == y_ml_oof.shape[0]):
        raise ValueError("N mismatch among spectra / xy / y_ml_oof")

    x_true = xy[:, 0].astype(np.float32)

    log("Step1/2: coarse NCC x fitting (scale-invariant)...")
    x_ncc, ncc_score = find_best_xfit_ncc(spectra, y_ml_oof, template_curve, n_steps=101, batch_size=4096)

    log("Step2/2: 2D CNN OOF refinement (z-score normalized input)...")
    ds = XRefineDataset(spectra=spectra, y_ml_oof=y_ml_oof, x_ncc=x_ncc, ncc_score=ncc_score, x_true=x_true)
    x_fit_oof = train_predict_oof_cnn(ds, n_splits=5, epochs=12, batch_size=4096, lr=1e-3)
    x_fit_oof = np.clip(x_fit_oof, 10.5, 11.5)

    stem = os.path.splitext(os.path.basename(spectra_path))[0]
    out_npy = os.path.join(base_dir, f"{stem}_x_fit_oof.npy")
    out_npz = os.path.join(base_dir, f"{stem}_x_fit_results.npz")
    out_fig = os.path.join(base_dir, f"{stem}_x_fit_vs_x_true.png")

    np.save(out_npy, x_fit_oof)
    np.savez(
        out_npz,
        x_true=x_true,
        x_fit=x_fit_oof,
        x_ncc=x_ncc,
        ncc_score=ncc_score,
        y_ml_oof=y_ml_oof,
    )

    plt.figure(figsize=(7, 6), dpi=150)
    plt.scatter(x_true, x_fit_oof, s=1, alpha=0.2, label="OOF CNN fit")
    plt.plot([10.5, 11.5], [10.5, 11.5], "r--", lw=1.2, label="y=x")
    plt.xlim(10.5, 11.5)
    plt.ylim(10.5, 11.5)
    plt.xlabel("x_true")
    plt.ylabel("x_fit")
    plt.title("x_fit vs x_true (NCC + 2D CNN, scale-invariant)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    mae = np.mean(np.abs(x_fit_oof - x_true))
    rmse = np.sqrt(np.mean((x_fit_oof - x_true) ** 2))
    mae_ncc = np.mean(np.abs(x_ncc - x_true))
    rmse_ncc = np.sqrt(np.mean((x_ncc - x_true) ** 2))

    log(f"Saved: {out_npy}")
    log(f"Saved: {out_npz}")
    log(f"Saved: {out_fig}")
    log(f"NCC  MAE={mae_ncc:.6f}, RMSE={rmse_ncc:.6f}")
    log(f"CNN  MAE={mae:.6f}, RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
