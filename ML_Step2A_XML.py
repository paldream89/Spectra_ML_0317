# -*- coding: utf-8 -*-
"""
x-only ML prediction using template curve + high-res interpolation

Input:
    *_spectra_dax.npy : (N,H,W), uint16
    *_xy_21x5.npy     : (N,2), float32, x_true in column 0

Output:
    - histogram PNG of x errors
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
import xgboost as xgb


def select_file(title="Select file", ext=".npy"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title,
                                      filetypes=[(f"{ext} file", ext), ("All files", "*.*")])
    root.destroy()
    return path


def compute_fwhm_gaussian(data):
    from scipy.stats import norm
    data = data[np.isfinite(data)]
    mu, sigma = norm.fit(data)
    fwhm = 2.355 * sigma
    return mu, sigma, fwhm


def plot_hist(data, out_png, bins=120):
    plt.figure(figsize=(6.8, 5.0))
    plt.hist(data, bins=bins)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("x error (pixel)")
    plt.xlim(-0.6,0.6)
    plt.ylabel("Counts")
    plt.title("x error histogram (ML OOF)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show(block=True)


# =======================
# Main
# =======================
if __name__ == "__main__":
    # ------------- select files -------------
    spectra_path = select_file("Select spectra_dax.npy", ".npy")
    xy_path = select_file("Select xy_21x5.npy", ".npy")

    if not spectra_path or not xy_path:
        raise RuntimeError("File selection cancelled.")

    # ------------- load data -------------
    spectra_all = np.load(spectra_path)  # (N,H,W)
    xy = np.load(xy_path)
    x_true = xy[:, 0]

    N, H, W = spectra_all.shape
    x_center = (W + 1) / 2.0
    dx_true = x_true - x_center

    # high-resolution x-grid for interpolation
    x_hr = np.linspace(0, W - 1, W * 10)  # 0.1 pixel step

    # ------------- generate template curve -------------
    template_curve = np.zeros_like(x_hr)
    for i in range(N):
        patch = spectra_all[i]
        dx = dx_true[i]
        curve_row = patch.sum(axis=0)  # sum over y
        f = interp1d(np.arange(W), curve_row, kind='cubic', fill_value=0, bounds_error=False)
        curve_hr = f(x_hr - dx)  # shift according to true dx
        template_curve += curve_hr
    template_curve /= N

    # ------------- generate features -------------
    features = []
    targets = []
    for i in range(N):
        patch = spectra_all[i]
        curve_row = patch.sum(axis=0)
        f = interp1d(np.arange(W), curve_row, kind='cubic', fill_value=0, bounds_error=False)
        curve_hr = f(x_hr)  # don't subtract dx here
        features.append(curve_hr)
        targets.append(dx_true[i])

    features = np.stack(features, axis=0)  # N x len(x_hr)
    targets = np.array(targets)

    # ------------- 5-fold OOF ML using XGBoost -------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dx_oof = np.zeros_like(targets)

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
        print(f"Fold {fold_id}/5 ...")
        X_train, y_train = features[train_idx], targets[train_idx]
        X_val, y_val = features[val_idx], targets[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        dx_oof[val_idx] = pred_val

    # ------------- convert back to x_pred -------------
    x_ml_oof = dx_oof + x_center

    # ------------- compute errors -------------
    x_err = x_true - x_ml_oof

    mean_x = np.mean(x_err)
    std_x = np.std(x_err)
    mae_x = np.mean(np.abs(x_err))
    mu_x, sigma_x, fwhm_x = compute_fwhm_gaussian(x_err)

    print("\nX ML OOF results:")
    print(f"mean = {mean_x:.8f}")
    print(f"std  = {std_x:.8f}")
    print(f"MAE  = {mae_x:.8f}")
    print("Gaussian fit results for x:")
    print(f"mu    = {mu_x:.8f}")
    print(f"sigma = {sigma_x:.8f}")
    print(f"FWHM  = {fwhm_x:.8f}")

    # ------------- plot histogram -------------
    base, _ = os.path.splitext(spectra_path)
    out_hist = base + "_x_err_ml_oof_hist_template_curve.png"
    plot_hist(x_err, out_hist)