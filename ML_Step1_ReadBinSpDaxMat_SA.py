# -*- coding: utf-8 -*-
"""
STEP1: Generate training dataset

Read:
    dax
    inf
    bin
    mat

Generate:
    spectra_dax : (M,5,21), uint16
    xy_21x5     : (M,2), float32
                  xy_21x5[:,0] = x_21x5
                  xy_21x5[:,1] = y_21x5

Save:
    *_spectra_dax.npy
    *_xy_21x5.npy
"""

import numpy as np
import os
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog
from scipy.io import loadmat
import matplotlib.pyplot as plt

from ReadSTORMBin import read_storm_bin


def select_file(initial_dir, ext):
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title=f"Select {ext} file",
        filetypes=[(ext, ext), ("All files", "*.*")]
    )

    root.destroy()
    return path


def load_tform_coeff(mat_path):
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "tform" not in mat:
        raise RuntimeError("tform not found in mat file")

    tform = mat["tform"]
    C = np.asarray(tform.tdata)

    if C.shape == (2, 10):
        C = C.T
    elif C.shape != (10, 2):
        raise RuntimeError(f"Unexpected tform.tdata shape: {C.shape}")

    cx = C[:, 0].astype(np.float64)
    cy = C[:, 1].astype(np.float64)
    return cx, cy


def eval_poly2d_3rd(x, y, c):
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = c
    return (a0
            + a1 * x + a2 * y
            + a3 * x**2 + a4 * x * y + a5 * y**2
            + a6 * x**3 + a7 * x**2 * y + a8 * x * y**2 + a9 * y**3)


def read_inf_dims(inf_path):
    with open(inf_path, 'r') as f:
        lines = f.readlines()

    info = {}
    for line in lines:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            info[k.strip().lower()] = v.strip()

    x_pixels = None
    y_pixels = None

    if "frame dimensions" in info:
        nums = re.findall(r'\d+', info["frame dimensions"])
        if len(nums) >= 2:
            x_pixels = int(nums[0])
            y_pixels = int(nums[1])

    if x_pixels is None or y_pixels is None:
        if "hend" not in info or "vend" not in info:
            raise RuntimeError("Cannot determine frame size from INF")
        x_pixels = int(re.findall(r'\d+', info["hend"])[0])
        y_pixels = int(re.findall(r'\d+', info["vend"])[0])

    num_frames = int(re.findall(r'\d+', info.get("number of frames", "1"))[0])

    return x_pixels, y_pixels, num_frames


def progress(i, total):
    p = (i + 1) / total
    n = int(p * 30)
    bar = "#" * n + "-" * (30 - n)
    sys.stdout.write(f"\r[{bar}] {p*100:.1f}%")
    sys.stdout.flush()


if __name__ == "__main__":

    crop_h = 5
    crop_w = 21

    half_h = crop_h // 2
    half_w = crop_w // 2

    I_thre = 500

    dax_path = select_file(".", ".dax")
    inf_path = select_file(".", ".inf")
    bin_path = select_file(".", ".bin")
    mat_path = select_file(".", ".mat")

    # read INF
    x_pixels, y_pixels, num_frames_inf = read_inf_dims(inf_path)
    pixels_per_frame = x_pixels * y_pixels

    # read DAX
    print("Reading dax...")
    with open(dax_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>u2'))

    num_frames_dax = len(data) // pixels_per_frame
    print("frames =", num_frames_dax)

    # read BIN
    total_frame, total_mol_num, original_mol_list = read_storm_bin(bin_path)
    total_number = int(np.asarray(total_mol_num).item())
    print("total molecules =", total_number)

    # read TFORM
    cx, cy = load_tform_coeff(mat_path)

    # collect spectra
    print("Collecting spectra...")
    t0 = time.time()

    spectra_list = []
    xy_list = []

    for i in range(total_number):
        if i % 5000 == 0 or i == total_number - 1:
            progress(i, total_number)

        I = float(original_mol_list['I'][i])
        if I < I_thre:
            continue

        x = float(original_mol_list['X'][i])
        y = float(original_mol_list['Y'][i])

        xt = eval_poly2d_3rd(x, y, cx)
        yt = eval_poly2d_3rd(x, y, cy)

        xint = int(round(xt))
        yint = int(round(yt)) - 1

        # coordinates inside 21x5 patch
        x_21x5 = (xt - xint) + half_w + 1
        y_21x5 = (yt - yint) + half_h 

        frame = int(original_mol_list['Frame'][i]) - 1
        if frame < 0 or frame >= num_frames_dax:
            continue

        s = frame * pixels_per_frame
        e = s + pixels_per_frame
        frame_img = data[s:e].reshape((y_pixels, x_pixels))

        if (xint - half_w < 0 or xint + half_w >= x_pixels or
            yint - half_h < 0 or yint + half_h >= y_pixels):
            continue

        patch = frame_img[
            yint - half_h:yint + half_h + 1,
            xint - half_w:xint + half_w + 1
        ]

        if patch.shape != (crop_h, crop_w):
            continue

        spectra_list.append(patch.astype(np.uint16, copy=False))
        xy_list.append([np.float32(x_21x5), np.float32(y_21x5)])

    print()

    if len(spectra_list) == 0:
        raise RuntimeError("No valid 21x5 patches collected.")

    spectra_dax = np.stack(spectra_list, axis=0).astype(np.uint16, copy=False)
    xy_21x5 = np.asarray(xy_list, dtype=np.float32)

    print("Total patches =", spectra_dax.shape[0])
    print("spectra_dax shape =", spectra_dax.shape, spectra_dax.dtype)
    print("xy_21x5 shape    =", xy_21x5.shape, xy_21x5.dtype)
    print("Time =", time.time() - t0)

    # save
    base, _ = os.path.splitext(bin_path)

    spectra_path = base + "_spectra_dax.npy"
    xy_path = base + "_xy_21x5.npy"

    np.save(spectra_path, spectra_dax)
    np.save(xy_path, xy_21x5)

    print("Saved:")
    print(spectra_path)
    print(xy_path)
    
    # ---------------- 统计 x_21x5 / y_21x5 直方图 ----------------
    # 假设 x_21x5 和 y_21x5 都是 Numpy 数组
    # x_21x5.shape = (N,), y_21x5.shape = (N,)
    
    xy_21x5 = np.array(xy_21x5)
    # xy_21x5.shape = (N,2)
    x_21x5 = xy_21x5[:,0]
    y_21x5 = xy_21x5[:,1]
    
    # ---------------- x_21x5 直方图 ----------------
    plt.figure(figsize=(6.8,4.5))
    plt.hist(x_21x5, bins=100, color='skyblue', edgecolor='k')
    plt.xlabel("x_21x5 (pixel)")
    plt.ylabel("Counts")
    plt.title("Distribution of x_21x5")
    plt.tight_layout()
    plt.show()
    
    # ---------------- y_21x5 直方图 ----------------
    plt.figure(figsize=(6.8,4.5))
    plt.hist(y_21x5, bins=100, color='salmon', edgecolor='k')
    plt.xlabel("y_21x5 (pixel)")
    plt.ylabel("Counts")
    plt.title("Distribution of y_21x5")
    plt.tight_layout()
    plt.show()
    
