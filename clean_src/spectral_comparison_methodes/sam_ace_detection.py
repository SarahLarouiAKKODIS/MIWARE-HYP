#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio

from spectral.algorithms import spectral_angles

from spectral_comparison_methodes.commun_functions import read_wavelengths_and_fwhm_from_csv, load_tab_spectrum, load_ecostress_txt_spectrum, find_reference_txts, resample_spectrum_gaussian_fwhm, l2_normalize_spectrum, normalize_image_l2, estimate_background_stats, sam_to_score

# -----------------------------
# 5) ACE
# -----------------------------
def ace_map_from_background_stats(
    img_brc: np.ndarray,          # (bands, rows, cols)
    target_b: np.ndarray,         # (bands,)
    bg_mean_b: np.ndarray,        # (bands,)
    bg_cov_bb: np.ndarray,        # (bands, bands)
    valid_mask_rc: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calcule une carte ACE à partir des stats de fond.
    Retourne une carte (rows, cols) en float32 dans [0, 1] en pratique.
    """
    img_brc = np.asarray(img_brc, dtype=np.float32)
    target_b = np.asarray(target_b, dtype=np.float32).reshape(-1)
    bg_mean_b = np.asarray(bg_mean_b, dtype=np.float32).reshape(-1)
    bg_cov_bb = np.asarray(bg_cov_bb, dtype=np.float32)

    bands, rows, cols = img_brc.shape
    if target_b.shape[0] != bands:
        raise ValueError(f"target_b incompatible: {target_b.shape[0]} vs bands={bands}")
    if bg_mean_b.shape[0] != bands:
        raise ValueError(f"bg_mean_b incompatible: {bg_mean_b.shape[0]} vs bands={bands}")
    if bg_cov_bb.shape != (bands, bands):
        raise ValueError(f"bg_cov_bb incompatible: {bg_cov_bb.shape} vs ({bands}, {bands})")

    X = img_brc.reshape(bands, -1).T   # (n_pix, bands)
    finite = np.isfinite(X).all(axis=1)

    if valid_mask_rc is not None:
        finite &= valid_mask_rc.reshape(-1)

    ace = np.full(X.shape[0], np.nan, dtype=np.float32)
    if not np.any(finite):
        return ace.reshape(rows, cols)

    Xv = X[finite, :] - bg_mean_b[None, :]
    tv = target_b - bg_mean_b

    inv_cov = np.linalg.pinv(bg_cov_bb).astype(np.float32)

    inv_t = inv_cov @ tv
    t_inv_t = float(tv @ inv_t)

    if not np.isfinite(t_inv_t) or t_inv_t <= eps:
        return ace.reshape(rows, cols)

    X_inv = Xv @ inv_cov                # (n_valid, bands)
    x_inv_x = np.sum(X_inv * Xv, axis=1)
    x_inv_t = Xv @ inv_t

    denom = t_inv_t * x_inv_x
    good = np.isfinite(denom) & (denom > eps)

    ace_valid = np.full(Xv.shape[0], np.nan, dtype=np.float32)
    ace_valid[good] = (x_inv_t[good] ** 2) / denom[good]

    ace_valid = np.clip(ace_valid, 0.0, 1.0).astype(np.float32)
    ace[finite] = ace_valid

    return ace.reshape(rows, cols).astype(np.float32)


# -----------------------------
# 6) Optional: combinaison SAM + ACE
# -----------------------------
def sam_to_score(sam: np.ndarray, sam_scale: float = 0.1) -> np.ndarray:
    sam = sam.astype(np.float32)
    out = np.exp(-sam / float(sam_scale)).astype(np.float32)
    out[~np.isfinite(sam)] = np.nan
    return out


def combine_sam_ace(
    sam_min: np.ndarray,
    ace_max: np.ndarray,
    sam_scale: float = 0.1,
    mode: str = "product",   # "product" ou "mean"
) -> np.ndarray:
    s_score = sam_to_score(sam_min, sam_scale=sam_scale)

    # ACE est déjà pseudo-normalisé entre 0 et 1
    a_score = ace_max.astype(np.float32).copy()
    a_score[~np.isfinite(a_score)] = np.nan
    a_score = np.clip(a_score, 0.0, 1.0)

    if mode == "mean":
        combo = 0.5 * (s_score + a_score)
    else:
        combo = s_score * a_score

    combo[~np.isfinite(combo)] = np.nan
    return combo.astype(np.float32)


# -----------------------------
# 7) MAIN TAB refs : ACE
# -----------------------------
def run_single_mineral_ace_from_tab_refs(
    img_tif_path: str,
    bands_csv: str,
    ref_root_dir: str,
    mineral: str,
    out_dir: str,
    max_refs: Optional[int] = None,
    seed: int = 0,
    band_id_is_one_based: bool = True,
    assume_img_already_normalized: bool = True,
    bg_n_samples: int = 200000,
    bg_ridge: float = 1e-3,
):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(img_tif_path) as src:
        img = src.read().astype(np.float32)   # (bands, rows, cols)
        profile = src.profile

    bands, rows, cols = img.shape

    if not assume_img_already_normalized:
        img = normalize_image_l2(img)

    valid = np.isfinite(img).all(axis=0)
    if not np.any(valid):
        raise ValueError("Aucun pixel valide (tout NaN/inf).")

    wl_img_nm, fwhm_img_nm = read_wavelengths_and_fwhm_from_csv(
        bands_csv=bands_csv,
        n_bands=bands,
        band_id_is_one_based=band_id_is_one_based
    )

    mineral_dir = Path(ref_root_dir) / mineral
    if not mineral_dir.exists():
        hits = [d for d in Path(ref_root_dir).iterdir()
                if d.is_dir() and d.name.lower() == mineral.lower()]
        if not hits:
            raise ValueError(f"Dossier du minéral introuvable: {mineral_dir}")
        mineral_dir = hits[0]

    ref_paths = sorted(str(p) for p in mineral_dir.glob("*.tab"))
    if not ref_paths:
        raise ValueError(f"Aucun fichier .tab trouvé dans {mineral_dir}")

    print(f"[INFO] {len(ref_paths)} fichiers .tab trouvés dans {mineral_dir}")

    if max_refs is not None and len(ref_paths) > int(max_refs):
        rng = np.random.default_rng(seed)
        ref_paths = list(rng.choice(ref_paths, size=int(max_refs), replace=False))
        print(f"[INFO] sous-échantillonnage: {len(ref_paths)} refs gardées (max_refs={max_refs})")

    X = img[:, valid].T
    bg_mean, bg_cov = estimate_background_stats(
        X, n_samples=bg_n_samples, ridge=bg_ridge, seed=seed
    )

    prof1 = profile.copy()
    prof1.update(count=1, dtype="float32", nodata=np.nan)

    sam_stack = []
    ace_stack = []
    used_refs = []
    skipped_refs = []

    for p in ref_paths:
        try:
            wl_ref_nm, s_ref = load_tab_spectrum(p)
        except Exception as e:
            print(f"[WARN] skip (parse): {os.path.basename(p)} ({e})")
            skipped_refs.append(os.path.basename(p))
            continue

        t = resample_spectrum_gaussian_fwhm(wl_ref_nm, s_ref, wl_img_nm, fwhm_img_nm)

        good = np.isfinite(t)
        if int(good.sum()) < 10:
            print(f"[WARN] skip (overlap<10 bands): {os.path.basename(p)}")
            skipped_refs.append(os.path.basename(p))
            continue

        img2 = img[good, :, :]
        t2 = l2_normalize_spectrum(t[good].astype(np.float32))
        if not np.isfinite(t2).all():
            print(f"[WARN] skip (target invalide): {os.path.basename(p)}")
            skipped_refs.append(os.path.basename(p))
            continue

        # SAM
        arr_rcb = np.transpose(img2, (1, 2, 0))
        t2_2d = t2.reshape(1, -1).astype(np.float32)
        sam_map = spectral_angles(arr_rcb, t2_2d).astype(np.float32)[:, :, 0]
        sam_map[~valid] = np.nan

        # ACE
        mean2 = bg_mean[good]
        cov2 = bg_cov[np.ix_(good, good)]
        ace_map = ace_map_from_background_stats(
            img_brc=img2,
            target_b=t2,
            bg_mean_b=mean2,
            bg_cov_bb=cov2,
            valid_mask_rc=valid
        )
        ace_map[~valid] = np.nan

        sam_stack.append(sam_map)
        ace_stack.append(ace_map)
        used_refs.append(os.path.basename(p))

    if len(ace_stack) == 0:
        raise ValueError("Aucune référence utilisable après parsing / recouvrement spectral.")

    sam_stack = np.stack(sam_stack, axis=0)
    ace_stack = np.stack(ace_stack, axis=0)

    sam_min = np.nanmin(sam_stack, axis=0).astype(np.float32)
    ace_max = np.nanmax(ace_stack, axis=0).astype(np.float32)

    safe_m = re.sub(r"[^a-z0-9_]+", "_", mineral.lower())
    sam_out = os.path.join(out_dir, f"sam_{safe_m}_min.tif")
    ace_out = os.path.join(out_dir, f"ace_{safe_m}_max.tif")

    combo = combine_sam_ace(
        sam_min=sam_min,
        ace_max=ace_max,
        sam_scale=0.1,
        mode="product"
    )
    combo_out = os.path.join(out_dir, f"combo_sam_ace_{safe_m}.tif")

    with rasterio.open(sam_out, "w", **prof1) as dst:
        dst.write(sam_min, 1)

    with rasterio.open(ace_out, "w", **prof1) as dst:
        dst.write(ace_max, 1)

    with rasterio.open(combo_out, "w", **prof1) as dst:
        dst.write(combo, 1)

    print(f"[OK] mineral='{mineral}' refs_used={len(used_refs)}/{len(ref_paths)}")
    print(f"     SAM_min -> {sam_out}")
    print(f"     ACE_max -> {ace_out}")
    print(f"     COMBO   -> {combo_out}")

    print("\nSpectres UTILISÉS :")
    for r in used_refs:
        print("  -", r)

    if skipped_refs:
        print("\nSpectres REJETÉS :")
        for r in skipped_refs:
            print("  -", r)

    return sam_out, ace_out, combo_out, used_refs


# -----------------------------
# 8) MAIN TXT refs : ACE
# -----------------------------
def run_single_mineral_ace_from_txt_refs(
    img_tif_path: str,
    bands_csv: str,
    ref_dir: str,
    mineral: str,
    out_dir: str,
    max_refs: Optional[int] = None,
    seed: int = 0,
    band_id_is_one_based: bool = True,
    assume_img_already_normalized: bool = True,
    bg_n_samples: int = 200000,
    bg_ridge: float = 1e-3,
):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(img_tif_path) as src:
        img = src.read().astype(np.float32)
        profile = src.profile

    bands, rows, cols = img.shape

    if not assume_img_already_normalized:
        img = normalize_image_l2(img)

    valid = np.isfinite(img).all(axis=0)
    if not np.any(valid):
        raise ValueError("Aucun pixel valide (tout NaN/inf).")

    wl_img_nm, fwhm_img_nm = read_wavelengths_and_fwhm_from_csv(
        bands_csv=bands_csv,
        n_bands=bands,
        band_id_is_one_based=band_id_is_one_based
    )

    ref_paths = find_reference_txts(ref_dir, mineral)
    if not ref_paths:
        raise ValueError(f"Aucun .txt trouvé dans {ref_dir} contenant '{mineral}'.")

    if max_refs is not None and len(ref_paths) > int(max_refs):
        rng = np.random.default_rng(seed)
        ref_paths = list(rng.choice(ref_paths, size=int(max_refs), replace=False))

    X = img[:, valid].T
    bg_mean, bg_cov = estimate_background_stats(
        X, n_samples=bg_n_samples, ridge=bg_ridge, seed=seed
    )

    prof1 = profile.copy()
    prof1.update(count=1, dtype="float32", nodata=np.nan)

    sam_stack = []
    ace_stack = []
    used_refs = []

    for p in ref_paths:
        try:
            wl_ref_nm, s_ref = load_ecostress_txt_spectrum(p)
        except Exception as e:
            print(f"[WARN] skip (parse): {os.path.basename(p)} ({e})")
            continue

        t = resample_spectrum_gaussian_fwhm(wl_ref_nm, s_ref, wl_img_nm, fwhm_img_nm)

        good = np.isfinite(t)
        if int(good.sum()) < 10:
            print(f"[WARN] skip (overlap<10 bands): {os.path.basename(p)}")
            continue

        img2 = img[good, :, :]
        t2 = l2_normalize_spectrum(t[good].astype(np.float32))
        if not np.isfinite(t2).all():
            print(f"[WARN] skip (target invalide): {os.path.basename(p)}")
            continue

        # SAM
        arr_rcb = np.transpose(img2, (1, 2, 0))
        t2_2d = t2.reshape(1, -1).astype(np.float32)
        sam_map = spectral_angles(arr_rcb, t2_2d).astype(np.float32)[:, :, 0]
        sam_map[~valid] = np.nan

        # ACE
        mean2 = bg_mean[good]
        cov2 = bg_cov[np.ix_(good, good)]
        ace_map = ace_map_from_background_stats(
            img_brc=img2,
            target_b=t2,
            bg_mean_b=mean2,
            bg_cov_bb=cov2,
            valid_mask_rc=valid
        )
        ace_map[~valid] = np.nan

        sam_stack.append(sam_map)
        ace_stack.append(ace_map)
        used_refs.append(os.path.basename(p))

    if len(ace_stack) == 0:
        raise ValueError("Aucune référence utilisable après parsing / recouvrement spectral.")

    sam_stack = np.stack(sam_stack, axis=0)
    ace_stack = np.stack(ace_stack, axis=0)

    sam_min = np.nanmin(sam_stack, axis=0).astype(np.float32)
    ace_max = np.nanmax(ace_stack, axis=0).astype(np.float32)

    safe_m = re.sub(r"[^a-z0-9_]+", "_", mineral.lower())
    sam_out = os.path.join(out_dir, f"sam_{safe_m}_min.tif")
    ace_out = os.path.join(out_dir, f"ace_{safe_m}_max.tif")

    combo = combine_sam_ace(
        sam_min=sam_min,
        ace_max=ace_max,
        sam_scale=0.1,
        mode="product"
    )
    combo_out = os.path.join(out_dir, f"combo_sam_ace_{safe_m}.tif")

    with rasterio.open(sam_out, "w", **prof1) as dst:
        dst.write(sam_min, 1)

    with rasterio.open(ace_out, "w", **prof1) as dst:
        dst.write(ace_max, 1)

    with rasterio.open(combo_out, "w", **prof1) as dst:
        dst.write(combo, 1)

    print(f"[OK] mineral='{mineral}' refs_used={len(used_refs)}/{len(ref_paths)}")
    print(f"     SAM_min -> {sam_out}")
    print(f"     ACE_max -> {ace_out}")
    print(f"     COMBO   -> {combo_out}")

    print("\nSpectres de référence utilisés :")
    for r in used_refs:
        print("  -", r)

    return sam_out, ace_out, combo_out, used_refs