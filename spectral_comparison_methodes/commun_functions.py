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


# -----------------------------
# 1) Wavelengths image (CSV)
# -----------------------------
def read_wavelengths_and_fwhm_from_csv(
    bands_csv: str,
    n_bands: int,
    band_id_is_one_based: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(bands_csv, sep=None, engine="python").copy()

    if "band_id" not in df.columns or "wavelength_nm" not in df.columns:
        raise ValueError(
            f"Le CSV doit contenir au moins band_id et wavelength_nm. Colonnes: {list(df.columns)}"
        )

    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

    if "fwhm_nm" in df.columns:
        fwhm_col = "fwhm_nm"
    else:
        if df.shape[1] < 4:
            raise ValueError(
                "Le CSV n'a pas de colonne 'fwhm_nm' et n'a pas 4 colonnes pour récupérer la 4e."
            )
        fwhm_col = df.columns[3]

    df[fwhm_col] = df[fwhm_col].astype(float)
    df["band_index"] = df["band_id"] - (1 if band_id_is_one_based else 0)

    if df["band_index"].min() < 0 or df["band_index"].max() >= n_bands:
        raise ValueError("Incohérence band_id vs nb de bandes de l'image.")

    wl = np.full(n_bands, np.nan, dtype=np.float32)
    fwhm = np.full(n_bands, np.nan, dtype=np.float32)

    idx = df["band_index"].to_numpy()
    wl[idx] = df["wavelength_nm"].to_numpy(dtype=np.float32)
    fwhm[idx] = df[fwhm_col].to_numpy(dtype=np.float32)

    if np.isnan(wl).any():
        miss = np.where(np.isnan(wl))[0]
        raise ValueError(f"Le CSV ne couvre pas toutes les bandes (wl). Ex: {miss[:10]}")
    if np.isnan(fwhm).any():
        miss = np.where(np.isnan(fwhm))[0]
        raise ValueError(f"Le CSV ne couvre pas toutes les bandes (fwhm). Ex: {miss[:10]}")

    if np.any(fwhm <= 0):
        bad = np.where(fwhm <= 0)[0][:10]
        raise ValueError(f"FWHM invalides (<=0) pour bandes: {bad}")

    return wl.astype(np.float32), fwhm.astype(np.float32)


# -----------------------------
# 2) RELAB - LOAD .tag spectrum
# -----------------------------

def load_tab_spectrum(
    tab_path: str,
    min_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge un spectre depuis un fichier .tab et retourne :
        wavelengths_nm (float32), reflectance (float32)

    Hypothèses souples :
    - fichier ASCII tabulaire
    - ignore lignes commentaires (#, ;, //)
    - séparateurs : tab, espaces, virgules
    - conserve les 2 premières colonnes numériques (wl, valeur)
    - heuristique unité :
        si max(wl) < 50 → µm → conversion nm
    """

    wavelengths = []
    values = []

    with open(tab_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", ";", "//")):
                continue

            # uniformiser séparateurs
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]

            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    pass

            if len(floats) >= 2:
                wavelengths.append(floats[0])
                values.append(floats[1])

    if len(wavelengths) < int(min_points):
        raise ValueError(f"Spectre invalide (trop peu de points): {tab_path}")

    wl = np.asarray(wavelengths, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)

    # nettoyage
    good = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[good], y[good]

    # tri
    order = np.argsort(wl)
    wl, y = wl[order], y[order]

    # µm → nm (heuristique standard spectral)
    if wl.size and np.nanmax(wl) < 50:
        wl = wl * 1000.0

    # dédoublonnage
    wl_unique, idx = np.unique(wl, return_index=True)
    y = y[idx]
    wl = wl_unique

    if wl.size < int(min_points):
        raise ValueError(f"Spectre trop court après nettoyage: {tab_path}")

    return wl.astype(np.float32), y.astype(np.float32)


# -----------------------------
# 2) Parse ECOSTRESS TXT spectrum
# -----------------------------
def load_ecostress_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge un spectre ECOSTRESS .txt (ASCII) et retourne:
        wavelengths_nm (float32), reflectance (float32)

    Le format exact peut varier. Cette fonction essaye d'être robuste:
    - ignore les lignes commentaires (#, ;, //)
    - accepte séparateurs espaces, tabs, virgules
    - garde les 2 premières colonnes numériques (wl, valeur)
    - détecte l'unité wl (nm vs µm) via heuristique
    """
    wavelengths = []
    values = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", ";", "//")):
                continue

            # remplace virgules par espaces pour simplifier
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]

            # on cherche au moins 2 floats sur la ligne
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    pass

            if len(floats) >= 2:
                wavelengths.append(floats[0])
                values.append(floats[1])

    if len(wavelengths) < 10:
        raise ValueError(f"Impossible de parser un spectre valide (trop peu de points): {txt_path}")

    wl = np.array(wavelengths, dtype=np.float32)
    y = np.array(values, dtype=np.float32)

    # retire NaN/inf
    good = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[good], y[good]

    # trier par wl
    order = np.argsort(wl)
    wl, y = wl[order], y[order]

    # heuristique unité:
    # - si max wl < 50 -> probablement µm
    # - si max wl entre 50 et 1000 -> parfois nm (VNIR) ou µm*100? rare
    # - en pratique ECOSTRESS VSWIR est souvent en µm
    if np.nanmax(wl) < 50:
        wl = wl * 1000.0  # µm -> nm

    # dédoublonne (np.interp aime pas les x identiques)
    wl_unique, idx = np.unique(wl, return_index=True)
    y = y[idx]
    wl = wl_unique

    return wl.astype(np.float32), y.astype(np.float32)


def find_reference_txts(ref_dir: str, mineral: str) -> List[str]:
    q = mineral.lower()
    paths = []
    for p in Path(ref_dir).glob("*.txt"):
        name = p.name.lower()
        if q in name and "ancillary" not in name:
            paths.append(str(p))
    paths.sort()
    return paths


# -----------------------------
# 3) Resampling + preprocessing
# -----------------------------

def resample_spectrum_gaussian_fwhm(
    wl_src_nm: np.ndarray,
    s_src: np.ndarray,
    wl_dst_nm: np.ndarray,
    fwhm_dst_nm: np.ndarray,
    min_weight: float = 1e-8
) -> np.ndarray:
    """
    Resampling 'capteur' par convolution gaussienne:
    - chaque bande dst est une gaussienne centrée sur wl_dst_nm[i]
      avec sigma = fwhm/2.355
    - renvoie NaN si pas de recouvrement spectral (poids trop faible)
    """
    wl_src_nm = np.asarray(wl_src_nm, dtype=np.float32)
    s_src = np.asarray(s_src, dtype=np.float32)
    wl_dst_nm = np.asarray(wl_dst_nm, dtype=np.float32)
    fwhm_dst_nm = np.asarray(fwhm_dst_nm, dtype=np.float32)

    # enlever NaN dans la source
    good_src = np.isfinite(wl_src_nm) & np.isfinite(s_src)
    wl = wl_src_nm[good_src]
    s = s_src[good_src]

    if wl.size < 2:
        return np.full_like(wl_dst_nm, np.nan, dtype=np.float32)

    out = np.full(wl_dst_nm.shape, np.nan, dtype=np.float32)

    for i in range(wl_dst_nm.size):
        c = float(wl_dst_nm[i])
        fwhm = float(fwhm_dst_nm[i])
        if not np.isfinite(c) or not np.isfinite(fwhm) or fwhm <= 0:
            continue

        sigma = fwhm / 2.355
        # poids gaussien
        w = np.exp(-0.5 * ((wl - c) / sigma) ** 2).astype(np.float32)
        ws = float(np.sum(w))

        if not np.isfinite(ws) or ws < min_weight:
            continue

        out[i] = float(np.sum(w * s) / ws)

    return out.astype(np.float32)


def l2_normalize_spectrum(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.sqrt(np.nansum(s * s)))
    if not np.isfinite(n) or n < eps:
        return np.full_like(s, np.nan, dtype=np.float32)
    return (s / n).astype(np.float32)


def normalize_image_l2(img_brc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt(np.nansum(img_brc * img_brc, axis=0)).astype(np.float32)
    norm = np.where(norm < eps, np.nan, norm)
    return (img_brc / norm).astype(np.float32)



# -----------------------------
# 4) Matched Filter: background stats
# -----------------------------
def estimate_background_stats(
    X: np.ndarray,
    n_samples: int = 200000,
    ridge: float = 1e-3,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("X est vide (0 pixels).")

    k = min(int(n_samples), n)
    idx = rng.choice(n, size=k, replace=False)
    bg = X[idx, :].astype(np.float32)

    mean = bg.mean(axis=0)
    Xc = bg - mean
    cov = (Xc.T @ Xc) / max(k - 1, 1)

    b = cov.shape[0]
    cov = cov + (ridge * np.trace(cov) / max(b, 1)) * np.eye(b, dtype=np.float32)

    return mean.astype(np.float32), cov.astype(np.float32)


def sam_to_score(sam: np.ndarray, sam_scale: float = 0.1) -> np.ndarray:
    """
    Convertit l'angle SAM en score [0..1] (plus grand = meilleur).
    sam_scale en radians ~ 0.1 est un bon départ (à ajuster).
    """
    sam = sam.astype(np.float32)
    out = np.exp(-sam / float(sam_scale)).astype(np.float32)
    out[~np.isfinite(sam)] = np.nan
    return out