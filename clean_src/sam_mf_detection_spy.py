#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import rasterio

from spectral.algorithms import spectral_angles
from spectral.algorithms.detectors import matched_filter


# -----------------------------
# 1) Wavelengths image (CSV)
# -----------------------------
def read_wavelengths_from_csv(
    bands_csv: str,
    n_bands: int,
    band_id_is_one_based: bool = True
) -> np.ndarray:
    """
    CSV attendu: band_id, wavelength_nm, ...
    Retourne wavelengths_nm alignées sur l'ordre des bandes du raster.
    """
    df = pd.read_csv(bands_csv, sep=None, engine="python").copy()
    required = {"band_id", "wavelength_nm"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le CSV doit contenir {required}. Colonnes: {list(df.columns)}")

    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)
    df["band_index"] = df["band_id"] - (1 if band_id_is_one_based else 0)

    if df["band_index"].min() < 0 or df["band_index"].max() >= n_bands:
        raise ValueError("Incohérence band_id vs nb de bandes de l'image.")

    wl = np.full(n_bands, np.nan, dtype=float)
    wl[df["band_index"].to_numpy()] = df["wavelength_nm"].to_numpy()

    if np.isnan(wl).any():
        miss = np.where(np.isnan(wl))[0]
        raise ValueError(f"Le CSV ne couvre pas toutes les bandes. Ex: {miss[:10]}")

    return wl.astype(np.float32)

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



def find_tab_spectra_for_mineral(ref_root: str, mineral: str) -> List[str]:
    """
    Cherche les fichiers .tab dans :
        ref_root/mineral/

    Exemple :
        spectral_lib/
            actinolite/
                spec1.tab
                spec2.tab
    """
    mineral_dir = Path(ref_root) / mineral.lower()

    if not mineral_dir.exists():
        raise ValueError(f"Dossier minéral introuvable: {mineral_dir}")

    paths = sorted(str(p) for p in mineral_dir.glob("*.tab"))

    if not paths:
        raise ValueError(f"Aucun fichier .tab trouvé dans {mineral_dir}")

    return paths

# -----------------------------
# 2) Parse USGS TXT spectrum
# -----------------------------

def load_splib_txt_spectrum_only_allowed_spectrometers(
    txt_path: str,
    allowed_spectrometers: Iterable[str] = ("BECKb", "ASDFRb"),
    min_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge un spectre .txt de type splib/USGS (ex: splib07a_Actinolite_HS22.3B_BECKb_AREF.txt)
    et retourne:
        wavelengths_nm (float32), reflectance (float32)

    + Filtre sur spectromètre ENCODÉ DANS LE NOM:
        - on n'accepte que les fichiers dont le nom contient un spectromètre autorisé
          (par défaut: BECKb ou ASDFRb)
        - parsing robuste (comme ta fonction ECOSTRESS):
            * ignore commentaires (#, ;, //)
            * accepte séparateurs espaces, tabs, virgules
            * garde les 2 premières colononnes numériques (wl, valeur)
            * heuristique unité: si max(wl) < 50 => µm -> nm

    Hypothèse sur le nom: champs séparés par "_" et le spectromètre est le 4ème champ.
        splib07a_<mineral>_<sample>_<spectrometer>_<type>.txt
    """
    p = Path(txt_path)
    parts = p.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de fichier inattendu (attendu au moins 4 champs séparés par '_'): {p.name}")

    spectrometer = parts[3]
    allowed = {s.lower() for s in allowed_spectrometers}
    if spectrometer.lower() not in allowed:
        raise ValueError(
            f"Spectromètre non autorisé '{spectrometer}' pour {p.name}. "
            f"Autorisés: {sorted(allowed)}"
        )

    wavelengths = []
    values = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", ";", "//")):
                continue

            s = s.replace(",", " ")
            parts_line = [x for x in s.split() if x]

            floats = []
            for x in parts_line:
                try:
                    floats.append(float(x))
                except ValueError:
                    pass

            if len(floats) >= 2:
                wavelengths.append(floats[0])
                values.append(floats[1])

    if len(wavelengths) < int(min_points):
        raise ValueError(f"Impossible de parser un spectre valide (trop peu de points): {txt_path}")

    wl = np.asarray(wavelengths, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)

    good = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[good], y[good]

    order = np.argsort(wl)
    wl, y = wl[order], y[order]

    # Heuristique unité (µm -> nm)
    if wl.size and np.nanmax(wl) < 50:
        wl = wl * 1000.0

    # Dé-doublonnage
    wl_unique, idx = np.unique(wl, return_index=True)
    y = y[idx]
    wl = wl_unique

    if wl.size < int(min_points):
        raise ValueError(f"Spectre trop court après nettoyage: {txt_path}")

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
def l2_normalize_spectrum(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.sqrt(np.nansum(s * s)))
    if not np.isfinite(n) or n < eps:
        return np.full_like(s, np.nan, dtype=np.float32)
    return (s / n).astype(np.float32)


def resample_spectrum_linear(
    wl_src_nm: np.ndarray,
    s_src: np.ndarray,
    wl_dst_nm: np.ndarray
) -> np.ndarray:
    """
    Rééchantillonnage linéaire d'un spectre défini sur wl_src vers wl_dst.
    En dehors de la plage -> NaN.
    """
    wl_src_nm = np.asarray(wl_src_nm, dtype=np.float32)
    s_src = np.asarray(s_src, dtype=np.float32)
    wl_dst_nm = np.asarray(wl_dst_nm, dtype=np.float32)
    return np.interp(wl_dst_nm, wl_src_nm, s_src, left=np.nan, right=np.nan).astype(np.float32)


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


def robust_logistic_score(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Convertit x en score [0..1] via une logistique robuste:
      center = median(x), scale = MAD(x)
    k contrôle la pente (plus grand = transition plus douce).
    """
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.full_like(x, np.nan, dtype=np.float32)

    med = np.nanmedian(x[finite])
    mad = np.nanmedian(np.abs(x[finite] - med))
    # MAD -> approx sigma (gauss)
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = np.nanstd(x[finite])
    if not np.isfinite(sigma) or sigma < 1e-6:
        # tout plat
        out = np.zeros_like(x, dtype=np.float32)
        out[finite] = 0.5
        out[~finite] = np.nan
        return out

    z = (x - med) / (k * sigma)
    out = 1.0 / (1.0 + np.exp(-z))
    out[~finite] = np.nan
    return out.astype(np.float32)


def sam_to_score(sam: np.ndarray, sam_scale: float = 0.1) -> np.ndarray:
    """
    Convertit l'angle SAM en score [0..1] (plus grand = meilleur).
    sam_scale en radians ~ 0.1 est un bon départ (à ajuster).
    """
    sam = sam.astype(np.float32)
    out = np.exp(-sam / float(sam_scale)).astype(np.float32)
    out[~np.isfinite(sam)] = np.nan
    return out


def combine_sam_mf(
    sam_min: np.ndarray,
    mf_max: np.ndarray,
    sam_scale: float = 0.1,
    mf_k: float = 3.0,
    mode: str = "product",   # "product" (AND soft) ou "mean"
) -> np.ndarray:
    """
    Combine SAM (min) et MF (max) en un score [0..1].
    """
    s_score = sam_to_score(sam_min, sam_scale=sam_scale)
    m_score = robust_logistic_score(mf_max, k=mf_k)

    if mode == "mean":
        combo = 0.5 * (s_score + m_score)
    else:
        combo = s_score * m_score  # AND soft, plus sélectif

    combo[~np.isfinite(combo)] = np.nan
    return combo.astype(np.float32)

# -----------------------------
# 5) MAIN: single mineral, multi TXT refs, SAM min + MF max
# -----------------------------
def run_single_mineral_detection_from_txt_refs(
    img_tif_path: str,
    bands_csv: str,
    ref_dir: str,                       # dossier ECOSTRESS .txt
    mineral: str,                       # mot-clé de filtrage
    out_dir: str,
    max_refs: Optional[int] = None,     # limite (ex: 10) ou None=all
    seed: int = 0,
    band_id_is_one_based: bool = True,
    assume_img_already_normalized: bool = True,
    mf_n_bg_samples: int = 200000,
    mf_ridge: float = 1e-3,
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Read image (GeoTIFF) ---
    with rasterio.open(img_tif_path) as src:
        img = src.read().astype(np.float32)   # (bands, rows, cols)
        profile = src.profile

    bands, rows, cols = img.shape

    # --- normalize image if needed ---
    if not assume_img_already_normalized:
        img = normalize_image_l2(img)

    # masque valide (NaN + inf)
    valid = np.isfinite(img).all(axis=0)
    if not np.any(valid):
        raise ValueError("Aucun pixel valide (tout NaN/inf).")


    # --- wavelengths image (nm) ---
    wl_img_nm = read_wavelengths_from_csv(
        bands_csv=bands_csv,
        n_bands=bands,
        band_id_is_one_based=band_id_is_one_based
    )

    # --- normalize image if needed ---
    if not assume_img_already_normalized:
        img = normalize_image_l2(img)

    # --- gather reference txt files ---
    ref_paths = find_reference_txts(ref_dir, mineral)
    if not ref_paths:
        raise ValueError(f"Aucun .txt trouvé dans {ref_dir} contenant '{mineral}'.")

    # optional subset
    if max_refs is not None and len(ref_paths) > int(max_refs):
        rng = np.random.default_rng(seed)
        ref_paths = list(rng.choice(ref_paths, size=int(max_refs), replace=False))

    # --- background stats for MF (computed once on full bands) ---
    X = img[:, valid].T
    bg_mean, bg_cov = estimate_background_stats(
        X, n_samples=mf_n_bg_samples, ridge=mf_ridge, seed=seed
    )

    # output profile for 1-band float32
    prof1 = profile.copy()
    prof1.update(count=1, dtype="float32", nodata=np.nan)

    sam_stack = []
    mf_stack = []
    used_refs = []

    for p in ref_paths:
        # load ref spectrum 
        try:
            # wl_ref_nm, s_ref = load_ecostress_txt_spectrum(p) # from txt (ECOSTRESS)
            wl_ref_nm, s_ref = load_splib_txt_spectrum_only_allowed_spectrometers(
            p, allowed_spectrometers=("BECKb", "ASDFRb") #from txt (USGS)
        )
           
        except Exception as e:
            print(f"[WARN] skip (parse): {os.path.basename(p)} ({e})")
            continue

        # resample reference to image bands
        t = resample_spectrum_linear(wl_ref_nm, s_ref, wl_img_nm)

        good = np.isfinite(t)
        if int(good.sum()) < 10:
            print(f"[WARN] skip (overlap<10 bands): {os.path.basename(p)}")
            continue

        img2 = img[good, :, :]
        t2 = l2_normalize_spectrum(t[good].astype(np.float32))

        # SAM
        arr_rcb = np.transpose(img2, (1, 2, 0))  # doit être (rows, cols, bands_good)

        # sécurité: forcer 3D
        if arr_rcb.ndim != 3:
            raise ValueError(f"arr_rcb doit être 3D (rows, cols, bands). Reçu shape={arr_rcb.shape}")

        # sécurité: t2 doit être 1D et longueur = bands_good
        t2 = np.asarray(t2, dtype=np.float32).reshape(-1)
        if t2.shape[0] != arr_rcb.shape[2]:
            raise ValueError(f"Incohérence SAM: len(t2)={t2.shape[0]} vs bands={arr_rcb.shape[2]}")

        print("img shape:", img.shape)
        print("img2 shape:", img2.shape)
        print("arr_rcb shape:", arr_rcb.shape)
        print("t2 shape:", t2.shape)
        print("good sum:", int(good.sum()))

        t2_2d = t2.reshape(1, -1).astype(np.float32)     # (1, bands)
        sam_map = spectral_angles(arr_rcb, t2_2d).astype(np.float32)  # (rows, cols, 1)
        sam_map = sam_map[:, :, 0].astype(np.float32)    # -> (rows, cols)

        sam_map[~valid] = np.nan

        # MF (reduce mean/cov to good bands)
        mean2 = bg_mean[good]
        cov2 = bg_cov[np.ix_(good, good)]

        arr_rcb_clean = np.where(np.isfinite(arr_rcb), arr_rcb, 0.0).astype(np.float32)
        mf_map = matched_filter(arr_rcb_clean, t2).astype(np.float32)
        mf_map[~valid] = np.nan

        sam_stack.append(sam_map)
        mf_stack.append(mf_map)
        used_refs.append(os.path.basename(p))

    if len(sam_stack) == 0:
        raise ValueError("Aucune référence utilisable après parsing / recouvrement spectral.")

    sam_stack = np.stack(sam_stack, axis=0)  # (k, rows, cols)
    mf_stack = np.stack(mf_stack, axis=0)

    # Aggregation for reliability
    sam_min = np.nanmin(sam_stack, axis=0).astype(np.float32)
    mf_max = np.nanmax(mf_stack, axis=0).astype(np.float32)

    safe_m = re.sub(r"[^a-z0-9_]+", "_", mineral.lower())
    sam_out = os.path.join(out_dir, f"sam_{safe_m}_min.tif")
    mf_out = os.path.join(out_dir, f"mf_{safe_m}_max.tif")

    # --- COMBINAISON SAM + MF ---
    combo = combine_sam_mf(
        sam_min=sam_min,
        mf_max=mf_max,
        sam_scale=0.1,   # ajuste si besoin
        mf_k=3.0,
        mode="product"
    )

    combo_out = os.path.join(out_dir, f"combo_sam_mf_{safe_m}.tif")
    with rasterio.open(combo_out, "w", **prof1) as dst:
        dst.write(combo, 1)

    print(f"     COMBO   -> {combo_out}")

    with rasterio.open(sam_out, "w", **prof1) as dst:
        dst.write(sam_min, 1)
    with rasterio.open(mf_out, "w", **prof1) as dst:
        dst.write(mf_max, 1)

    print(f"[OK] mineral='{mineral}' refs_used={len(used_refs)}/{len(ref_paths)}")
    print(f"     SAM_min -> {sam_out}")
    print(f"     MF_max  -> {mf_out}")

    print("\nSpectres de référence utilisés :")
    for r in used_refs:
        print("  -", r)

    return sam_out, mf_out, combo_out, used_refs





def run_single_mineral_detection_from_tab_refs(
    img_tif_path: str,
    bands_csv: str,
    ref_root_dir: str,                  # racine des dossiers minerais
    mineral: str,                       # nom du dossier mineral à traiter
    out_dir: str,
    max_refs: Optional[int] = None,
    seed: int = 0,
    band_id_is_one_based: bool = True,
    assume_img_already_normalized: bool = True,
    mf_n_bg_samples: int = 200000,
    mf_ridge: float = 1e-3,
) -> Tuple[str, str, List[str]]:
    os.makedirs(out_dir, exist_ok=True)

    # --- Read image (GeoTIFF) ---
    with rasterio.open(img_tif_path) as src:
        img = src.read().astype(np.float32)   # (bands, rows, cols)
        profile = src.profile

    bands, rows, cols = img.shape

    # --- normalize image if needed ---
    if not assume_img_already_normalized:
        img = normalize_image_l2(img)

    # masque valide (NaN + inf) - après normalisation
    valid = np.isfinite(img).all(axis=0)
    if not np.any(valid):
        raise ValueError("Aucun pixel valide (tout NaN/inf).")

    # --- wavelengths image (nm) ---
    wl_img_nm = read_wavelengths_from_csv(
        bands_csv=bands_csv,
        n_bands=bands,
        band_id_is_one_based=band_id_is_one_based
    )

    # --- gather reference .tab files from mineral folder ---
    mineral_dir = Path(ref_root_dir) / mineral
    if not mineral_dir.exists():
        # fallback case-insensitive: cherche un dossier dont le nom match mineral.lower()
        hits = [d for d in Path(ref_root_dir).iterdir()
                if d.is_dir() and d.name.lower() == mineral.lower()]
        if not hits:
            raise ValueError(f"Dossier du minéral introuvable: {mineral_dir}")
        mineral_dir = hits[0]

    ref_paths = sorted(str(p) for p in mineral_dir.glob("*.tab"))
    if not ref_paths:
        raise ValueError(f"Aucun fichier .tab trouvé dans {mineral_dir}")

    print(f"[INFO] {len(ref_paths)} fichiers .tab trouvés dans {mineral_dir}")

    # optional subset
    if max_refs is not None and len(ref_paths) > int(max_refs):
        rng = np.random.default_rng(seed)
        ref_paths = list(rng.choice(ref_paths, size=int(max_refs), replace=False))
        print(f"[INFO] sous-échantillonnage: {len(ref_paths)} refs gardées (max_refs={max_refs})")

    # --- background stats for MF (computed once on full bands) ---
    X = img[:, valid].T
    bg_mean, bg_cov = estimate_background_stats(
        X, n_samples=mf_n_bg_samples, ridge=mf_ridge, seed=seed
    )

    # output profile for 1-band float32
    prof1 = profile.copy()
    prof1.update(count=1, dtype="float32", nodata=np.nan)

    sam_stack = []
    mf_stack = []
    used_refs = []
    skipped_refs = []

    for p in ref_paths:
        # load ref spectrum from .tab
        try:
            wl_ref_nm, s_ref = load_tab_spectrum(p)   # <- ta fonction
        except Exception as e:
            print(f"[WARN] skip (parse): {os.path.basename(p)} ({e})")
            skipped_refs.append(os.path.basename(p))
            continue

        # resample reference to image bands
        t = resample_spectrum_linear(wl_ref_nm, s_ref, wl_img_nm)

        good = np.isfinite(t)
        if int(good.sum()) < 10:
            print(f"[WARN] skip (overlap<10 bands): {os.path.basename(p)}")
            skipped_refs.append(os.path.basename(p))
            continue

        img2 = img[good, :, :]
        t2 = l2_normalize_spectrum(t[good].astype(np.float32))

        # SAM input shape: (rows, cols, bands_good)
        arr_rcb = np.transpose(img2, (1, 2, 0))

        t2 = np.asarray(t2, dtype=np.float32).reshape(-1)
        if t2.shape[0] != arr_rcb.shape[2]:
            print(f"[WARN] skip (shape mismatch): {os.path.basename(p)}")
            skipped_refs.append(os.path.basename(p))
            continue

        # SAM
        t2_2d = t2.reshape(1, -1).astype(np.float32)
        sam_map = spectral_angles(arr_rcb, t2_2d).astype(np.float32)  # (rows, cols, 1)
        sam_map = sam_map[:, :, 0].astype(np.float32)
        sam_map[~valid] = np.nan

        # MF: matched_filter refuse les NaN
        arr_rcb_clean = np.where(np.isfinite(arr_rcb), arr_rcb, 0.0).astype(np.float32)
        mf_map = matched_filter(arr_rcb_clean, t2).astype(np.float32)
        mf_map[~valid] = np.nan

        sam_stack.append(sam_map)
        mf_stack.append(mf_map)
        used_refs.append(os.path.basename(p))

    if len(sam_stack) == 0:
        raise ValueError("Aucune référence utilisable après parsing / recouvrement spectral.")

    sam_stack = np.stack(sam_stack, axis=0)
    mf_stack = np.stack(mf_stack, axis=0)

    # Aggregation
    sam_min = np.nanmin(sam_stack, axis=0).astype(np.float32)
    mf_max = np.nanmax(mf_stack, axis=0).astype(np.float32)

    safe_m = re.sub(r"[^a-z0-9_]+", "_", mineral.lower())
    sam_out = os.path.join(out_dir, f"sam_{safe_m}_min.tif")
    mf_out = os.path.join(out_dir, f"mf_{safe_m}_max.tif")

    # --- COMBINAISON SAM + MF ---
    combo = combine_sam_mf(
        sam_min=sam_min,
        mf_max=mf_max,
        sam_scale=0.1,
        mf_k=3.0,
        mode="product"
    )

    combo_out = os.path.join(out_dir, f"combo_sam_mf_{safe_m}.tif")
    with rasterio.open(combo_out, "w", **prof1) as dst:
        dst.write(combo, 1)

    print(f"     COMBO   -> {combo_out}")

    with rasterio.open(sam_out, "w", **prof1) as dst:
        dst.write(sam_min, 1)
    with rasterio.open(mf_out, "w", **prof1) as dst:
        dst.write(mf_max, 1)

    print(f"[OK] mineral='{mineral}' refs_used={len(used_refs)}/{len(ref_paths)}")
    print(f"     SAM_min -> {sam_out}")
    print(f"     MF_max  -> {mf_out}")

    print("\nSpectres UTILISÉS :")
    for r in used_refs:
        print("  -", r)

    if skipped_refs:
        print("\nSpectres REJETÉS :")
        for r in skipped_refs:
            print("  -", r)

    return sam_out, mf_out, combo_out, used_refs

