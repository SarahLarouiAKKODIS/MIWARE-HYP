from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from ..utils.enmap_indices_calculation_utils import band_depth
from ..utils.enmap_metadata_utils import closest_band_dict


def detect_pyroxene_bd1um_bd2um_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "pyroxene",
    targets_nm: list[float] | None = None,
    # Seuils masque
    bd1um_thresh: float = 0.05,
    bd2um_thresh: float = 0.03,
    # Score
    bd1um_score_min: float = 0.00,
    bd1um_score_max: float = 0.15,
    bd2um_score_min: float = 0.00,
    bd2um_score_max: float = 0.12,
    # Echantillonnage spectral
    sampling_mode: str = "nearest",   # "nearest" | "linear"
    interpolation_max_gap_nm: float | None = 80.0,
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection pyroxène sur image déjà prétraitée.

    Modes d'échantillonnage spectral :
    - sampling_mode="nearest" : prend la bande la plus proche
    - sampling_mode="linear"  : interpole linéairement à la longueur d'onde cible

    Indices :
    - BD1um = band_depth(900, 1000, 1200)
    - BD2um = band_depth(1800, 2000, 2300)

    masque = (BD1um > seuil) & (BD2um > seuil)
    prob   = sqrt(score_BD1um * score_BD2um)
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [900, 1000, 1200, 1800, 2000, 2300]

    if sampling_mode not in {"nearest", "linear"}:
        raise ValueError("sampling_mode doit être 'nearest' ou 'linear'.")

    # -----------------------------
    # Métadonnées bandes
    # -----------------------------
    df = pd.read_csv(bands_csv).copy()

    if "band_id" not in df.columns or "wavelength_nm" not in df.columns:
        raise ValueError("Le CSV doit contenir au minimum 'band_id' et 'wavelength_nm'.")

    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def build_wavelength_vector_from_csv(n_bands: int) -> np.ndarray:
        band_index = df["band_id"].to_numpy() - 1  # on suppose band_id 1-based
        wl = np.full(n_bands, np.nan, dtype=np.float32)
        ok = (band_index >= 0) & (band_index < n_bands)
        wl[band_index[ok]] = df.loc[ok, "wavelength_nm"].to_numpy(dtype=np.float32)

        if np.isnan(wl).any():
            miss = np.where(np.isnan(wl))[0][:10]
            raise ValueError(
                f"Le CSV ne couvre pas toutes les bandes de l'image. Bandes manquantes ex: {miss}"
            )
        return wl

    def interpolate_band_linear_from_cube(
        cube_brc: np.ndarray,          # (bands, rows, cols)
        wavelengths_nm: np.ndarray,    # (bands,)
        target_nm: float,
        max_gap_nm: float | None = 80.0,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        """
        Interpolation linéaire de la réflectance à target_nm.
        Retourne :
          - raster interpolé (rows, cols)
          - indices des bandes encadrantes (i_left, i_right)
          - longueurs d'onde encadrantes (wl_left, wl_right)

        Refuse l'extrapolation.
        """
        wl = np.asarray(wavelengths_nm, dtype=np.float32)
        if wl.ndim != 1:
            raise ValueError("wavelengths_nm doit être 1D.")
        if cube_brc.ndim != 3:
            raise ValueError("cube_brc doit être de forme (bands, rows, cols).")
        if cube_brc.shape[0] != wl.size:
            raise ValueError("cube_brc et wavelengths_nm n'ont pas le même nombre de bandes.")

        if not np.all(np.isfinite(wl)):
            raise ValueError("wavelengths_nm contient des NaN/inf.")
        if np.any(np.diff(wl) <= 0):
            raise ValueError("wavelengths_nm doit être strictement croissant.")

        if target_nm < float(wl.min()) or target_nm > float(wl.max()):
            raise ValueError(
                f"Impossible d'interpoler {target_nm} nm : hors domaine spectral "
                f"[{float(wl.min())}, {float(wl.max())}] nm."
            )

        j = int(np.searchsorted(wl, target_nm))

        if j < wl.size and np.isclose(float(wl[j]), float(target_nm), atol=1e-6):
            band = cube_brc[j].astype(np.float32)
            return band, (j, j), (float(wl[j]), float(wl[j]))

        if j == 0 or j == wl.size:
            raise ValueError(f"Impossible d'interpoler {target_nm} nm sans extrapolation.")

        i_left = j - 1
        i_right = j

        wl_left = float(wl[i_left])
        wl_right = float(wl[i_right])

        if wl_right <= wl_left:
            raise ValueError(
                f"Longueurs d'onde encadrantes invalides autour de {target_nm} nm: "
                f"{wl_left}, {wl_right}"
            )

        if max_gap_nm is not None:
            if (target_nm - wl_left) > max_gap_nm or (wl_right - target_nm) > max_gap_nm:
                raise ValueError(
                    f"Interpolation refusée pour {target_nm} nm : bandes trop éloignées "
                    f"({wl_left} nm, {wl_right} nm), max_gap_nm={max_gap_nm}"
                )

        left = cube_brc[i_left].astype(np.float32)
        right = cube_brc[i_right].astype(np.float32)

        alpha = (float(target_nm) - wl_left) / (wl_right - wl_left)
        out = left + alpha * (right - left)
        return out.astype(np.float32), (i_left, i_right), (wl_left, wl_right)

    # -----------------------------
    # Lecture image
    # -----------------------------
    with rasterio.open(tif_path) as src:
        out_bd1um = outdir / f"BD1um_{target_name}.tif"
        out_bd2um = outdir / f"BD2um_{target_name}.tif"
        out_mask = outdir / f"{target_name}_mask.tif"
        out_prob = outdir / f"{target_name}_probability.tif"

        if sampling_mode == "nearest":
            bands = closest_band_dict(df, targets_nm)

            b900, lam900 = bands[900.0]
            b1000, lam1000 = bands[1000.0]
            b1200, lam1200 = bands[1200.0]
            b1800, lam1800 = bands[1800.0]
            b2000, lam2000 = bands[2000.0]
            b2300, lam2300 = bands[2300.0]

            R900 = src.read(b900).astype(np.float32)
            R1000 = src.read(b1000).astype(np.float32)
            R1200 = src.read(b1200).astype(np.float32)
            R1800 = src.read(b1800).astype(np.float32)
            R2000 = src.read(b2000).astype(np.float32)
            R2300 = src.read(b2300).astype(np.float32)

            valid = (
                np.isfinite(R900) &
                np.isfinite(R1000) &
                np.isfinite(R1200) &
                np.isfinite(R1800) &
                np.isfinite(R2000) &
                np.isfinite(R2300)
            )

            bd1um = band_depth(R900, R1000, R1200, lam900, lam1000, lam1200).astype(np.float32)
            bd2um = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype(np.float32)

            spectral_info = {
                "sampling_mode": "nearest",
                "bd1um_bands": {
                    "900": (int(b900), float(lam900)),
                    "1000": (int(b1000), float(lam1000)),
                    "1200": (int(b1200), float(lam1200)),
                },
                "bd2um_bands": {
                    "1800": (int(b1800), float(lam1800)),
                    "2000": (int(b2000), float(lam2000)),
                    "2300": (int(b2300), float(lam2300)),
                },
            }

        else:
            cube = src.read().astype(np.float32)  # (bands, rows, cols)
            wavelengths_nm = build_wavelength_vector_from_csv(src.count)

            order = np.argsort(wavelengths_nm)
            wavelengths_nm = wavelengths_nm[order]
            cube = cube[order, :, :]

            R900, idx_900, wl_900 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 900.0, max_gap_nm=interpolation_max_gap_nm
            )
            R1000, idx_1000, wl_1000 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 1000.0, max_gap_nm=interpolation_max_gap_nm
            )
            R1200, idx_1200, wl_1200 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 1200.0, max_gap_nm=interpolation_max_gap_nm
            )
            R1800, idx_1800, wl_1800 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 1800.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2000, idx_2000, wl_2000 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2000.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2300, idx_2300, wl_2300 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2300.0, max_gap_nm=interpolation_max_gap_nm
            )

            valid = (
                np.isfinite(R900) &
                np.isfinite(R1000) &
                np.isfinite(R1200) &
                np.isfinite(R1800) &
                np.isfinite(R2000) &
                np.isfinite(R2300)
            )

            # Ici on utilise les longueurs d'onde nominales cibles
            bd1um = band_depth(R900, R1000, R1200, 900.0, 1000.0, 1200.0).astype(np.float32)
            bd2um = band_depth(R1800, R2000, R2300, 1800.0, 2000.0, 2300.0).astype(np.float32)

            spectral_info = {
                "sampling_mode": "linear",
                "interpolation_max_gap_nm": interpolation_max_gap_nm,
                "bd1um_interp_support": {
                    "900": {"band_indices": tuple(map(int, idx_900)), "wavelengths_nm": tuple(map(float, wl_900))},
                    "1000": {"band_indices": tuple(map(int, idx_1000)), "wavelengths_nm": tuple(map(float, wl_1000))},
                    "1200": {"band_indices": tuple(map(int, idx_1200)), "wavelengths_nm": tuple(map(float, wl_1200))},
                },
                "bd2um_interp_support": {
                    "1800": {"band_indices": tuple(map(int, idx_1800)), "wavelengths_nm": tuple(map(float, wl_1800))},
                    "2000": {"band_indices": tuple(map(int, idx_2000)), "wavelengths_nm": tuple(map(float, wl_2000))},
                    "2300": {"band_indices": tuple(map(int, idx_2300)), "wavelengths_nm": tuple(map(float, wl_2300))},
                },
            }

        bd1um[~valid] = np.nan
        bd2um[~valid] = np.nan

        # -----------------------------
        # Masque
        # -----------------------------
        mask_bool = valid & (bd1um > bd1um_thresh) & (bd2um > bd2um_thresh)
        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # -----------------------------
        # Score
        # -----------------------------
        s1 = normalize01(bd1um, bd1um_score_min, bd1um_score_max)
        s2 = normalize01(bd2um, bd2um_score_min, bd2um_score_max)

        prob = np.sqrt(s1 * s2).astype(np.float32)
        prob[~valid] = np.nan

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype(np.float32)

        # -----------------------------
        # Écriture
        # -----------------------------
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        with rasterio.open(out_bd1um, "w", **prof_f) as dst:
            dst.write(bd1um, 1)

        with rasterio.open(out_bd2um, "w", **prof_f) as dst:
            dst.write(bd2um, 1)

        with rasterio.open(out_mask, "w", **prof_u8) as dst:
            dst.write(mask, 1)

        with rasterio.open(out_prob, "w", **prof_f) as dst:
            dst.write(prob, 1)

    # -----------------------------
    # Stats
    # -----------------------------
    n_pixels = int(np.sum(mask == 255))
    detected = n_pixels > 0

    prob_stats = None
    if n_pixels > 0:
        p = prob[mask == 255]
        if p.size > 0:
            prob_stats = {
                "min": float(np.nanmin(p)),
                "mean": float(np.nanmean(p)),
                "max": float(np.nanmax(p)),
            }

    if verbose:
        print("Mode spectral :", sampling_mode)
        print("Créés :")
        print(" -", out_bd1um)
        print(" -", out_bd2um)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection pyroxène :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)

        if sampling_mode == "nearest":
            print("Bandes utilisées (nearest):", spectral_info)
        else:
            print("Supports d'interpolation:", spectral_info)

    return {
        "outputs": {
            "bd1um": out_bd1um,
            "bd2um": out_bd2um,
            "mask": out_mask,
            "prob": out_prob,
        },
        "spectral_sampling": spectral_info,
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }