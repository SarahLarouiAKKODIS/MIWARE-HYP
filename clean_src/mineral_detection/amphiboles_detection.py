from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


def detect_amphiboles_bd2320_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "amphiboles",
    targets_nm: list[float] | None = None,
    # Seuils
    bd2320_thresh: float = 0.03,
    bd2000_thresh: float = 0.02,
    # Score
    bd2320_score_min: float = 0.00,
    bd2320_score_max: float = 0.10,
    bd2000_score_min: float = 0.00,
    bd2000_score_max: float = 0.08,
    # Options
    use_bd2000_control: bool = True,
    sampling_mode: str = "nearest",   # "nearest" | "linear"
    interpolation_max_gap_nm: float | None = 40.0,
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection amphiboles sur image prétraitée.

    Modes d'échantillonnage spectral :
    - sampling_mode="nearest" : prend la bande la plus proche
    - sampling_mode="linear"  : interpole linéairement à la longueur d'onde cible

    Indices :
    - BD2320 = band_depth(2250, 2320, 2390)
    - Option: BD2000 = band_depth(1900, 2000, 2100)

    masque :
        BD2320 > seuil ET (optionnel) BD2000 > seuil
    prob :
        sqrt(score2320 * score2000) ou score2320 seul
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [1900, 2000, 2100, 2250, 2320, 2390]

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
        band_index = df["band_id"].to_numpy() - 1  # suppose CSV 1-based
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
        out_bd2320 = outdir / f"BD2320_{target_name}.tif"
        out_mask = outdir / f"{target_name}_mask.tif"
        out_prob = outdir / f"{target_name}_probability.tif"
        out_bd2000 = outdir / "BD2000_control.tif"

        has_bd2000 = False

        if sampling_mode == "nearest":
            band_map = closest_band_dict(df, targets_nm)

            b2250, lam2250 = band_map[2250.0]
            b2320, lam2320 = band_map[2320.0]
            b2390, lam2390 = band_map[2390.0]

            R2250 = src.read(b2250).astype(np.float32)
            R2320 = src.read(b2320).astype(np.float32)
            R2390 = src.read(b2390).astype(np.float32)

            valid = (
                np.isfinite(R2250) &
                np.isfinite(R2320) &
                np.isfinite(R2390)
            )

            bd2320 = band_depth(R2250, R2320, R2390, lam2250, lam2320, lam2390).astype(np.float32)

            spectral_info = {
                "sampling_mode": "nearest",
                "bd2320_bands": {
                    "2250": (int(b2250), float(lam2250)),
                    "2320": (int(b2320), float(lam2320)),
                    "2390": (int(b2390), float(lam2390)),
                }
            }

            if use_bd2000_control:
                try:
                    b1900, lam1900 = band_map[1900.0]
                    b2000, lam2000 = band_map[2000.0]
                    b2100, lam2100 = band_map[2100.0]
                    has_bd2000 = True

                    R1900 = src.read(b1900).astype(np.float32)
                    R2000 = src.read(b2000).astype(np.float32)
                    R2100 = src.read(b2100).astype(np.float32)

                    valid &= (
                        np.isfinite(R1900) &
                        np.isfinite(R2000) &
                        np.isfinite(R2100)
                    )

                    bd2000 = band_depth(R1900, R2000, R2100, lam1900, lam2000, lam2100).astype(np.float32)

                    spectral_info["bd2000_bands"] = {
                        "1900": (int(b1900), float(lam1900)),
                        "2000": (int(b2000), float(lam2000)),
                        "2100": (int(b2100), float(lam2100)),
                    }
                except KeyError:
                    has_bd2000 = False
                    bd2000 = None
            else:
                bd2000 = None

        else:
            cube = src.read().astype(np.float32)  # (bands, rows, cols)
            wavelengths_nm = build_wavelength_vector_from_csv(src.count)

            order = np.argsort(wavelengths_nm)
            wavelengths_nm = wavelengths_nm[order]
            cube = cube[order, :, :]

            R2250, idx_2250, wl_2250 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2250.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2320, idx_2320, wl_2320 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2320.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2390, idx_2390, wl_2390 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2390.0, max_gap_nm=interpolation_max_gap_nm
            )

            valid = (
                np.isfinite(R2250) &
                np.isfinite(R2320) &
                np.isfinite(R2390)
            )

            bd2320 = band_depth(R2250, R2320, R2390, 2250.0, 2320.0, 2390.0).astype(np.float32)

            spectral_info = {
                "sampling_mode": "linear",
                "interpolation_max_gap_nm": interpolation_max_gap_nm,
                "bd2320_interp_support": {
                    "2250": {"band_indices": tuple(map(int, idx_2250)), "wavelengths_nm": tuple(map(float, wl_2250))},
                    "2320": {"band_indices": tuple(map(int, idx_2320)), "wavelengths_nm": tuple(map(float, wl_2320))},
                    "2390": {"band_indices": tuple(map(int, idx_2390)), "wavelengths_nm": tuple(map(float, wl_2390))},
                }
            }

            if use_bd2000_control:
                try:
                    R1900, idx_1900, wl_1900 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 1900.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R2000, idx_2000, wl_2000 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2000.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R2100, idx_2100, wl_2100 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2100.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    has_bd2000 = True

                    valid &= (
                        np.isfinite(R1900) &
                        np.isfinite(R2000) &
                        np.isfinite(R2100)
                    )

                    bd2000 = band_depth(R1900, R2000, R2100, 1900.0, 2000.0, 2100.0).astype(np.float32)

                    spectral_info["bd2000_interp_support"] = {
                        "1900": {"band_indices": tuple(map(int, idx_1900)), "wavelengths_nm": tuple(map(float, wl_1900))},
                        "2000": {"band_indices": tuple(map(int, idx_2000)), "wavelengths_nm": tuple(map(float, wl_2000))},
                        "2100": {"band_indices": tuple(map(int, idx_2100)), "wavelengths_nm": tuple(map(float, wl_2100))},
                    }
                except ValueError:
                    has_bd2000 = False
                    bd2000 = None
            else:
                bd2000 = None

        # -----------------------------
        # Masques NaN
        # -----------------------------
        bd2320[~valid] = np.nan
        if has_bd2000 and bd2000 is not None:
            bd2000[~valid] = np.nan

        # -----------------------------
        # Masque
        # -----------------------------
        if has_bd2000 and bd2000 is not None:
            mask_bool = valid & (bd2320 > bd2320_thresh) & (bd2000 > bd2000_thresh)
        else:
            mask_bool = valid & (bd2320 > bd2320_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # -----------------------------
        # Score
        # -----------------------------
        s2320 = normalize01(bd2320, bd2320_score_min, bd2320_score_max)

        if has_bd2000 and bd2000 is not None:
            s2000 = normalize01(bd2000, bd2000_score_min, bd2000_score_max)
            prob = np.sqrt(s2320 * s2000).astype(np.float32)
        else:
            prob = s2320.astype(np.float32)

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

        with rasterio.open(out_bd2320, "w", **prof_f) as dst:
            dst.write(bd2320, 1)

        if has_bd2000 and bd2000 is not None:
            with rasterio.open(out_bd2000, "w", **prof_f) as dst:
                dst.write(bd2000, 1)

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
        print(" -", out_bd2320)
        if has_bd2000 and bd2000 is not None:
            print(" -", out_bd2000)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection amphiboles :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)

        if sampling_mode == "nearest":
            print("Bandes utilisées (nearest):", spectral_info)
        else:
            print("Supports d'interpolation:", spectral_info)

    return {
        "outputs": {
            "bd2320": out_bd2320,
            "bd2000": out_bd2000 if (has_bd2000 and bd2000 is not None) else None,
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