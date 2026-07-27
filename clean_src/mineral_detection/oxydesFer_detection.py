from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from ..utils.enmap_indices_calculation_utils import band_depth
from ..utils.enmap_metadata_utils import closest_band_dict


def detect_iron_oxides_bd900_redness_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "iron_oxides",
    targets_nm: list[float] | None = None,
    # Seuils masque
    bd900_thresh: float = 0.04,
    redness_thresh: float = 0.05,
    # Normalisation score [0..1]
    bd900_score_min: float = 0.00,
    bd900_score_max: float = 0.12,
    red_score_min: float = 0.00,
    red_score_max: float = 0.20,
    # Masquage pixels sombres pour stabiliser le ratio
    dark_den_thresh: float = 1e-3,
    dark_den_thresh_raw: float = 100.0,
    assume_reflectance_01: bool = True,
    # Options
    sampling_mode: str = "nearest",   # "nearest" | "linear"
    interpolation_max_gap_nm: float | None = 80.0,
    prob_zero_outside_mask: bool = True,
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection oxydes de fer sur image déjà prétraitée.

    Modes d'échantillonnage spectral :
    - sampling_mode="nearest" : prend la bande la plus proche
    - sampling_mode="linear"  : interpole linéairement à la longueur d'onde cible

    Indices :
    - BD900 = band_depth(860, 900, 940)
    - REDNESS = (R650 - R550) / (R650 + R550)

    masque :
      - (BD900 > bd900_thresh) & (REDNESS > redness_thresh)

    score :
      - sqrt(norm(BD900) * norm(REDNESS))
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [550, 650, 860, 900, 940]

    if sampling_mode not in {"nearest", "linear"}:
        raise ValueError("sampling_mode doit être 'nearest' ou 'linear'.")

    def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = a + b
        out = np.full(a.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (denom != 0)
        out[valid] = ((a[valid] - b[valid]) / denom[valid]).astype(np.float32)
        return out

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    df = pd.read_csv(bands_csv).copy()

    if "band_id" not in df.columns or "wavelength_nm" not in df.columns:
        raise ValueError("Le CSV doit contenir au minimum 'band_id' et 'wavelength_nm'.")

    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

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

    out_bd900 = outdir / "BD900_iron_oxides.tif"
    out_redness = outdir / "REDNESS_iron_oxides.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        if sampling_mode == "nearest":
            band_map = closest_band_dict(df, targets_nm)

            b860, lam860 = band_map[860.0]
            b900, lam900 = band_map[900.0]
            b940, lam940 = band_map[940.0]
            b650, lam650 = band_map[650.0]
            b550, lam550 = band_map[550.0]

            R860 = src.read(b860).astype(np.float32)
            R900 = src.read(b900).astype(np.float32)
            R940 = src.read(b940).astype(np.float32)
            R650 = src.read(b650).astype(np.float32)
            R550 = src.read(b550).astype(np.float32)

            spectral_info = {
                "sampling_mode": "nearest",
                "bd900_bands": {
                    "860": (int(b860), float(lam860)),
                    "900": (int(b900), float(lam900)),
                    "940": (int(b940), float(lam940)),
                },
                "redness_bands": {
                    "650": (int(b650), float(lam650)),
                    "550": (int(b550), float(lam550)),
                },
            }

        else:
            cube = src.read().astype(np.float32)  # (bands, rows, cols)
            wavelengths_nm = build_wavelength_vector_from_csv(src.count)

            order = np.argsort(wavelengths_nm)
            wavelengths_nm = wavelengths_nm[order]
            cube = cube[order, :, :]

            R860, idx_860, wl_860 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 860.0, max_gap_nm=interpolation_max_gap_nm
            )
            R900, idx_900, wl_900 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 900.0, max_gap_nm=interpolation_max_gap_nm
            )
            R940, idx_940, wl_940 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 940.0, max_gap_nm=interpolation_max_gap_nm
            )
            R650, idx_650, wl_650 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 650.0, max_gap_nm=interpolation_max_gap_nm
            )
            R550, idx_550, wl_550 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 550.0, max_gap_nm=interpolation_max_gap_nm
            )

            spectral_info = {
                "sampling_mode": "linear",
                "interpolation_max_gap_nm": interpolation_max_gap_nm,
                "bd900_interp_support": {
                    "860": {"band_indices": tuple(map(int, idx_860)), "wavelengths_nm": tuple(map(float, wl_860))},
                    "900": {"band_indices": tuple(map(int, idx_900)), "wavelengths_nm": tuple(map(float, wl_900))},
                    "940": {"band_indices": tuple(map(int, idx_940)), "wavelengths_nm": tuple(map(float, wl_940))},
                },
                "redness_interp_support": {
                    "650": {"band_indices": tuple(map(int, idx_650)), "wavelengths_nm": tuple(map(float, wl_650))},
                    "550": {"band_indices": tuple(map(int, idx_550)), "wavelengths_nm": tuple(map(float, wl_550))},
                },
            }

        valid = (
            np.isfinite(R860) &
            np.isfinite(R900) &
            np.isfinite(R940) &
            np.isfinite(R650) &
            np.isfinite(R550)
        )

        # 1) BD900
        if sampling_mode == "nearest":
            bd900 = band_depth(R860, R900, R940, lam860, lam900, lam940).astype(np.float32)
        else:
            bd900 = band_depth(R860, R900, R940, 860.0, 900.0, 940.0).astype(np.float32)

        bd900[~valid] = np.nan

        # 2) Redness
        redness = safe_norm_diff(R650, R550).astype(np.float32)
        redness = np.clip(redness, -1.0, 1.0)

        den = R650 + R550
        if assume_reflectance_01:
            dark = ~np.isfinite(den) | (den <= float(dark_den_thresh))
        else:
            dark = ~np.isfinite(den) | (den <= float(dark_den_thresh_raw))

        redness[dark] = np.nan
        redness[~valid] = np.nan

        if verbose:
            try:
                print("redness min/max:", float(np.nanmin(redness)), float(np.nanmax(redness)))
                print("percentiles:", np.nanpercentile(redness, [1, 5, 50, 95, 99]))
            except Exception:
                pass
            print("redness_thresh:", redness_thresh)
            print("bd900_thresh:", bd900_thresh)

        # masque
        mask_bool = valid & np.isfinite(redness) & (bd900 > bd900_thresh) & (redness > redness_thresh)

        if verbose:
            n_det = int(mask_bool.sum())
            n_fin = int(np.isfinite(bd900).sum())
            print("pixels détectés:", n_det, "sur", n_fin)
            if n_fin > 0:
                print("ratio (%):", 100.0 * float(n_det) / float(n_fin))

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # score
        s_bd900 = normalize01(bd900, bd900_score_min, bd900_score_max)
        s_red = normalize01(redness, red_score_min, red_score_max)
        prob = np.sqrt(s_bd900 * s_red).astype(np.float32)
        prob[~valid] = np.nan

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype(np.float32)

        if write_outputs:
            prof_f = src.profile.copy()
            prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

            prof_u8 = src.profile.copy()
            prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

            with rasterio.open(out_bd900, "w", **prof_f) as dst:
                dst.write(bd900, 1)

            with rasterio.open(out_redness, "w", **prof_f) as dst:
                dst.write(redness, 1)

            with rasterio.open(out_mask, "w", **prof_u8) as dst:
                dst.write(mask, 1)

            with rasterio.open(out_prob, "w", **prof_f) as dst:
                dst.write(prob, 1)

    n_pixels = int(np.sum(mask == 255))
    detected = n_pixels > 0

    prob_stats = None
    if detected:
        p = prob[mask == 255]
        if p.size > 0:
            prob_stats = {
                "min": float(np.nanmin(p)),
                "mean": float(np.nanmean(p)),
                "max": float(np.nanmax(p)),
            }

    if verbose:
        print("Mode spectral :", sampling_mode)
        if write_outputs:
            print("Créés :", str(out_bd900), str(out_redness), str(out_mask), str(out_prob))
        print("Détection oxydes de fer :", "OUI" if detected else "NON")
        print("Nombre de pixels à 255 :", n_pixels)
        if prob_stats is not None:
            print(
                "Probability score on detected pixels (min/mean/max):",
                prob_stats["min"], prob_stats["mean"], prob_stats["max"]
            )

        if sampling_mode == "nearest":
            print("Bandes utilisées (nearest):", spectral_info)
        else:
            print("Supports d'interpolation:", spectral_info)

    return {
        "outputs": {
            "bd900": out_bd900 if write_outputs else None,
            "redness": out_redness if write_outputs else None,
            "mask": out_mask if write_outputs else None,
            "prob": out_prob if write_outputs else None,
        },
        "spectral_sampling": spectral_info,
        "params": {
            "bd900_thresh": float(bd900_thresh),
            "redness_thresh": float(redness_thresh),
            "bd900_score_min": float(bd900_score_min),
            "bd900_score_max": float(bd900_score_max),
            "red_score_min": float(red_score_min),
            "red_score_max": float(red_score_max),
            "assume_reflectance_01": bool(assume_reflectance_01),
            "dark_den_thresh": float(dark_den_thresh),
            "dark_den_thresh_raw": float(dark_den_thresh_raw),
            "sampling_mode": sampling_mode,
            "interpolation_max_gap_nm": interpolation_max_gap_nm,
            "write_outputs": bool(write_outputs),
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
        "arrays": {
            "bd900": bd900,
            "redness": redness,
            "mask": mask,
            "prob": prob,
        },
    }