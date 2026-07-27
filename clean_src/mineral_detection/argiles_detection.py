from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from ..utils.enmap_indices_calculation_utils import band_depth
from ..utils.enmap_metadata_utils import closest_band_dict


def detect_argiles_bd2200_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "argiles",
    targets_nm: list[float] | None = None,
    # Seuils masque
    bd2200_thresh: float = 0.03,
    bd1900_thresh: float = 0.02,
    # Normalisation score [0..1]
    bd2200_score_min: float = 0.00,
    bd2200_score_max: float = 0.10,
    bd1900_score_min: float = 0.00,
    bd1900_score_max: float = 0.08,
    # Options
    use_bd1900_control: bool = True,
    sampling_mode: str = "nearest",   # "nearest" | "linear"
    interpolation_max_gap_nm: float | None = 80.0,
    prob_zero_outside_mask: bool = True,
    write_outputs: bool = False,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection argiles sur image déjà prétraitée.

    Modes d'échantillonnage spectral :
    - sampling_mode="nearest" : prend la bande la plus proche
    - sampling_mode="linear"  : interpole linéairement à la longueur d'onde cible

    Indices :
    - BD2200 = band_depth(2100, 2200, 2300)
    - Optionnel BD1900 = band_depth(1800, 1900, 2000)

    masque :
      - si BD1900 dispo/utilisé : (BD2200 > bd2200_thresh) & (BD1900 > bd1900_thresh)
      - sinon : (BD2200 > bd2200_thresh)

    score :
      - sans BD1900 : norm(BD2200)
      - avec BD1900 : sqrt(norm(BD2200) * norm(BD1900))
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [1800, 1900, 2000, 2100, 2200, 2300]

    if sampling_mode not in {"nearest", "linear"}:
        raise ValueError("sampling_mode doit être 'nearest' ou 'linear'.")

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

    out_bd2200 = outdir / f"BD2200_{target_name}.tif"
    out_bd1900 = outdir / "BD1900_control.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        has_bd1900 = False

        if sampling_mode == "nearest":
            band_map = closest_band_dict(df, targets_nm)

            # Triplet principal ~2200
            b2100, lam2100 = band_map[2100.0]
            b2200, lam2200 = band_map[2200.0]
            b2300, lam2300 = band_map[2300.0]

            R2100 = src.read(b2100).astype(np.float32)
            R2200 = src.read(b2200).astype(np.float32)
            R2300 = src.read(b2300).astype(np.float32)

            valid = (
                np.isfinite(R2100) &
                np.isfinite(R2200) &
                np.isfinite(R2300)
            )

            bd2200 = band_depth(R2100, R2200, R2300, lam2100, lam2200, lam2300).astype(np.float32)

            spectral_info = {
                "sampling_mode": "nearest",
                "bd2200_bands": {
                    "2100": (int(b2100), float(lam2100)),
                    "2200": (int(b2200), float(lam2200)),
                    "2300": (int(b2300), float(lam2300)),
                }
            }

            # Triplet contrôle ~1900
            if use_bd1900_control:
                try:
                    b1800, lam1800 = band_map[1800.0]
                    b1900, lam1900 = band_map[1900.0]
                    b2000, lam2000 = band_map[2000.0]
                    has_bd1900 = True

                    R1800 = src.read(b1800).astype(np.float32)
                    R1900 = src.read(b1900).astype(np.float32)
                    R2000 = src.read(b2000).astype(np.float32)

                    valid &= (
                        np.isfinite(R1800) &
                        np.isfinite(R1900) &
                        np.isfinite(R2000)
                    )

                    bd1900 = band_depth(R1800, R1900, R2000, lam1800, lam1900, lam2000).astype(np.float32)

                    spectral_info["bd1900_bands"] = {
                        "1800": (int(b1800), float(lam1800)),
                        "1900": (int(b1900), float(lam1900)),
                        "2000": (int(b2000), float(lam2000)),
                    }
                except KeyError:
                    has_bd1900 = False
                    bd1900 = None
            else:
                bd1900 = None

        else:
            cube = src.read().astype(np.float32)  # (bands, rows, cols)
            wavelengths_nm = build_wavelength_vector_from_csv(src.count)

            order = np.argsort(wavelengths_nm)
            wavelengths_nm = wavelengths_nm[order]
            cube = cube[order, :, :]

            # Triplet principal ~2200
            R2100, idx_2100, wl_2100 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2100.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2200, idx_2200, wl_2200 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2200.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2300, idx_2300, wl_2300 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2300.0, max_gap_nm=interpolation_max_gap_nm
            )

            valid = (
                np.isfinite(R2100) &
                np.isfinite(R2200) &
                np.isfinite(R2300)
            )

            bd2200 = band_depth(R2100, R2200, R2300, 2100.0, 2200.0, 2300.0).astype(np.float32)

            spectral_info = {
                "sampling_mode": "linear",
                "interpolation_max_gap_nm": interpolation_max_gap_nm,
                "bd2200_interp_support": {
                    "2100": {"band_indices": tuple(map(int, idx_2100)), "wavelengths_nm": tuple(map(float, wl_2100))},
                    "2200": {"band_indices": tuple(map(int, idx_2200)), "wavelengths_nm": tuple(map(float, wl_2200))},
                    "2300": {"band_indices": tuple(map(int, idx_2300)), "wavelengths_nm": tuple(map(float, wl_2300))},
                }
            }

            # Triplet contrôle ~1900
            if use_bd1900_control:
                try:
                    R1800, idx_1800, wl_1800 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 1800.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R1900, idx_1900, wl_1900 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 1900.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R2000, idx_2000, wl_2000 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2000.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    has_bd1900 = True

                    valid &= (
                        np.isfinite(R1800) &
                        np.isfinite(R1900) &
                        np.isfinite(R2000)
                    )

                    bd1900 = band_depth(R1800, R1900, R2000, 1800.0, 1900.0, 2000.0).astype(np.float32)

                    spectral_info["bd1900_interp_support"] = {
                        "1800": {"band_indices": tuple(map(int, idx_1800)), "wavelengths_nm": tuple(map(float, wl_1800))},
                        "1900": {"band_indices": tuple(map(int, idx_1900)), "wavelengths_nm": tuple(map(float, wl_1900))},
                        "2000": {"band_indices": tuple(map(int, idx_2000)), "wavelengths_nm": tuple(map(float, wl_2000))},
                    }
                except ValueError:
                    has_bd1900 = False
                    bd1900 = None
            else:
                bd1900 = None

        bd2200[~valid] = np.nan
        if has_bd1900 and bd1900 is not None:
            bd1900[~valid] = np.nan

        # Masque
        if has_bd1900 and bd1900 is not None:
            mask_bool = valid & (bd2200 > bd2200_thresh) & (bd1900 > bd1900_thresh)
        else:
            mask_bool = valid & (bd2200 > bd2200_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # Score
        s2200 = normalize01(bd2200, bd2200_score_min, bd2200_score_max)

        if has_bd1900 and bd1900 is not None:
            s1900 = normalize01(bd1900, bd1900_score_min, bd1900_score_max)
            prob = np.sqrt(s2200 * s1900).astype(np.float32)
        else:
            prob = s2200.astype(np.float32)

        prob[~valid] = np.nan

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype(np.float32)

        if write_outputs:
            prof_f = src.profile.copy()
            prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

            prof_u8 = src.profile.copy()
            prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

            with rasterio.open(out_bd2200, "w", **prof_f) as dst:
                dst.write(bd2200, 1)

            if has_bd1900 and bd1900 is not None:
                with rasterio.open(out_bd1900, "w", **prof_f) as dst:
                    dst.write(bd1900, 1)

            with rasterio.open(out_mask, "w", **prof_u8) as dst:
                dst.write(mask, 1)

            with rasterio.open(out_prob, "w", **prof_f) as dst:
                dst.write(prob, 1)

    # Stats
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
            print("Créés :")
            print(" -", out_bd2200)
            if has_bd1900 and bd1900 is not None:
                print(" -", out_bd1900)
            print(" -", out_mask)
            print(" -", out_prob)
        print("Détection argiles :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)
        if prob_stats is not None:
            print(
                "Probability score on detected pixels (min/mean/max) :",
                prob_stats["min"],
                prob_stats["mean"],
                prob_stats["max"],
            )

        if sampling_mode == "nearest":
            print("Bandes utilisées (nearest):", spectral_info)
        else:
            print("Supports d'interpolation:", spectral_info)

    return {
        "outputs": {
            "bd2200": out_bd2200 if write_outputs else None,
            "bd1900": out_bd1900 if (write_outputs and has_bd1900 and bd1900 is not None) else None,
            "mask": out_mask if write_outputs else None,
            "prob": out_prob if write_outputs else None,
        },
        "spectral_sampling": spectral_info,
        "params": {
            "bd2200_thresh": float(bd2200_thresh),
            "bd1900_thresh": float(bd1900_thresh),
            "bd2200_score_min": float(bd2200_score_min),
            "bd2200_score_max": float(bd2200_score_max),
            "bd1900_score_min": float(bd1900_score_min),
            "bd1900_score_max": float(bd1900_score_max),
            "use_bd1900_control": bool(use_bd1900_control),
            "has_bd1900": bool(has_bd1900),
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
            "bd2200": bd2200,
            "bd1900": bd1900,
            "mask": mask,
            "prob": prob,
        },
    }