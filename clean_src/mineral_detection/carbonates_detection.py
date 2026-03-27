from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


def detect_carbonates_bd2330_bd2500_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "carbonates",
    targets_nm: list[float] | None = None,
    # Seuils masque
    bd2330_thresh: float = 0.03,
    bd2500_thresh: float = 0.02,
    # Normalisation score [0..1]
    bd2330_score_min: float = 0.00,
    bd2330_score_max: float = 0.10,
    bd2500_score_min: float = 0.00,
    bd2500_score_max: float = 0.08,
    # Options
    use_bd2500: bool = False,
    sampling_mode: str = "nearest",   # "nearest" | "linear"
    interpolation_max_gap_nm: float | None = 40.0,
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection carbonates sur image déjà prétraitée.

    Modes d'échantillonnage spectral :
    - sampling_mode="nearest" : prend la bande la plus proche
    - sampling_mode="linear"  : interpole linéairement à la longueur d'onde cible

    Indices :
    - BD2330 = band_depth(2200, 2330, 2450)
    - Optionnel BD2500 = band_depth(2400, 2500, 2600)

    masque :
      - si BD2500 dispo/utilisé : (BD2330 > seuil) & (BD2500 > seuil)
      - sinon : (BD2330 > seuil)

    score :
      - sans BD2500 : norm(BD2330)
      - avec BD2500 : sqrt(norm(BD2330) * norm(BD2500))
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [2200, 2330, 2450, 2400, 2500, 2600]

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

    out_bd2330 = outdir / f"BD2330_{target_name}.tif"
    out_bd2500 = outdir / f"BD2500_{target_name}.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        has_bd2500 = False

        if sampling_mode == "nearest":
            band_map = closest_band_dict(df, targets_nm)

            # Triplet principal ~2330
            b2200, lam2200 = band_map[2200.0]
            b2330, lam2330 = band_map[2330.0]
            b2450, lam2450 = band_map[2450.0]

            R2200 = src.read(b2200).astype(np.float32)
            R2330 = src.read(b2330).astype(np.float32)
            R2450 = src.read(b2450).astype(np.float32)

            valid = (
                np.isfinite(R2200) &
                np.isfinite(R2330) &
                np.isfinite(R2450)
            )

            bd2330 = band_depth(R2200, R2330, R2450, lam2200, lam2330, lam2450).astype(np.float32)

            spectral_info = {
                "sampling_mode": "nearest",
                "bd2330_bands": {
                    "2200": (int(b2200), float(lam2200)),
                    "2330": (int(b2330), float(lam2330)),
                    "2450": (int(b2450), float(lam2450)),
                }
            }

            # Triplet optionnel ~2500
            if use_bd2500:
                try:
                    b2400, lam2400 = band_map[2400.0]
                    b2500, lam2500 = band_map[2500.0]
                    b2600, lam2600 = band_map[2600.0]
                    has_bd2500 = True

                    R2400 = src.read(b2400).astype(np.float32)
                    R2500 = src.read(b2500).astype(np.float32)
                    R2600 = src.read(b2600).astype(np.float32)

                    valid &= (
                        np.isfinite(R2400) &
                        np.isfinite(R2500) &
                        np.isfinite(R2600)
                    )

                    bd2500 = band_depth(R2400, R2500, R2600, lam2400, lam2500, lam2600).astype(np.float32)

                    spectral_info["bd2500_bands"] = {
                        "2400": (int(b2400), float(lam2400)),
                        "2500": (int(b2500), float(lam2500)),
                        "2600": (int(b2600), float(lam2600)),
                    }
                except KeyError:
                    has_bd2500 = False
                    bd2500 = None
            else:
                bd2500 = None

        else:
            cube = src.read().astype(np.float32)  # (bands, rows, cols)
            wavelengths_nm = build_wavelength_vector_from_csv(src.count)

            order = np.argsort(wavelengths_nm)
            wavelengths_nm = wavelengths_nm[order]
            cube = cube[order, :, :]

            # Triplet principal ~2330
            R2200, idx_2200, wl_2200 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2200.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2330, idx_2330, wl_2330 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2330.0, max_gap_nm=interpolation_max_gap_nm
            )
            R2450, idx_2450, wl_2450 = interpolate_band_linear_from_cube(
                cube, wavelengths_nm, 2450.0, max_gap_nm=interpolation_max_gap_nm
            )

            valid = (
                np.isfinite(R2200) &
                np.isfinite(R2330) &
                np.isfinite(R2450)
            )

            bd2330 = band_depth(R2200, R2330, R2450, 2200.0, 2330.0, 2450.0).astype(np.float32)

            spectral_info = {
                "sampling_mode": "linear",
                "interpolation_max_gap_nm": interpolation_max_gap_nm,
                "bd2330_interp_support": {
                    "2200": {"band_indices": tuple(map(int, idx_2200)), "wavelengths_nm": tuple(map(float, wl_2200))},
                    "2330": {"band_indices": tuple(map(int, idx_2330)), "wavelengths_nm": tuple(map(float, wl_2330))},
                    "2450": {"band_indices": tuple(map(int, idx_2450)), "wavelengths_nm": tuple(map(float, wl_2450))},
                }
            }

            # Triplet optionnel ~2500
            if use_bd2500:
                try:
                    R2400, idx_2400, wl_2400 = interpolate_band_linear_fromCube = None
                except Exception:
                    pass
                try:
                    R2400, idx_2400, wl_2400 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2400.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R2500, idx_2500, wl_2500 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2500.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    R2600, idx_2600, wl_2600 = interpolate_band_linear_from_cube(
                        cube, wavelengths_nm, 2600.0, max_gap_nm=interpolation_max_gap_nm
                    )
                    has_bd2500 = True

                    valid &= (
                        np.isfinite(R2400) &
                        np.isfinite(R2500) &
                        np.isfinite(R2600)
                    )

                    bd2500 = band_depth(R2400, R2500, R2600, 2400.0, 2500.0, 2600.0).astype(np.float32)

                    spectral_info["bd2500_interp_support"] = {
                        "2400": {"band_indices": tuple(map(int, idx_2400)), "wavelengths_nm": tuple(map(float, wl_2400))},
                        "2500": {"band_indices": tuple(map(int, idx_2500)), "wavelengths_nm": tuple(map(float, wl_2500))},
                        "2600": {"band_indices": tuple(map(int, idx_2600)), "wavelengths_nm": tuple(map(float, wl_2600))},
                    }
                except ValueError:
                    has_bd2500 = False
                    bd2500 = None
            else:
                bd2500 = None

        bd2330[~valid] = np.nan
        if has_bd2500 and bd2500 is not None:
            bd2500[~valid] = np.nan

        # Masque
        if has_bd2500 and bd2500 is not None:
            mask_bool = valid & (bd2330 > bd2330_thresh) & (bd2500 > bd2500_thresh)
        else:
            mask_bool = valid & (bd2330 > bd2330_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # Score
        s2330 = normalize01(bd2330, bd2330_score_min, bd2330_score_max)
        if has_bd2500 and bd2500 is not None:
            s2500 = normalize01(bd2500, bd2500_score_min, bd2500_score_max)
            prob = np.sqrt(s2330 * s2500).astype(np.float32)
        else:
            prob = s2330.astype(np.float32)

        prob[~valid] = np.nan

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype(np.float32)

        # Écriture
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        with rasterio.open(out_bd2330, "w", **prof_f) as dst:
            dst.write(bd2330, 1)

        if has_bd2500 and bd2500 is not None:
            with rasterio.open(out_bd2500, "w", **prof_f) as dst:
                dst.write(bd2500, 1)

        with rasterio.open(out_mask, "w", **prof_u8) as dst:
            dst.write(mask, 1)

        with rasterio.open(out_prob, "w", **prof_f) as dst:
            dst.write(prob, 1)

    # Stats
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
        print(" -", out_bd2330)
        if has_bd2500 and bd2500 is not None:
            print(" -", out_bd2500)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection carbonates :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)

        if sampling_mode == "nearest":
            print("Bandes utilisées (nearest):", spectral_info)
        else:
            print("Supports d'interpolation:", spectral_info)

    return {
        "outputs": {
            "bd2330": out_bd2330,
            "bd2500": out_bd2500 if (has_bd2500 and bd2500 is not None) else None,
            "mask": out_mask,
            "prob": out_prob,
        },
        "spectral_sampling": spectral_info,
        "params": {
            "bd2330_thresh": float(bd2330_thresh),
            "bd2500_thresh": float(bd2500_thresh),
            "bd2330_score_min": float(bd2330_score_min),
            "bd2330_score_max": float(bd2330_score_max),
            "bd2500_score_min": float(bd2500_score_min),
            "bd2500_score_max": float(bd2500_score_max),
            "use_bd2500": bool(use_bd2500),
            "has_bd2500": bool(has_bd2500),
            "sampling_mode": sampling_mode,
            "interpolation_max_gap_nm": interpolation_max_gap_nm,
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }