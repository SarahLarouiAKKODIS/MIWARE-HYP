from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


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
    prob_zero_outside_mask: bool = True,
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection oxydes de fer sur image déjà prétraitée.

    - BD900 = band_depth(860, 900, 940)
    - REDNESS = (R650 - R550) / (R650 + R550)
    - masque = (BD900 > bd900_thresh) & (REDNESS > redness_thresh)
    - score = sqrt(norm(BD900) * norm(REDNESS))
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [550, 650, 860, 900, 940]

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

    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    b860, lam860 = band_map[860.0]
    b900, lam900 = band_map[900.0]
    b940, lam940 = band_map[940.0]
    b650, lam650 = band_map[650.0]
    b550, lam550 = band_map[550.0]

    out_bd900 = outdir / "BD900_iron_oxides.tif"
    out_redness = outdir / "REDNESS_iron_oxides.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        R860 = src.read(b860).astype(np.float32)
        R900 = src.read(b900).astype(np.float32)
        R940 = src.read(b940).astype(np.float32)
        R650 = src.read(b650).astype(np.float32)
        R550 = src.read(b550).astype(np.float32)

        valid = (
            np.isfinite(R860) &
            np.isfinite(R900) &
            np.isfinite(R940) &
            np.isfinite(R650) &
            np.isfinite(R550)
        )

        # 1) BD900
        bd900 = band_depth(R860, R900, R940, lam860, lam900, lam940).astype(np.float32)
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
        if write_outputs:
            print("Créés :", str(out_bd900), str(out_redness), str(out_mask), str(out_prob))
        print("Détection oxydes de fer :", "OUI" if detected else "NON")
        print("Nombre de pixels à 255 :", n_pixels)
        if prob_stats is not None:
            print(
                "Probability score on detected pixels (min/mean/max):",
                prob_stats["min"], prob_stats["mean"], prob_stats["max"]
            )

    return {
        "outputs": {
            "bd900": out_bd900 if write_outputs else None,
            "redness": out_redness if write_outputs else None,
            "mask": out_mask if write_outputs else None,
            "prob": out_prob if write_outputs else None,
        },
        "bands_1based_and_lambda_nm": {
            860.0: (int(b860), float(lam860)),
            900.0: (int(b900), float(lam900)),
            940.0: (int(b940), float(lam940)),
            650.0: (int(b650), float(lam650)),
            550.0: (int(b550), float(lam550)),
        },
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