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
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection carbonates sur image déjà prétraitée.

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

    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # Triplet principal ~2330
    b2200, lam2200 = band_map[2200.0]
    b2330, lam2330 = band_map[2330.0]
    b2450, lam2450 = band_map[2450.0]

    # Triplet optionnel ~2500
    has_bd2500 = False
    if use_bd2500:
        try:
            b2400, lam2400 = band_map[2400.0]
            b2500, lam2500 = band_map[2500.0]
            b2600, lam2600 = band_map[2600.0]
            has_bd2500 = True
        except KeyError:
            has_bd2500 = False


    out_bd2330 = outdir / f"BD2330_{target_name}.tif"
    out_bd2500 = outdir / f"BD2500_{target_name}.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        R2200 = src.read(b2200).astype(np.float32)
        R2330 = src.read(b2330).astype(np.float32)
        R2450 = src.read(b2450).astype(np.float32)

        valid = (
            np.isfinite(R2200) &
            np.isfinite(R2330) &
            np.isfinite(R2450)
        )

        if has_bd2500:
            R2400 = src.read(b2400).astype(np.float32)
            R2500 = src.read(b2500).astype(np.float32)
            R2600 = src.read(b2600).astype(np.float32)

            valid &= (
                np.isfinite(R2400) &
                np.isfinite(R2500) &
                np.isfinite(R2600)
            )

        bd2330 = band_depth(R2200, R2330, R2450, lam2200, lam2330, lam2450).astype(np.float32)
        bd2330[~valid] = np.nan

        if has_bd2500:
            bd2500 = band_depth(R2400, R2500, R2600, lam2400, lam2500, lam2600).astype(np.float32)
            bd2500[~valid] = np.nan
        else:
            bd2500 = None

        # Masque
        if has_bd2500:
            mask_bool = valid & (bd2330 > bd2330_thresh) & (bd2500 > bd2500_thresh)
        else:
            mask_bool = valid & (bd2330 > bd2330_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # Score
        s2330 = normalize01(bd2330, bd2330_score_min, bd2330_score_max)
        if has_bd2500:
            s2500 = normalize01(bd2500, bd2500_score_min, bd2500_score_max)
            prob = np.sqrt(s2330 * s2500).astype(np.float32)
        else:
            prob = s2330.astype(np.float32)

        prob[~valid] = np.nan

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype(np.float32)

        # Profils
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        with rasterio.open(out_bd2330, "w", **prof_f) as dst:
            dst.write(bd2330, 1)

        if has_bd2500:
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
        print("Créés :")
        print(" -", out_bd2330)
        if has_bd2500:
            print(" -", out_bd2500)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection carbonates :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)
        if prob_stats is not None:
            print(
                "Probability score on detected pixels (min/mean/max) :",
                prob_stats["min"],
                prob_stats["mean"],
                prob_stats["max"],
            )

    return {
        "outputs": {
            "bd2330": out_bd2330,
            "bd2500": out_bd2500 if has_bd2500 else None,
            "mask": out_mask,
            "prob": out_prob,
        },
        "bands_1based_and_lambda_nm": {
            2200.0: (int(b2200), float(lam2200)),
            2330.0: (int(b2330), float(lam2330)),
            2450.0: (int(b2450), float(lam2450)),
            **({
                2400.0: (int(b2400), float(lam2400)),
                2500.0: (int(b2500), float(lam2500)),
                2600.0: (int(b2600), float(lam2600)),
            } if has_bd2500 else {})
        },
        "params": {
            "bd2330_thresh": float(bd2330_thresh),
            "bd2500_thresh": float(bd2500_thresh),
            "bd2330_score_min": float(bd2330_score_min),
            "bd2330_score_max": float(bd2330_score_max),
            "bd2500_score_min": float(bd2500_score_min),
            "bd2500_score_max": float(bd2500_score_max),
            "use_bd2500": bool(use_bd2500),
            "has_bd2500": bool(has_bd2500),
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }