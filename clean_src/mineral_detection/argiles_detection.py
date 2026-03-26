from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


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
    prob_zero_outside_mask: bool = True,
    write_outputs: bool = False,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection argiles sur image déjà prétraitée.

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

    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # Triplet principal ~2200
    b2100, lam2100 = band_map[2100.0]
    b2200, lam2200 = band_map[2200.0]
    b2300, lam2300 = band_map[2300.0]

    # Triplet contrôle ~1900
    has_bd1900 = False
    if use_bd1900_control:
        try:
            b1800, lam1800 = band_map[1800.0]
            b1900, lam1900 = band_map[1900.0]
            b2000, lam2000 = band_map[2000.0]
            has_bd1900 = True
        except KeyError:
            has_bd1900 = False

    out_bd2200 = outdir / f"BD2200_{target_name}.tif"
    out_bd1900 = outdir / "BD1900_control.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        R2100 = src.read(b2100).astype(np.float32)
        R2200 = src.read(b2200).astype(np.float32)
        R2300 = src.read(b2300).astype(np.float32)

        valid = (
            np.isfinite(R2100) &
            np.isfinite(R2200) &
            np.isfinite(R2300)
        )

        if has_bd1900:
            R1800 = src.read(b1800).astype(np.float32)
            R1900 = src.read(b1900).astype(np.float32)
            R2000 = src.read(b2000).astype(np.float32)

            valid &= (
                np.isfinite(R1800) &
                np.isfinite(R1900) &
                np.isfinite(R2000)
            )

        bd2200 = band_depth(R2100, R2200, R2300, lam2100, lam2200, lam2300).astype(np.float32)
        bd2200[~valid] = np.nan

        if has_bd1900:
            bd1900 = band_depth(R1800, R1900, R2000, lam1800, lam1900, lam2000).astype(np.float32)
            bd1900[~valid] = np.nan
        else:
            bd1900 = None

        # Masque
        if has_bd1900:
            mask_bool = valid & (bd2200 > bd2200_thresh) & (bd1900 > bd1900_thresh)
        else:
            mask_bool = valid & (bd2200 > bd2200_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # Score
        s2200 = normalize01(bd2200, bd2200_score_min, bd2200_score_max)

        if has_bd1900:
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

            if has_bd1900:
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
        if write_outputs:
            print("Créés :")
            print(" -", out_bd2200)
            if has_bd1900:
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

    return {
        "outputs": {
            "bd2200": out_bd2200 if write_outputs else None,
            "bd1900": out_bd1900 if (write_outputs and has_bd1900) else None,
            "mask": out_mask if write_outputs else None,
            "prob": out_prob if write_outputs else None,
        },
        "bands_1based_and_lambda_nm": {
            2100.0: (int(b2100), float(lam2100)),
            2200.0: (int(b2200), float(lam2200)),
            2300.0: (int(b2300), float(lam2300)),
            **({
                1800.0: (int(b1800), float(lam1800)),
                1900.0: (int(b1900), float(lam1900)),
                2000.0: (int(b2000), float(lam2000)),
            } if has_bd1900 else {})
        },
        "params": {
            "bd2200_thresh": float(bd2200_thresh),
            "bd1900_thresh": float(bd1900_thresh),
            "bd2200_score_min": float(bd2200_score_min),
            "bd2200_score_max": float(bd2200_score_max),
            "bd1900_score_min": float(bd1900_score_min),
            "bd1900_score_max": float(bd1900_score_max),
            "use_bd1900_control": bool(use_bd1900_control),
            "has_bd1900": bool(has_bd1900),
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