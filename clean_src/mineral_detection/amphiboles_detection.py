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
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection amphiboles sur image prétraitée.

    - BD2320 = band_depth(2250, 2320, 2390)
    - Option: BD2000 = band_depth(1900, 2000, 2100)
    - masque:
        BD2320 > seuil ET (optionnel) BD2000 > seuil
    - prob:
        sqrt(BD2320 * BD2000) ou BD2320 seul
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [1900, 2000, 2100, 2250, 2320, 2390]

    # -----------------------------
    # Bandes
    # -----------------------------
    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    b2250, lam2250 = band_map[2250.0]
    b2320, lam2320 = band_map[2320.0]
    b2390, lam2390 = band_map[2390.0]

    has_bd2000 = False
    if use_bd2000_control:
        try:
            b1900, lam1900 = band_map[1900.0]
            b2000, lam2000 = band_map[2000.0]
            b2100, lam2100 = band_map[2100.0]
            has_bd2000 = True
        except KeyError:
            has_bd2000 = False

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # -----------------------------
    # Lecture image
    # -----------------------------
    with rasterio.open(tif_path) as src:

        R2250 = src.read(b2250).astype(np.float32)
        R2320 = src.read(b2320).astype(np.float32)
        R2390 = src.read(b2390).astype(np.float32)

        valid = (
            np.isfinite(R2250) &
            np.isfinite(R2320) &
            np.isfinite(R2390)
        )

        if has_bd2000:
            R1900 = src.read(b1900).astype(np.float32)
            R2000 = src.read(b2000).astype(np.float32)
            R2100 = src.read(b2100).astype(np.float32)

            valid &= (
                np.isfinite(R1900) &
                np.isfinite(R2000) &
                np.isfinite(R2100)
            )

        # -----------------------------
        # Band depth
        # -----------------------------
        bd2320 = band_depth(R2250, R2320, R2390, lam2250, lam2320, lam2390).astype(np.float32)
        bd2320[~valid] = np.nan

        if has_bd2000:
            bd2000 = band_depth(R1900, R2000, R2100, lam1900, lam2000, lam2100).astype(np.float32)
            bd2000[~valid] = np.nan
        else:
            bd2000 = None

        # -----------------------------
        # Masque
        # -----------------------------
        if has_bd2000:
            mask_bool = valid & (bd2320 > bd2320_thresh) & (bd2000 > bd2000_thresh)
        else:
            mask_bool = valid & (bd2320 > bd2320_thresh)

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # -----------------------------
        # Score
        # -----------------------------
        s2320 = normalize01(bd2320, bd2320_score_min, bd2320_score_max)

        if has_bd2000:
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

        out_bd2320 = outdir / f"BD2320_{target_name}.tif"
        out_mask = outdir / f"{target_name}_mask.tif"
        out_prob = outdir / f"{target_name}_probability.tif"
        out_bd2000 = outdir / "BD2000_control.tif"

        with rasterio.open(out_bd2320, "w", **prof_f) as dst:
            dst.write(bd2320, 1)

        if has_bd2000:
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
        print("Créés :")
        print(" -", out_bd2320)
        if has_bd2000:
            print(" -", out_bd2000)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection amphiboles :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)

    return {
        "outputs": {
            "bd2320": out_bd2320,
            "bd2000": out_bd2000 if has_bd2000 else None,
            "mask": out_mask,
            "prob": out_prob,
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }