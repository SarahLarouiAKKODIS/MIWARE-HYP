from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


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
    prob_zero_outside_mask: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection pyroxène sur image déjà prétraitée.

    - BD1um = band_depth(900, 1000, 1200)
    - BD2um = band_depth(1800, 2000, 2300)
    - masque = (BD1um > seuil) & (BD2um > seuil)
    - prob = sqrt(score_BD1um * score_BD2um)
    """

    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [900, 1000, 1200, 1800, 2000, 2300]

    # -----------------------------
    # Bandes
    # -----------------------------
    df = pd.read_csv(bands_csv)
    bands = closest_band_dict(df, targets_nm)

    b900, lam900 = bands[900.0]
    b1000, lam1000 = bands[1000.0]
    b1200, lam1200 = bands[1200.0]
    b1800, lam1800 = bands[1800.0]
    b2000, lam2000 = bands[2000.0]
    b2300, lam2300 = bands[2300.0]

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # -----------------------------
    # Lecture image
    # -----------------------------
    with rasterio.open(tif_path) as src:

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

        # -----------------------------
        # Band depth
        # -----------------------------
        bd1um = band_depth(R900, R1000, R1200, lam900, lam1000, lam1200).astype(np.float32)
        bd2um = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype(np.float32)

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

        out_bd1um = outdir / f"BD1um_{target_name}.tif"
        out_bd2um = outdir / f"BD2um_{target_name}.tif"
        out_mask = outdir / f"{target_name}_mask.tif"
        out_prob = outdir / f"{target_name}_probability.tif"

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
        print("Créés :")
        print(" -", out_bd1um)
        print(" -", out_bd2um)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection pyroxène :", "OUI" if detected else "NON")
        print("Pixels détectés :", n_pixels)

    return {
        "outputs": {
            "bd1um": out_bd1um,
            "bd2um": out_bd2um,
            "mask": out_mask,
            "prob": out_prob,
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }