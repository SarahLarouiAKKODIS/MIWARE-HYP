from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from utils.enmap_indices_calculation_utils import band_depth
from utils.enmap_metadata_utils import closest_band_dict


def detect_olivine_bd1050_bd2000_clean(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "olivine",
    targets_nm: list[float] | None = None,
    # Seuils du masque
    bd1050_thresh: float = 0.05,
    bd2000_max: float = 0.02,
    # Paramètres du score
    bd1050_score_min: float = 0.00,
    bd1050_score_max: float = 0.15,
    bd2000_good_min: float = 0.00,
    bd2000_good_max: float = 0.06,
    prob_zero_outside_mask: bool = True,
    # Outputs
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection d'olivine sur image hyperspectrale déjà prétraitée :
    - image déjà nettoyée / lissée / normalisée
    - végétation et eau déjà masquées
    - bandes mauvaises déjà retirées

    Méthode :
      - BD1050 = band_depth(860, 1050, 1280)
      - BD2000 = band_depth(1800, 2000, 2300)
      - masque = (BD1050 > bd1050_thresh) & (BD2000 < bd2000_max)
      - score/probabilité = sqrt(score_BD1050 * score_inverse_BD2000)

    Sorties :
      - BD1050_{target_name}.tif
      - BD2000_control.tif
      - {target_name}_mask.tif
      - {target_name}_probability.tif
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [860, 1050, 1280, 1800, 2000, 2300]

    # -----------------------------
    # Lecture de la table des bandes
    # -----------------------------
    df = pd.read_csv(bands_csv)
    bands = closest_band_dict(df, targets_nm)

    b860, lam860 = bands[860.0]
    b1050, lam1050 = bands[1050.0]
    b1280, lam1280 = bands[1280.0]
    b1800, lam1800 = bands[1800.0]
    b2000, lam2000 = bands[2000.0]
    b2300, lam2300 = bands[2300.0]

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # -----------------------------
    # Lecture de l'image prétraitée
    # -----------------------------
    with rasterio.open(tif_path) as src:
        # rasterio attend des indices 1-based
        R860 = src.read(b860).astype(np.float32)
        R1050 = src.read(b1050).astype(np.float32)
        R1280 = src.read(b1280).astype(np.float32)
        R1800 = src.read(b1800).astype(np.float32)
        R2000 = src.read(b2000).astype(np.float32)
        R2300 = src.read(b2300).astype(np.float32)

        # pixels valides = toutes les bandes utiles finies
        valid = (
            np.isfinite(R860) &
            np.isfinite(R1050) &
            np.isfinite(R1280) &
            np.isfinite(R1800) &
            np.isfinite(R2000) &
            np.isfinite(R2300)
        )

        # -----------------------------
        # Calcul des band depths
        # -----------------------------
        bd1050 = band_depth(R860, R1050, R1280, lam860, lam1050, lam1280).astype(np.float32)
        bd2000 = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype(np.float32)

        bd1050[~valid] = np.nan
        bd2000[~valid] = np.nan

        # -----------------------------
        # Masque binaire
        # -----------------------------
        mask_bool = valid & (bd1050 > bd1050_thresh) & (bd2000 < bd2000_max)
        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # -----------------------------
        # Score / probabilité
        # -----------------------------
        s1050 = normalize01(bd1050, bd1050_score_min, bd1050_score_max)

        # BD2000 est un critère de rejet :
        # plus BD2000 est fort, moins la probabilité est bonne
        s2000_bad = normalize01(bd2000, bd2000_good_min, bd2000_good_max)
        s2000 = 1.0 - s2000_bad

        prob = np.sqrt(s1050 * s2000).astype(np.float32)
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

        out_bd1050 = outdir / f"BD1050_{target_name}.tif"
        out_bd2000 = outdir / "BD2000_control.tif"
        out_mask = outdir / f"{target_name}_mask.tif"
        out_prob = outdir / f"{target_name}_probability.tif"

        with rasterio.open(out_bd1050, "w", **prof_f) as dst:
            dst.write(bd1050, 1)

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
        print(" -", out_bd1050)
        print(" -", out_bd2000)
        print(" -", out_mask)
        print(" -", out_prob)
        print("Détection :", "OUI" if detected else "NON")
        print("Nombre de pixels à 255 :", n_pixels)
        if prob_stats is not None:
            print(
                "Probability score on detected pixels (min/mean/max) :",
                prob_stats["min"],
                prob_stats["mean"],
                prob_stats["max"],
            )

    return {
        "outputs": {
            "bd1050": out_bd1050,
            "bd2000": out_bd2000,
            "mask": out_mask,
            "prob": out_prob,
        },
        "bands_1based_and_lambda_nm": {
            860.0: (int(b860), float(lam860)),
            1050.0: (int(b1050), float(lam1050)),
            1280.0: (int(b1280), float(lam1280)),
            1800.0: (int(b1800), float(lam1800)),
            2000.0: (int(b2000), float(lam2000)),
            2300.0: (int(b2300), float(lam2300)),
        },
        "params": {
            "bd1050_thresh": float(bd1050_thresh),
            "bd2000_max": float(bd2000_max),
            "bd1050_score_min": float(bd1050_score_min),
            "bd1050_score_max": float(bd1050_score_max),
            "bd2000_good_min": float(bd2000_good_min),
            "bd2000_good_max": float(bd2000_good_max),
            "prob_zero_outside_mask": bool(prob_zero_outside_mask),
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }