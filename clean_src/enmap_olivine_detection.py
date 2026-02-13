from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_olivine_bd1050_bd2000(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "olivine",
    targets_nm: list[float] | None = None,
    # Seuils masque
    bd1050_thresh: float = 0.05,
    bd2000_max: float = 0.02,
    # Paramètres score/probabilité
    bd1050_score_min: float = 0.00,
    bd1050_score_max: float = 0.15,
    bd2000_good_min: float = 0.00,
    bd2000_good_max: float = 0.06,
    prob_zero_outside_mask: bool = True,
    # --- POST-TRAITEMENT : exclusion végétation / eau
    veg_mask_path: str | Path | None = None,
    water_mask_path: str | Path | None = None,
    apply_land_mask: bool = True,
    # Lecture/scale/clip
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    # Outputs
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection d'olivine (ou cible similaire) avec :
      - BD1050 = band_depth(860, 1050, 1280)
      - BD2000 = band_depth(1800, 2000, 2300) (contrôle / pyroxènes)
      - masque binaire: (BD1050 > thresh) & (BD2000 < max) -> 255 sinon 0
      - "probabilité" [0..1]: sqrt(score1050 * score2000_inversé)

    Écrit 4 rasters :
      - BD1050_{target_name}.tif (float32)
      - BD2000_control.tif (float32)
      - {target_name}_mask.tif (uint8: 0/255)
      - {target_name}_probability.tif (float32, 0 hors masque si option)

    Returns
    -------
    dict avec chemins, bandes choisies, stats.
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [860, 1050, 1280, 1800, 2000, 2300]

    # --- lire table bandes
    df = pd.read_csv(bands_csv)

    # --- sélectionner band_id + lambda réel
    bands = closest_band_dict(df, targets_nm)

    # unpack (band_id est attendu 1-based pour rasterio)
    b860, lam860 = bands[860.0]
    b1050, lam1050 = bands[1050.0]
    b1280, lam1280 = bands[1280.0]
    b1800, lam1800 = bands[1800.0]
    b2000, lam2000 = bands[2000.0]
    b2300, lam2300 = bands[2300.0]

    # --- util normalisation
    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # --- calcul
    with rasterio.open(tif_path) as src:
        bands_idx = {
            "b860": b860,
            "b1050": b1050,
            "b1280": b1280,
            "b1800": b1800,
            "b2000": b2000,
            "b2300": b2300,
        }

        arrs = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        R860 = arrs["b860"]
        R1050 = arrs["b1050"]
        R1280 = arrs["b1280"]
        R1800 = arrs["b1800"]
        R2000 = arrs["b2000"]
        R2300 = arrs["b2300"]

        bd1050 = band_depth(R860, R1050, R1280, lam860, lam1050, lam1280).astype("float32")
        bd2000 = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype("float32")

        # masque
        mask_bool = (bd1050 > bd1050_thresh) & (bd2000 < bd2000_max)
        mask = (mask_bool.astype("uint8") * 255).astype("uint8")

        # probabilité
        s1050 = normalize01(bd1050, bd1050_score_min, bd1050_score_max)
        s2000_bad = normalize01(bd2000, bd2000_good_min, bd2000_good_max)
        s2000 = 1.0 - s2000_bad

        prob = np.sqrt(s1050 * s2000).astype("float32")
        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        # profils + écriture
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

    # --- stats / logs
    n_pixels = int(np.sum(mask == 255))
    detected = n_pixels > 0

    prob_stats = None
    if n_pixels > 0:
        p = prob[mask == 255]
        prob_stats = {
            "min": float(np.nanmin(p)),
            "mean": float(np.nanmean(p)),
            "max": float(np.nanmax(p)),
        }

    if verbose:
        print("Créés :", str(out_bd1050), str(out_bd2000), str(out_mask), str(out_prob))
        print("Détection:", "OUI" if detected else "NON")
        print("Nombre de pixels à 255 :", n_pixels)
        if prob_stats is not None:
            print("Probability score on detected pixels (min/mean/max):",
                  prob_stats["min"], prob_stats["mean"], prob_stats["max"])
            
    # -------------------------------------------------
    # OPTION : exclusion végétation + eau (land only)
    # -------------------------------------------------
    land_outputs = None

    if apply_land_mask and veg_mask_path is not None and water_mask_path is not None:
        from enmap_land_masking_utils import mask_mineral_land_only

        out_mask_land = outdir / f"{target_name}_mask_land_only.tif"
        out_prob_land = outdir / f"{target_name}_probability_land_only.tif"

        land_res = mask_mineral_land_only(
            mineral_mask_path=out_mask,
            mineral_score_path=out_prob,
            veg_mask_path=veg_mask_path,
            water_mask_path=water_mask_path,
            out_mask_path=out_mask_land,
            out_score_path=out_prob_land,
            score_excluded_value=0.0,
            verbose=verbose
        )

        land_outputs = land_res["outputs"]

        if verbose:
            print("Masques land-only créés :", land_outputs)


    return {
        "outputs": {
            "bd1050": out_bd1050,
            "bd2000": out_bd2000,
            "mask": out_mask,
            "prob": out_prob,
            "land_only": land_outputs,   # <-- NOUVEAU
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
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }

