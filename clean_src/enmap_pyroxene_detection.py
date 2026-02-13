from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_pyroxene_bd1um_bd2um(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "pyroxene",
    targets_nm: list[float] | None = None,
    # Seuils masque (à ajuster)
    bd1um_thresh: float = 0.05,
    bd2um_thresh: float = 0.03,
    # Normalisation score [0..1] (à ajuster)
    bd1um_score_min: float = 0.00,
    bd1um_score_max: float = 0.15,
    bd2um_score_min: float = 0.00,
    bd2um_score_max: float = 0.12,
     # --- POST-TRAITEMENT : exclusion végétation / eau
    veg_mask_path: str | Path | None = None,
    water_mask_path: str | Path | None = None,
    apply_land_mask: bool = True,
    # Lecture/scale/clip
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    # Écriture
    compress: str = "lzw",
    prob_zero_outside_mask: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Détection pyroxène via band depth autour de ~1µm et ~2µm :

      BD1um  = band_depth(900, 1000, 1200)
      BD2um  = band_depth(1800, 2000, 2300)
      masque = (BD1um > bd1um_thresh) & (BD2um > bd2um_thresh) -> 255 sinon 0
      prob   = sqrt(norm(BD1um) * norm(BD2um)) ; option: prob=0 hors masque

    Écrit 4 rasters :
      - BD1um_pyroxene.tif (float32)
      - BD2um_pyroxene.tif (float32)
      - pyroxene_mask.tif (uint8: 0/255)
      - pyroxene_probability.tif (float32)

    Returns
    -------
    dict avec:
      - outputs (chemins)
      - bands_1based_and_lambda_nm
      - params
      - stats
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [900, 1000, 1200, 1800, 2000, 2300]

    # --- lire table bandes + sélectionner band_id + lambda réel
    df = pd.read_csv(bands_csv)
    bands = closest_band_dict(df, targets_nm)

    b900, lam900 = bands[900.0]
    b1000, lam1000 = bands[1000.0]
    b1200, lam1200 = bands[1200.0]
    b1800, lam1800 = bands[1800.0]
    b2000, lam2000 = bands[2000.0]
    b2300, lam2300 = bands[2300.0]

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # --- outputs
    out_bd1um = outdir / f"BD1um_{target_name}.tif"
    out_bd2um = outdir / f"BD2um_{target_name}.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    # --- calcul
    with rasterio.open(tif_path) as src:
        bands_idx = {
            "b900": b900,
            "b1000": b1000,
            "b1200": b1200,
            "b1800": b1800,
            "b2000": b2000,
            "b2300": b2300,
        }

        arrs = read_scale_and_clip_bands(
            src, bands=bands_idx, scale=scale, min_val=min_val, max_val=max_val, verbose=verbose
        )

        R900 = arrs["b900"]
        R1000 = arrs["b1000"]
        R1200 = arrs["b1200"]
        R1800 = arrs["b1800"]
        R2000 = arrs["b2000"]
        R2300 = arrs["b2300"]

        bd1um = band_depth(R900, R1000, R1200, lam900, lam1000, lam1200).astype("float32")
        bd2um = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype("float32")

        mask_bool = (bd1um > bd1um_thresh) & (bd2um > bd2um_thresh)
        mask = (mask_bool.astype("uint8") * 255).astype("uint8")

        s1 = normalize01(bd1um, bd1um_score_min, bd1um_score_max)
        s2 = normalize01(bd2um, bd2um_score_min, bd2um_score_max)
        prob = np.sqrt(s1 * s2).astype("float32")
        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        with rasterio.open(out_bd1um, "w", **prof_f) as dst:
            dst.write(bd1um, 1)
        with rasterio.open(out_bd2um, "w", **prof_f) as dst:
            dst.write(bd2um, 1)
        with rasterio.open(out_mask, "w", **prof_u8) as dst:
            dst.write(mask, 1)
        with rasterio.open(out_prob, "w", **prof_f) as dst:
            dst.write(prob, 1)

    # --- stats
    n_pixels = int(np.sum(mask == 255))
    detected = n_pixels > 0

    prob_stats = None
    if detected:
        p = prob[mask == 255]
        prob_stats = {
            "min": float(np.nanmin(p)),
            "mean": float(np.nanmean(p)),
            "max": float(np.nanmax(p)),
        }

    if verbose:
        print("Créés :", str(out_bd1um), str(out_bd2um), str(out_mask), str(out_prob))
        print("Détection pyroxène :", "OUI" if detected else "❌ NON")
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
            "bd1um": out_bd1um,
            "bd2um": out_bd2um,
            "mask": out_mask,
            "prob": out_prob,
        },
        "bands_1based_and_lambda_nm": {
            900.0: (int(b900), float(lam900)),
            1000.0: (int(b1000), float(lam1000)),
            1200.0: (int(b1200), float(lam1200)),
            1800.0: (int(b1800), float(lam1800)),
            2000.0: (int(b2000), float(lam2000)),
            2300.0: (int(b2300), float(lam2300)),
        },
        "params": {
            "bd1um_thresh": float(bd1um_thresh),
            "bd2um_thresh": float(bd2um_thresh),
            "bd1um_score_min": float(bd1um_score_min),
            "bd1um_score_max": float(bd1um_score_max),
            "bd2um_score_min": float(bd2um_score_min),
            "bd2um_score_max": float(bd2um_score_max),
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }
