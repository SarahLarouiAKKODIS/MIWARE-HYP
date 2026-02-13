from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_carbonates_bd2330_bd2500(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "carbonates",
    targets_nm: list[float] | None = None,
     # --- POST-TRAITEMENT : exclusion végétation / eau
    veg_mask_path: str | Path | None = None,
    water_mask_path: str | Path | None = None,
    apply_land_mask: bool = True,
    # Lecture/scale/clip
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    # Seuils masque (à ajuster)
    bd2330_thresh: float = 0.03,
    bd2500_thresh: float = 0.02,
    # Normalisation score [0..1]
    bd2330_score_min: float = 0.00,
    bd2330_score_max: float = 0.10,
    bd2500_score_min: float = 0.00,
    bd2500_score_max: float = 0.08,
    # Options
    use_bd2500: bool = True,             # ignore bd2500 même si bandes dispo si False
    prob_zero_outside_mask: bool = True, # prob=0 hors détection
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection carbonates via band depth autour de ~2.33 µm (CO3) + optionnel ~2.50 µm.

    BD2330 = band_depth(2200, 2330, 2450)
    Optionnel BD2500 = band_depth(2400, 2500, 2600)

    masque:
      - si BD2500 utilisé/dispo: (BD2330 > bd2330_thresh) & (BD2500 > bd2500_thresh)
      - sinon: (BD2330 > bd2330_thresh)

    probabilité (score [0..1]):
      - sans BD2500: norm(BD2330)
      - avec BD2500: sqrt(norm(BD2330) * norm(BD2500))
      - option: prob = 0 hors masque

    Écrit (si write_outputs=True) :
      - BD2330_carbonates.tif (float32)
      - BD2500_carbonates.tif (float32, si calculé)
      - carbonates_mask.tif (uint8: 0/255)
      - carbonates_probability.tif (float32)

    Returns
    -------
    dict avec outputs, bandes, params, stats.
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        # inclut le triplet 2.33 et le triplet 2.50
        targets_nm = [2200, 2330, 2450, 2400, 2500, 2600]

    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # --- Triplet principal ~2330
    b2200, lam2200 = band_map[2200.0] if 2200.0 in band_map else band_map[2200]
    b2330, lam2330 = band_map[2330.0] if 2330.0 in band_map else band_map[2330]
    b2450, lam2450 = band_map[2450.0] if 2450.0 in band_map else band_map[2450]

    # --- Triplet optionnel ~2500
    has_bd2500 = False
    if use_bd2500:
        try:
            b2400, lam2400 = band_map[2400.0] if 2400.0 in band_map else band_map[2400]
            b2500, lam2500 = band_map[2500.0] if 2500.0 in band_map else band_map[2500]
            b2600, lam2600 = band_map[2600.0] if 2600.0 in band_map else band_map[2600]
            has_bd2500 = True
        except KeyError:
            has_bd2500 = False

    # outputs
    out_bd2330 = outdir / f"BD2330_{target_name}.tif"
    out_bd2500 = outdir / f"BD2500_{target_name}.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        bands_idx = {"b2200": b2200, "b2330": b2330, "b2450": b2450}
        if has_bd2500:
            bands_idx.update({"b2400": b2400, "b2500": b2500, "b2600": b2600})

        arrs = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        R2200, R2330, R2450 = arrs["b2200"], arrs["b2330"], arrs["b2450"]
        bd2330 = band_depth(R2200, R2330, R2450, lam2200, lam2330, lam2450).astype("float32")

        if has_bd2500:
            R2400, R2500, R2600 = arrs["b2400"], arrs["b2500"], arrs["b2600"]
            bd2500 = band_depth(R2400, R2500, R2600, lam2400, lam2500, lam2600).astype("float32")
        else:
            bd2500 = None

        # masque
        if has_bd2500:
            mask_bool = (bd2330 > bd2330_thresh) & (bd2500 > bd2500_thresh)
        else:
            mask_bool = (bd2330 > bd2330_thresh)

        mask = (mask_bool.astype("uint8") * 255).astype("uint8")

        # prob
        s2330 = normalize01(bd2330, bd2330_score_min, bd2330_score_max)
        if has_bd2500:
            s2500 = normalize01(bd2500, bd2500_score_min, bd2500_score_max)
            prob = np.sqrt(s2330 * s2500).astype("float32")
        else:
            prob = s2330.astype("float32")

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        # profils
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        if write_outputs:
            with rasterio.open(out_bd2330, "w", **prof_f) as dst:
                dst.write(bd2330, 1)

            if has_bd2500:
                with rasterio.open(out_bd2500, "w", **prof_f) as dst:
                    dst.write(bd2500, 1)

            with rasterio.open(out_mask, "w", **prof_u8) as dst:
                dst.write(mask, 1)

            with rasterio.open(out_prob, "w", **prof_f) as dst:
                dst.write(prob, 1)

    # stats
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
        print("✅ Créés :", str(out_bd2330),
              (str(out_bd2500) if has_bd2500 else "(BD2500 non calculé)"),
              str(out_mask), str(out_prob))
        print("Détection carbonates :", "✅ OUI" if detected else "❌ NON")
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
