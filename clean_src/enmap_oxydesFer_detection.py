from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_iron_oxides_bd900_redness(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "iron_oxides",
    targets_nm: list[float] | None = None,
     # --- POST-TRAITEMENT : exclusion végétation / eau
    veg_mask_path: str | Path | None = None,
    water_mask_path: str | Path | None = None,
    apply_land_mask: bool = True,
    # Lecture/scale/clip
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    # Seuils masque
    bd900_thresh: float = 0.04,
    redness_thresh: float = 0.05,
    # Normalisation score [0..1]
    bd900_score_min: float = 0.00,
    bd900_score_max: float = 0.12,
    red_score_min: float = 0.00,
    red_score_max: float = 0.20,
    # Masquage "pixels sombres" pour éviter un ratio instable
    dark_den_thresh: float = 1e-3,        # si réflectance déjà en [0..1]
    dark_den_thresh_raw: float = 100.0,   # si réflectance est plutôt en [0..10000] (au cas où)
    assume_reflectance_01: bool = True,   # True -> compare à dark_den_thresh ; False -> dark_den_thresh_raw
    # Options
    prob_zero_outside_mask: bool = True,
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection "iron oxides" avec :
      - BD900  = band_depth(860, 900, 940)  (absorption ferrique ~0.9 µm)
      - REDNESS = (R650 - R550)/(R650 + R550) (rougeur)
      - masque = (BD900 > bd900_thresh) & (REDNESS > redness_thresh) -> 255 sinon 0
      - prob   = sqrt(norm(BD900) * norm(REDNESS)) ; option: prob=0 hors masque

    Écrit :
      - BD900_iron_oxides.tif (float32)
      - REDNESS_iron_oxides.tif (float32)
      - iron_oxides_mask.tif (uint8 0/255)
      - iron_oxides_probability.tif (float32 0..1)

    Returns
    -------
    dict avec outputs, bandes, params, stats, + arrays.
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [550, 650, 860, 900, 940]

    # --- helpers
    def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = a + b
        out = np.full(a.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (denom != 0)
        out[valid] = ((a[valid] - b[valid]) / denom[valid]).astype(np.float32)
        return out

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # --- bande_id + lambda réels
    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    b860, lam860 = band_map[860.0] if 860.0 in band_map else band_map[860]
    b900, lam900 = band_map[900.0] if 900.0 in band_map else band_map[900]
    b940, lam940 = band_map[940.0] if 940.0 in band_map else band_map[940]

    b650, lam650 = band_map[650.0] if 650.0 in band_map else band_map[650]
    b550, lam550 = band_map[550.0] if 550.0 in band_map else band_map[550]

    # outputs
    out_bd900 = outdir / "BD900_iron_oxides.tif"
    out_redness = outdir / "REDNESS_iron_oxides.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        bands_idx = {"b860": b860, "b900": b900, "b940": b940, "b650": b650, "b550": b550}

        arrs = read_scale_and_clip_bands(
            src, bands=bands_idx, scale=scale, min_val=min_val, max_val=max_val, verbose=verbose
        )

        R860 = arrs["b860"]
        R900 = arrs["b900"]
        R940 = arrs["b940"]
        R650 = arrs["b650"]
        R550 = arrs["b550"]

        # 1) BD900
        bd900 = band_depth(R860, R900, R940, lam860, lam900, lam940).astype("float32")

        # 2) Redness
        redness = safe_norm_diff(R650, R550).astype("float32")
        redness = np.clip(redness, -1.0, 1.0)

        den = R650 + R550
        if assume_reflectance_01:
            dark = ~np.isfinite(den) | (den <= float(dark_den_thresh))
        else:
            dark = ~np.isfinite(den) | (den <= float(dark_den_thresh_raw))
        redness[dark] = np.nan

        if verbose:
            try:
                print("redness min/max:", float(np.nanmin(redness)), float(np.nanmax(redness)))
                print("percentiles:", np.nanpercentile(redness, [1, 5, 50, 95, 99]))
            except Exception:
                pass
            print("redness_thresh:", redness_thresh)
            print("bd900_thresh:", bd900_thresh)

        # masque
        mask_bool = (bd900 > bd900_thresh) & (redness > redness_thresh)

        if verbose:
            n_det = int(mask_bool.sum())
            n_fin = int(np.isfinite(bd900).sum())
            print("pixels détectés:", n_det, "sur", n_fin)
            if n_fin > 0:
                print("ratio (%):", 100.0 * float(n_det) / float(n_fin))

        mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        # prob
        s_bd900 = normalize01(bd900, bd900_score_min, bd900_score_max)
        s_red = normalize01(redness, red_score_min, red_score_max)
        prob = np.sqrt(s_bd900 * s_red).astype("float32")

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        # profils
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        if write_outputs:
            with rasterio.open(out_bd900, "w", **prof_f) as dst:
                dst.write(bd900, 1)

            with rasterio.open(out_redness, "w", **prof_f) as dst:
                dst.write(redness, 1)

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
        print("✅ Créés :", str(out_bd900), str(out_redness), str(out_mask), str(out_prob))
        print("Détection oxydes de fer :", "✅ OUI" if detected else "❌ NON")
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
            "bd900": out_bd900,
            "redness": out_redness,
            "mask": out_mask,
            "prob": out_prob,
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
