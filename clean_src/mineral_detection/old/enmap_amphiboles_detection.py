from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_amphiboles_bd2320(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "amphiboles",
    targets_nm: list[float] | None = None,
    # --- POST-TRAITEMENT : exclusion végétation / eau
    veg_mask_path: str | Path | None = None,
    water_mask_path: str | Path | None = None,
    apply_land_mask: bool = True,
    # Lecture/scale/clip
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    # Seuils (à ajuster)
    bd2320_thresh: float = 0.03,
    bd2000_thresh: float = 0.02,
    # Normalisation score [0..1]
    bd2320_score_min: float = 0.00,
    bd2320_score_max: float = 0.10,
    bd2000_score_min: float = 0.00,
    bd2000_score_max: float = 0.08,
    # Options
    use_bd2000_control: bool = True,      # si False, ignore le contrôle même si bandes dispo
    prob_zero_outside_mask: bool = True,  # prob=0 hors pixels détectés
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection amphiboles basée sur une absorption ~2.32 µm (Mg/Fe-OH) via band depth.

    BD2320 = band_depth(2250, 2320, 2390)  (diagnostic amphiboles)
    Optionnel contrôle : BD2000 = band_depth(1900, 2000, 2100)
      - masque: BD2320 > bd2320_thresh  ET (si contrôle activé/dispo) BD2000 > bd2000_thresh
      - prob:   si contrôle -> sqrt(norm(BD2320) * norm(BD2000)) sinon -> norm(BD2320)

    Écrit (si write_outputs=True) :
      - BD2320_amphiboles.tif (float32)
      - BD2000_control.tif (float32, si dispo/activé)
      - amphiboles_mask.tif (uint8: 0/255)
      - amphiboles_probability.tif (float32)

    Returns
    -------
    dict avec outputs, bandes, params, stats.
    """
    tif_path = Path(tif_path)
    bands_csv = Path(bands_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if targets_nm is None:
        targets_nm = [1900, 2000, 2100, 2250, 2320, 2390]

    df = pd.read_csv(bands_csv)
    band_map = closest_band_dict(df, targets_nm)

    # Triplet diagnostic ~2320
    b2250, lam2250 = band_map[2250.0] if 2250.0 in band_map else band_map[2250]
    b2320, lam2320 = band_map[2320.0] if 2320.0 in band_map else band_map[2320]
    b2390, lam2390 = band_map[2390.0] if 2390.0 in band_map else band_map[2390]

    # Contrôle ~2000 (optionnel)
    has_bd2000 = False
    if use_bd2000_control:
        try:
            b1900, lam1900 = band_map[1900.0] if 1900.0 in band_map else band_map[1900]
            b2000, lam2000 = band_map[2000.0] if 2000.0 in band_map else band_map[2000]
            b2100, lam2100 = band_map[2100.0] if 2100.0 in band_map else band_map[2100]
            has_bd2000 = True
        except KeyError:
            has_bd2000 = False

    def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # outputs paths
    out_bd2320 = outdir / f"BD2320_{target_name}.tif"
    out_bd2000 = outdir / "BD2000_control.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    # calcul
    with rasterio.open(tif_path) as src:
        bands_idx = {"b2250": b2250, "b2320": b2320, "b2390": b2390}
        if has_bd2000:
            bands_idx.update({"b1900": b1900, "b2000": b2000, "b2100": b2100})

        arrs = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        R2250, R2320, R2390 = arrs["b2250"], arrs["b2320"], arrs["b2390"]
        bd2320 = band_depth(R2250, R2320, R2390, lam2250, lam2320, lam2390).astype("float32")

        if has_bd2000:
            R1900, R2000, R2100 = arrs["b1900"], arrs["b2000"], arrs["b2100"]
            bd2000 = band_depth(R1900, R2000, R2100, lam1900, lam2000, lam2100).astype("float32")
        else:
            bd2000 = None

        # masque
        if has_bd2000:
            mask_bool = (bd2320 > bd2320_thresh) & (bd2000 > bd2000_thresh)
        else:
            mask_bool = (bd2320 > bd2320_thresh)

        mask = (mask_bool.astype("uint8") * 255).astype("uint8")

        # prob
        s2320 = normalize01(bd2320, bd2320_score_min, bd2320_score_max)
        if has_bd2000:
            s2000 = normalize01(bd2000, bd2000_score_min, bd2000_score_max)
            prob = np.sqrt(s2320 * s2000).astype("float32")
        else:
            prob = s2320.astype("float32")

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        # profils d’écriture
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        if write_outputs:
            with rasterio.open(out_bd2320, "w", **prof_f) as dst:
                dst.write(bd2320, 1)

            if has_bd2000:
                with rasterio.open(out_bd2000, "w", **prof_f) as dst:
                    dst.write(bd2000, 1)

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
        print("Créés :", str(out_bd2320),
              (str(out_bd2000) if has_bd2000 else "(BD2000 non calculé)"),
              str(out_mask), str(out_prob))
        print("Détection amphiboles :", "OUI" if detected else "❌ NON")
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
            "bd2320": out_bd2320,
            "bd2000": out_bd2000 if has_bd2000 else None,
            "mask": out_mask,
            "prob": out_prob,
        },
        "bands_1based_and_lambda_nm": {
            2250.0: (int(b2250), float(lam2250)),
            2320.0: (int(b2320), float(lam2320)),
            2390.0: (int(b2390), float(lam2390)),
            **({
                1900.0: (int(b1900), float(lam1900)),
                2000.0: (int(b2000), float(lam2000)),
                2100.0: (int(b2100), float(lam2100)),
            } if has_bd2000 else {})
        },
        "params": {
            "bd2320_thresh": float(bd2320_thresh),
            "bd2000_thresh": float(bd2000_thresh),
            "bd2320_score_min": float(bd2320_score_min),
            "bd2320_score_max": float(bd2320_score_max),
            "bd2000_score_min": float(bd2000_score_min),
            "bd2000_score_max": float(bd2000_score_max),
            "use_bd2000_control": bool(use_bd2000_control),
            "has_bd2000": bool(has_bd2000),
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }
