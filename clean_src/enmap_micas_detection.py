from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth
from enmap_metadata_utils import closest_band_dict


def detect_micas_bd2200(
    tif_path: str | Path,
    bands_csv: str | Path,
    outdir: str | Path,
    *,
    target_name: str = "micas",
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
    write_outputs: bool = True,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Détection micas via band depth ~2200 nm (Al-OH) + optionnel contrôle ~1900 nm (H2O/OH).

    BD2200 = band_depth(2100, 2200, 2300)
    Optionnel BD1900 = band_depth(1800, 1900, 2000)

    masque:
      - si BD1900 activé/dispo: (BD2200 > bd2200_thresh) & (BD1900 > bd1900_thresh)
      - sinon: (BD2200 > bd2200_thresh)

    probabilité (score [0..1]):
      - sans BD1900: norm(BD2200)
      - avec BD1900: sqrt(norm(BD2200) * norm(BD1900))
      - option: prob=0 hors masque

    Écrit (si write_outputs=True) :
      - BD2200_micas.tif (float32)
      - BD1900_control.tif (float32, si calculé)
      - micas_mask.tif (uint8: 0/255)
      - micas_probability.tif (float32)

    Returns
    -------
    dict avec outputs, bandes, params, stats.
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
        x = x.astype("float32", copy=False)
        y = (x - vmin) / (vmax - vmin + 1e-12)
        return np.clip(y, 0.0, 1.0).astype("float32")

    # Triplet ~2200 (diagnostic)
    b2100, lam2100 = band_map[2100.0] if 2100.0 in band_map else band_map[2100]
    b2200, lam2200 = band_map[2200.0] if 2200.0 in band_map else band_map[2200]
    b2300, lam2300 = band_map[2300.0] if 2300.0 in band_map else band_map[2300]

    # Triplet ~1900 (optionnel)
    has_bd1900 = False
    if use_bd1900_control:
        try:
            b1800, lam1800 = band_map[1800.0] if 1800.0 in band_map else band_map[1800]
            b1900, lam1900 = band_map[1900.0] if 1900.0 in band_map else band_map[1900]
            b2000, lam2000 = band_map[2000.0] if 2000.0 in band_map else band_map[2000]
            has_bd1900 = True
        except KeyError:
            has_bd1900 = False

    # outputs
    out_bd2200 = outdir / f"BD2200_{target_name}.tif"
    out_bd1900 = outdir / "BD1900_control.tif"
    out_mask = outdir / f"{target_name}_mask.tif"
    out_prob = outdir / f"{target_name}_probability.tif"

    with rasterio.open(tif_path) as src:
        bands_idx = {"b2100": b2100, "b2200": b2200, "b2300": b2300}
        if has_bd1900:
            bands_idx.update({"b1800": b1800, "b1900": b1900, "b2000": b2000})

        arrs = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        R2100, R2200, R2300 = arrs["b2100"], arrs["b2200"], arrs["b2300"]
        bd2200 = band_depth(R2100, R2200, R2300, lam2100, lam2200, lam2300).astype("float32")

        if has_bd1900:
            R1800, R1900, R2000 = arrs["b1800"], arrs["b1900"], arrs["b2000"]
            bd1900 = band_depth(R1800, R1900, R2000, lam1800, lam1900, lam2000).astype("float32")
        else:
            bd1900 = None

        # masque
        if has_bd1900:
            mask_bool = (bd2200 > bd2200_thresh) & (bd1900 < bd1900_thresh)
        else:
            mask_bool = (bd2200 > bd2200_thresh)

        mask = (mask_bool.astype("uint8") * 255).astype("uint8")

        # prob
        s2200 = normalize01(bd2200, bd2200_score_min, bd2200_score_max)
        if has_bd1900:
            s1900 = normalize01(bd1900, bd1900_score_min, bd1900_score_max)
            prob = np.sqrt(s2200 * s1900).astype("float32")
        else:
            prob = s2200.astype("float32")

        if prob_zero_outside_mask:
            prob = np.where(mask_bool, prob, 0.0).astype("float32")

        # stats prob min/max sur pixels détectés
        prob_minmax = None
        if np.any(mask_bool):
            p = prob[mask_bool]
            prob_minmax = {"min": float(np.nanmin(p)), "max": float(np.nanmax(p))}
            if verbose:
                print("Probabilité min :", prob_minmax["min"])
                print("Probabilité max :", prob_minmax["max"])

        # écriture
        prof_f = src.profile.copy()
        prof_f.update(count=1, dtype="float32", nodata=np.nan, compress=compress)

        prof_u8 = src.profile.copy()
        prof_u8.update(count=1, dtype="uint8", nodata=0, compress=compress)

        if write_outputs:
            with rasterio.open(out_bd2200, "w", **prof_f) as dst:
                dst.write(bd2200, 1)

            if has_bd1900:
                with rasterio.open(out_bd1900, "w", **prof_f) as dst:
                    dst.write(bd1900, 1)

            with rasterio.open(out_mask, "w", **prof_u8) as dst:
                dst.write(mask, 1)

            with rasterio.open(out_prob, "w", **prof_f) as dst:
                dst.write(prob, 1)

    # stats globales
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
        print("Créés :", str(out_bd2200),
              (str(out_bd1900) if has_bd1900 else "(BD1900 non calculé)"),
              str(out_mask), str(out_prob))
        print("Détection micas :", "OUI" if detected else "NON")
        print("Nombre de pixels à 255 :", n_pixels)
        if prob_stats is not None:
            print("Probabilité (score) sur pixels détectés -> min/mean/max:",
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
            "bd2200": out_bd2200,
            "bd1900": out_bd1900 if has_bd1900 else None,
            "mask": out_mask,
            "prob": out_prob,
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
        },
        "stats": {
            "n_pixels_255": n_pixels,
            "detected": detected,
            "prob_on_detected": prob_stats,
        },
    }
