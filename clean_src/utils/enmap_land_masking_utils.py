from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio


def mask_mineral_land_only(
    mineral_mask_path: str | Path,
    mineral_score_path: str | Path,
    veg_mask_path: str | Path,
    water_mask_path: str | Path,
    out_mask_path: str | Path,
    out_score_path: str | Path,
    *,
    exclude_veg_if_gt: float = 0,
    exclude_water_if_gt: float = 0,
    score_excluded_value: float = 0.0,   # mets np.nan si tu veux du NoData
    compress: str = "lzw",
    verbose: bool = True,
) -> dict:
    """
    Exclut (met à 0 ou NoData) les pixels végétation et/ou eau d'un masque minéral
    et de son score/probabilité.

    - exclude = (veg_mask > exclude_veg_if_gt) OR (water_mask > exclude_water_if_gt)
    - masque: pixels exclus -> 0, sinon conserve (0/255)
    - score : pixels exclus -> score_excluded_value, sinon conserve le score

    Retourne un dict avec chemins + stats.
    """
    mineral_mask_path = Path(mineral_mask_path)
    mineral_score_path = Path(mineral_score_path)
    veg_mask_path = Path(veg_mask_path)
    water_mask_path = Path(water_mask_path)
    out_mask_path = Path(out_mask_path)
    out_score_path = Path(out_score_path)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_score_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- READ
    with rasterio.open(mineral_mask_path) as src:
        mineral_mask = src.read(1)
        profile_mask = src.profile.copy()

    with rasterio.open(mineral_score_path) as src:
        mineral_score = src.read(1).astype(np.float32, copy=False)
        profile_score = src.profile.copy()

    with rasterio.open(veg_mask_path) as src:
        veg_mask = src.read(1)

    with rasterio.open(water_mask_path) as src:
        water_mask = src.read(1)

    # ---- check shapes
    if not (
        mineral_mask.shape == mineral_score.shape
        and mineral_mask.shape == veg_mask.shape
        and mineral_mask.shape == water_mask.shape
    ):
        raise ValueError(
            "Les rasters n'ont pas la même taille (shape). "
            "Il faut les aligner/reprojeter avant."
        )

    # ---- EXCLUDE
    exclude = (veg_mask > exclude_veg_if_gt) | (water_mask > exclude_water_if_gt)

    # ---- MASK output (uint8 0/255)
    mineral_land_only = np.where(
        exclude,
        0,
        (mineral_mask > 0).astype(np.uint8) * 255
    ).astype(np.uint8)

    profile_mask.update(dtype="uint8", count=1, nodata=0, compress=compress)

    with rasterio.open(out_mask_path, "w", **profile_mask) as dst:
        dst.write(mineral_land_only, 1)

    # ---- SCORE output (float32)
    mineral_score_land_only = np.where(exclude, score_excluded_value, mineral_score).astype(np.float32)

    profile_score.update(dtype="float32", count=1, compress=compress)
    # nodata : si np.nan, rasterio gère souvent, mais certains workflows préfèrent un nodata numérique
    if isinstance(score_excluded_value, float) and np.isnan(score_excluded_value):
        profile_score.update(nodata=np.nan)
    else:
        profile_score.update(nodata=float(score_excluded_value))

    with rasterio.open(out_score_path, "w", **profile_score) as dst:
        dst.write(mineral_score_land_only, 1)

    # ---- stats
    valid = ~exclude
    stats = None
    if np.any(valid):
        arr = mineral_score_land_only[valid]
        stats = {
            "min": float(np.nanmin(arr)),
            "mean": float(np.nanmean(arr)),
            "max": float(np.nanmax(arr)),
        }

    if verbose:
        print("✅ Masque final écrit :", str(out_mask_path))
        print("✅ Image score final écrit :", str(out_score_path))
        if stats is not None:
            print("Score (min/mean/max) sur pixels non exclus:",
                  stats["min"], stats["mean"], stats["max"])

    return {
        "outputs": {
            "mask_land_only": out_mask_path,
            "score_land_only": out_score_path,
        },
        "stats_non_excluded": stats,
        "n_excluded": int(np.sum(exclude)),
        "n_total": int(exclude.size),
    }
