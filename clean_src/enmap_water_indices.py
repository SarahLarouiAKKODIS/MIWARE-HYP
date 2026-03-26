from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
import tifffile

from utils.enmap_indices_calculation_utils import (
    load_wavelengths_from_csv,
    compute_auto_tol_nm,
    nearest_band_index,
    safe_norm_diff,
    write_imagej_tiff,
    read_scale_and_clip_bands,
)


def compute_mndwi_and_water_mask(
    tif_path: str | Path,
    wavelengths_csv: str | Path,
    outdir: str | Path,
    prefix: str = "enmap",
    *,
    green_nm: float = 560.0,
    swir1_nm: float = 1610.0,
    mndwi_th: float = 0.55,
    auto_convert_um_to_nm: bool = True,
    use_auto_tol: bool = True,
    tol_nm_fixed: float = 12.0,
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    nodata_f32: float = -9999.0,
    verbose: bool = True,
) -> dict:
    """
    Calcule MNDWI (Xu) et un masque eau à partir d'un cube EnMAP (GeoTIFF).

    Sorties écrites dans `outdir` :
      - {prefix}_MNDWI_Xu.tiff (float32)
      - {prefix}_WATER_MASK.tiff (int16, valeurs: -1 nodata / 0 non-eau / 255 eau)
      - {prefix}_WATER_MASK_VISUAL.tiff (RGB: rouge=exclu, bleu=eau, gris=non-eau)

    Returns
    -------
    dict
        {
          "mndwi": np.ndarray float32 (H,W),
          "water_out": np.ndarray int16 (H,W),
          "tol_nm": float,
          "bands_1based": {"GREEN": int, "SWIR1": int},
          "wavelengths_nm": np.ndarray,
          "paths": {"mndwi": Path, "mask": Path, "visual": Path}
        }
    """
    tif_path = Path(tif_path)
    wavelengths_csv = Path(wavelengths_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Charger longueurs d'onde ---
    wv = load_wavelengths_from_csv(wavelengths_csv)
    if auto_convert_um_to_nm and np.nanmax(wv) < 50:
        wv = wv * 1000.0  # µm -> nm

    tol_nm = compute_auto_tol_nm(wv) if use_auto_tol else tol_nm_fixed

    # --- Lire le TIF + bandes utiles ---
    with rasterio.open(tif_path) as src:
        nb = src.count
        if len(wv) != nb:
            raise ValueError(f"len(wavelengths)={len(wv)} != nb_bandes_tif={nb}")

        # Match bandes (0-based) -> rasterio (1-based)
        b_green = nearest_band_index(wv, green_nm, tol_nm) + 1
        b_swir1 = nearest_band_index(wv, swir1_nm, tol_nm) + 1

        bands_idx = {"GREEN": b_green, "SWIR1": b_swir1}

        bands = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        green = bands["GREEN"]
        swir1 = bands["SWIR1"]

        # Invalid pixels
        nd = src.nodata
        invalid = (~np.isfinite(green) | ~np.isfinite(swir1))
        if nd is not None:
            nd = float(nd)
            invalid |= (green == nd) | (swir1 == nd)

    if verbose:
        print("GREEN:", np.nanmin(green), np.nanmax(green))
        print("SWIR1:", np.nanmin(swir1), np.nanmax(swir1))

    # --- MNDWI (Xu) ---
    mndwi = safe_norm_diff(green, swir1, invalid)
    mndwi = mndwi.astype(np.float32, copy=False)
    mndwi[mndwi == nodata_f32] = np.nan

    # --- Masque eau (0/255/-1) ---
    water = np.full(mndwi.shape, np.nan, dtype=np.float32)
    water[np.isfinite(mndwi) & (mndwi <= mndwi_th)] = 0
    water[np.isfinite(mndwi) & (mndwi >  mndwi_th)] = 1

    water_out = np.full(water.shape, -1, dtype=np.int16)  # -1 = NoData
    water_out[water == 0] = 0
    water_out[water == 1] = 255

    # --- Écriture ---
    p_mndwi = outdir / f"{prefix}_MNDWI_Xu.tiff"
    p_mask = outdir / f"{prefix}_WATER_MASK.tiff"
    p_vis  = outdir / f"{prefix}_WATER_MASK_VISUAL.tiff"

    write_imagej_tiff(str(p_mndwi), mndwi, dtype="float32")
    write_imagej_tiff(str(p_mask), water_out, dtype="int16", nodata=-1)

    # Visual RGB (rouge=exclu, bleu=eau, gris=non-eau)
    rgb = np.zeros((*water_out.shape, 3), dtype=np.uint8)
    rgb[water_out == -1] = [255, 0, 0]      # exclu = rouge
    rgb[water_out == 255] = [0, 0, 255]     # eau = bleu
    rgb[water_out == 0]   = [160, 160, 160] # non-eau = gris

    tifffile.imwrite(str(p_vis), rgb, photometric="rgb", imagej=True)

    if verbose:
        print("OK.")
        print(f"Tolérance utilisée: {tol_nm:.1f} nm")
        print("Bandes utilisées (cible -> bande 1-based -> nm réel) :")
        print(f"  GREEN {green_nm} -> band {b_green} -> {float(wv[b_green-1]):.1f} nm")
        print(f"  SWIR1 {swir1_nm} -> band {b_swir1} -> {float(wv[b_swir1-1]):.1f} nm")
        print(f"Seuil eau MNDWI_TH = {mndwi_th}")

    return {
        "mndwi": mndwi,
        "water_out": water_out,
        "tol_nm": float(tol_nm),
        "bands_1based": {"GREEN": int(b_green), "SWIR1": int(b_swir1)},
        "wavelengths_nm": wv,
        "paths": {"mndwi": p_mndwi, "mask": p_mask, "visual": p_vis},
    }
