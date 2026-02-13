from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import rasterio
import tifffile

from enmap_indices_calculation_utils import (
    load_wavelengths_from_csv,
    compute_auto_tol_nm,
    nearest_band_index,
    safe_norm_diff,
    write_imagej_tiff,
    msavi,
    read_scale_and_clip_bands,
    band_depth,
)


def compute_vegetation_indices_wdi_vii(
    tif_path: str | Path,
    wavelengths_csv: str | Path,
    outdir: str | Path,
    prefix: str = "enmap",
    *,
    # Options longueurs d’onde / tolérance
    auto_convert_um_to_nm: bool = True,
    use_auto_tol: bool = True,
    tol_nm_fixed: float = 12.0,
    # Indices "classiques"
    red_nm: float = 665.0,
    green_nm: float = 560.0,
    rededge_nm: float = 705.0,
    nir_nm: float = 865.0,
    swir1_nm: float = 1610.0,
    # VII Zhang 2012 (plages)
    vii_green_lo: float = 497.0,
    vii_green_hi: float = 635.0,
    vii_nir_lo: float = 700.0,
    vii_nir_hi: float = 1200.0,
    # WDI Zhang 2012 (band depth water valleys)
    w1_center: float = 968.0,
    w1_left: float = 940.0,
    w1_right: float = 990.0,
    w2_center: float = 1181.0,
    w2_left: float = 1140.0,
    w2_right: float = 1240.0,
    # Masque végétation
    ndvi_th: float = 0.3,
    # Lecture / mise à l’échelle
    scale: float = 10000.0,
    min_val: float = 0.0,
    max_val: float = 1.2,
    nodata_f32: float = -9999.0,
    # Calcul de k (WDI) : critères “qualité”
    ndvi_k_th: float = 0.45,
    lwai_968_th: float = 0.02,
    lwai_1181_th: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    Calcule indices végétation + masque végétation + VII (Zhang 2012) + LWAI + WDI (Zhang 2012),
    et écrit les outputs (ImageJ TIFF + visuels RGB).

    Outputs écrits dans outdir :
      - NDVI, GNDVI, NDRE, NDWI_Gao, MSAVI
      - VII_Zhang2012, VII_Zhang2012_veg
      - LWAI_968, LWAI_1181, WDI_article
      - VEG_MASK (int16: -1/0/255) + VEG_MASK_VISUAL (RGB)
      - NDWI_veg, MSAVI_veg, WDI_veg, GNDVI_veg

    Returns
    -------
    dict : contient les arrays principaux, bandes utilisées, tolérance et chemins de sortie.
    """

    # -------------------------
    # Helpers internes
    # -------------------------
    def band_indices_in_range(wv_nm: np.ndarray, lo: float, hi: float) -> np.ndarray:
        idx = np.where((wv_nm >= lo) & (wv_nm <= hi))[0]
        if idx.size == 0:
            raise ValueError(f"Aucune bande trouvée dans [{lo}, {hi}] nm")
        return idx

    def scale_clip_to_reflectance(arr: np.ndarray, scale_: float, min_: float, max_: float) -> np.ndarray:
        out = arr.astype(np.float32) / float(scale_)
        out = np.where(np.isfinite(out), out, np.nan).astype(np.float32)
        out = np.clip(out, min_, max_).astype(np.float32)
        return out

    def compute_vii_zhang2012_blockwise(
        src: rasterio.io.DatasetReader,
        wv_nm: np.ndarray,
        scale_: float,
        min_: float,
        max_: float,
        green_lo: float,
        green_hi: float,
        nir_lo: float,
        nir_hi: float,
    ) -> tuple[np.ndarray, int, int]:
        """
        VII (Zhang et al. 2012) :
          a = somme reflectance sur [green_lo, green_hi]
          b = somme reflectance sur [nir_lo, nir_hi]
          Na = a / mean(a)
          Nb = b / mean(b)
          VII = (Na - Nb)/(Na + Nb)*100
        """
        idx_green = band_indices_in_range(wv_nm, green_lo, green_hi)
        idx_nir = band_indices_in_range(wv_nm, nir_lo, nir_hi)

        bands_green = (idx_green + 1).tolist()  # rasterio 1-based
        bands_nir = (idx_nir + 1).tolist()

        h, w = src.height, src.width
        a = np.full((h, w), np.nan, dtype=np.float32)
        b = np.full((h, w), np.nan, dtype=np.float32)

        nd = src.nodata
        nd = float(nd) if nd is not None else None

        for (_, window) in src.block_windows(1):
            g_raw = src.read(indexes=bands_green, window=window)  # (nb, wh, ww)
            n_raw = src.read(indexes=bands_nir, window=window)

            g = scale_clip_to_reflectance(g_raw, scale_, min_, max_)
            n = scale_clip_to_reflectance(n_raw, scale_, min_, max_)

            if nd is not None:
                g_mask = np.any(g_raw == nd, axis=0)
                n_mask = np.any(n_raw == nd, axis=0)
                bad = g_mask | n_mask
            else:
                bad = np.zeros((g.shape[1], g.shape[2]), dtype=bool)

            a_win = np.nansum(g, axis=0).astype(np.float32)
            b_win = np.nansum(n, axis=0).astype(np.float32)

            a_win = np.where(bad, np.nan, a_win).astype(np.float32)
            b_win = np.where(bad, np.nan, b_win).astype(np.float32)

            r0, c0 = window.row_off, window.col_off
            a[r0:r0 + window.height, c0:c0 + window.width] = a_win
            b[r0:r0 + window.height, c0:c0 + window.width] = b_win

        a_mean = np.nanmean(a)
        b_mean = np.nanmean(b)

        Na = a / a_mean
        Nb = b / b_mean

        denom = (Na + Nb)
        vii = (Na - Nb) / denom * 100.0
        vii = np.where(np.isfinite(vii) & np.isfinite(denom) & (denom != 0), vii, np.nan).astype(np.float32)

        return vii, len(bands_green), len(bands_nir)

    def mask_to_veg(arr: np.ndarray, veg_mask_float: np.ndarray) -> np.ndarray:
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        out[veg_mask_float == 1] = arr[veg_mask_float == 1]
        out[veg_mask_float == 0] = 0.0
        return out

    # -------------------------
    # Préparation
    # -------------------------
    tif_path = Path(tif_path)
    wavelengths_csv = Path(wavelengths_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wv = load_wavelengths_from_csv(wavelengths_csv)
    if auto_convert_um_to_nm and np.nanmax(wv) < 50:
        wv = wv * 1000.0

    tol_nm = compute_auto_tol_nm(wv) if use_auto_tol else tol_nm_fixed

    # -------------------------
    # Lecture & calculs
    # -------------------------
    with rasterio.open(tif_path) as src:
        nb = src.count
        if len(wv) != nb:
            raise ValueError(f"len(wavelengths)={len(wv)} != nb_bandes_tif={nb}")

        # VII Zhang 2012 (plages hyperspectrales)
        vii_zhang, n_green_bands, n_nir_bands = compute_vii_zhang2012_blockwise(
            src=src,
            wv_nm=wv,
            scale_=scale,
            min_=min_val,
            max_=max_val,
            green_lo=vii_green_lo,
            green_hi=vii_green_hi,
            nir_lo=vii_nir_lo,
            nir_hi=vii_nir_hi,
        )

        # Indices "classiques" + WDI : match bandes (0-based -> 1-based)
        b_red = nearest_band_index(wv, red_nm, tol_nm) + 1
        b_green = nearest_band_index(wv, green_nm, tol_nm) + 1
        b_nir = nearest_band_index(wv, nir_nm, tol_nm) + 1
        b_re = nearest_band_index(wv, rededge_nm, tol_nm) + 1
        b_swir1 = nearest_band_index(wv, swir1_nm, tol_nm) + 1

        b_w1c = nearest_band_index(wv, w1_center, tol_nm) + 1
        b_w1l = nearest_band_index(wv, w1_left, tol_nm) + 1
        b_w1r = nearest_band_index(wv, w1_right, tol_nm) + 1

        b_w2c = nearest_band_index(wv, w2_center, tol_nm) + 1
        b_w2l = nearest_band_index(wv, w2_left, tol_nm) + 1
        b_w2r = nearest_band_index(wv, w2_right, tol_nm) + 1

        bands_idx = {
            "RED": b_red,
            "GREEN": b_green,
            "NIR": b_nir,
            "REDEDGE": b_re,
            "SWIR1": b_swir1,
            "W1C": b_w1c,
            "W1L": b_w1l,
            "W1R": b_w1r,
            "W2C": b_w2c,
            "W2L": b_w2l,
            "W2R": b_w2r,
        }

        bands = read_scale_and_clip_bands(
            src,
            bands=bands_idx,
            scale=scale,
            min_val=min_val,
            max_val=max_val,
            verbose=verbose
        )

        red = bands["RED"]
        green = bands["GREEN"]
        nir = bands["NIR"]
        rededge = bands["REDEDGE"]
        swir1 = bands["SWIR1"]

        w1c, w1l, w1r = bands["W1C"], bands["W1L"], bands["W1R"]
        w2c, w2l, w2r = bands["W2C"], bands["W2L"], bands["W2R"]

        # invalid / nodata
        nd = src.nodata
        invalid = (
            ~np.isfinite(red) | ~np.isfinite(green) | ~np.isfinite(nir) |
            ~np.isfinite(rededge) | ~np.isfinite(swir1) |
            ~np.isfinite(w1c) | ~np.isfinite(w1l) | ~np.isfinite(w1r) |
            ~np.isfinite(w2c) | ~np.isfinite(w2l) | ~np.isfinite(w2r)
        )
        if nd is not None:
            nd = float(nd)
            invalid |= (
                (red == nd) | (green == nd) | (nir == nd) |
                (rededge == nd) | (swir1 == nd) |
                (w1c == nd) | (w1l == nd) | (w1r == nd) |
                (w2c == nd) | (w2l == nd) | (w2r == nd)
            )

        # Indices
        ndvi = safe_norm_diff(nir, red, invalid);        ndvi[ndvi == nodata_f32] = np.nan
        gndvi = safe_norm_diff(nir, green, invalid);     gndvi[gndvi == nodata_f32] = np.nan
        ndre = safe_norm_diff(nir, rededge, invalid);    ndre[ndre == nodata_f32] = np.nan
        ndwi_gao = safe_norm_diff(nir, swir1, invalid);  ndwi_gao[ndwi_gao == nodata_f32] = np.nan

        msavi_idx = msavi(nir, red, invalid);            msavi_idx[msavi_idx == nodata_f32] = np.nan

        # Masque végétation (float 0/1 + sortie int16 -1/0/255)
        veg = np.full(ndvi.shape, np.nan, dtype=np.float32)
        veg[np.isfinite(ndvi) & (ndvi <= ndvi_th)] = 0.0
        veg[np.isfinite(ndvi) & (ndvi >  ndvi_th)] = 1.0

        veg_out = np.full(veg.shape, -1, dtype=np.int16)
        veg_out[veg == 0] = 0
        veg_out[veg == 1] = 255

        # VII masqué sur végétation
        vii_veg = np.full(vii_zhang.shape, np.nan, dtype=np.float32)
        vii_veg[veg == 1] = vii_zhang[veg == 1]
        vii_veg[veg == 0] = 0.0

        # LWAI + WDI
        lam_w1l, lam_w1c, lam_w1r = float(wv[b_w1l - 1]), float(wv[b_w1c - 1]), float(wv[b_w1r - 1])
        lam_w2l, lam_w2c, lam_w2r = float(wv[b_w2l - 1]), float(wv[b_w2c - 1]), float(wv[b_w2r - 1])

        lwai_968 = band_depth(w1l, w1c, w1r, lam_w1l, lam_w1c, lam_w1r)
        lwai_1181 = band_depth(w2l, w2c, w2r, lam_w2l, lam_w2c, lam_w2r)

        lwai_968 = np.where((lwai_968 < 0) | (lwai_968 > 1), np.nan, lwai_968).astype(np.float32)
        lwai_1181 = np.where((lwai_1181 < 0) | (lwai_1181 > 1), np.nan, lwai_1181).astype(np.float32)

        mask_k = (
            (veg == 1) &
            np.isfinite(lwai_968) & np.isfinite(lwai_1181) &
            (ndvi > ndvi_k_th) &
            (lwai_968 > lwai_968_th) &
            (lwai_1181 > lwai_1181_th)
        )

        mean_968 = np.nanmean(lwai_968[mask_k])
        mean_1181 = np.nanmean(lwai_1181[mask_k])
        if not np.isfinite(mean_968) or mean_968 == 0 or not np.isfinite(mean_1181):
            k = np.nan
            wdi = np.full(lwai_968.shape, np.nan, dtype=np.float32)
        else:
            k = float(mean_1181 / mean_968)
            wdi = (k * lwai_968 - lwai_1181).astype(np.float32)

        # Indices masqués végétation
        ndwi_veg = mask_to_veg(ndwi_gao, veg)
        msavi_veg = mask_to_veg(msavi_idx, veg)
        gndvi_veg = mask_to_veg(gndvi, veg)
        wdi_veg = mask_to_veg(wdi, veg)

    # -------------------------
    # Écriture des outputs
    # -------------------------
    paths = {}

    def w(path: Path, arr: np.ndarray, dtype: str, nodata=None):
        write_imagej_tiff(str(path), arr, dtype=dtype, nodata=nodata)
        paths[path.stem] = path

    w(outdir / f"{prefix}_NDVI.tiff", ndvi, "float32")
    w(outdir / f"{prefix}_GNDVI.tiff", gndvi, "float32")
    w(outdir / f"{prefix}_NDRE.tiff", ndre, "float32")
    w(outdir / f"{prefix}_NDWI_Gao.tiff", ndwi_gao, "float32")
    w(outdir / f"{prefix}_MSAVI.tiff", msavi_idx, "float32")

    w(outdir / f"{prefix}_VII_Zhang2012.tiff", vii_zhang, "float32")
    w(outdir / f"{prefix}_VII_Zhang2012_veg.tiff", vii_veg, "float32")

    w(outdir / f"{prefix}_LWAI_968.tiff", lwai_968, "float32")
    w(outdir / f"{prefix}_LWAI_1181.tiff", lwai_1181, "float32")
    w(outdir / f"{prefix}_WDI_article.tiff", wdi, "float32")

    w(outdir / f"{prefix}_VEG_MASK.tiff", veg_out, "int16", nodata=-1)
    w(outdir / f"{prefix}_NDWI_veg.tiff", ndwi_veg, "float32")
    w(outdir / f"{prefix}_MSAVI_veg.tiff", msavi_veg, "float32")
    w(outdir / f"{prefix}_WDI_veg.tiff", wdi_veg, "float32")
    w(outdir / f"{prefix}_GNDVI_veg.tiff", gndvi_veg, "float32")

    # Visuel RGB du masque veg
    rgb = np.zeros((*veg_out.shape, 3), dtype=np.uint8)
    rgb[veg_out == -1] = [255, 0, 0]     # exclu
    rgb[veg_out == 255] = [0, 255, 0]    # végétation
    rgb[veg_out == 0] = [160, 160, 160]  # non-végétation

    p_vis = outdir / f"{prefix}_VEG_MASK_VISUAL.tiff"
    tifffile.imwrite(str(p_vis), rgb, photometric="rgb", imagej=True)
    paths[p_vis.stem] = p_vis

    # -------------------------
    # Logs
    # -------------------------
    if verbose:
        print("OK.")
        print(f"Tolérance utilisée: {tol_nm:.1f} nm")
        print("Bandes utilisées (cible -> bande 1-based -> nm réel) :")
        print(f"  RED      {red_nm}     -> band {b_red}   -> {float(wv[b_red-1]):.1f} nm")
        print(f"  GREEN    {green_nm}   -> band {b_green} -> {float(wv[b_green-1]):.1f} nm")
        print(f"  NIR      {nir_nm}     -> band {b_nir}   -> {float(wv[b_nir-1]):.1f} nm")
        print(f"  REDEDGE  {rededge_nm} -> band {b_re}    -> {float(wv[b_re-1]):.1f} nm")
        print(f"  SWIR1    {swir1_nm}   -> band {b_swir1} -> {float(wv[b_swir1-1]):.1f} nm")

        print(
            f"VII_Zhang2012: green[{vii_green_lo}-{vii_green_hi}] nm -> {n_green_bands} bandes | "
            f"nir[{vii_nir_lo}-{vii_nir_hi}] nm -> {n_nir_bands} bandes"
        )
        print("WDI water bands :")
        print(f"  W1_LEFT   {w1_left}   -> band {b_w1l} -> {float(wv[b_w1l-1]):.1f} nm")
        print(f"  W1_CENTER {w1_center} -> band {b_w1c} -> {float(wv[b_w1c-1]):.1f} nm")
        print(f"  W1_RIGHT  {w1_right}  -> band {b_w1r} -> {float(wv[b_w1r-1]):.1f} nm")
        print(f"  W2_LEFT   {w2_left}   -> band {b_w2l} -> {float(wv[b_w2l-1]):.1f} nm")
        print(f"  W2_CENTER {w2_center} -> band {b_w2c} -> {float(wv[b_w2c-1]):.1f} nm")
        print(f"  W2_RIGHT  {w2_right}  -> band {b_w2r} -> {float(wv[b_w2r-1]):.1f} nm")

        print("VII_Zhang2012 min/max/mean:", np.nanmin(vii_zhang), np.nanmax(vii_zhang), np.nanmean(vii_zhang))
        print("LWAI_968  min/max/mean:", np.nanmin(lwai_968), np.nanmax(lwai_968), np.nanmean(lwai_968))
        print("LWAI_1181 min/max/mean:", np.nanmin(lwai_1181), np.nanmax(lwai_1181), np.nanmean(lwai_1181))
        print("WDI min/max/mean:", np.nanmin(wdi), np.nanmax(wdi), np.nanmean(wdi))
        print("k =", k)

    return {
        "tol_nm": float(tol_nm),
        "wavelengths_nm": wv,
        "bands_1based": {
            "RED": int(b_red), "GREEN": int(b_green), "NIR": int(b_nir),
            "REDEDGE": int(b_re), "SWIR1": int(b_swir1),
            "W1L": int(b_w1l), "W1C": int(b_w1c), "W1R": int(b_w1r),
            "W2L": int(b_w2l), "W2C": int(b_w2c), "W2R": int(b_w2r),
        },
        "counts_vii": {"n_green_bands": int(n_green_bands), "n_nir_bands": int(n_nir_bands)},
        "k": float(k) if np.isfinite(k) else np.nan,
        "arrays": {
            "ndvi": ndvi, "gndvi": gndvi, "ndre": ndre, "ndwi_gao": ndwi_gao, "msavi": msavi_idx,
            "vii_zhang2012": vii_zhang, "vii_zhang2012_veg": vii_veg,
            "lwai_968": lwai_968, "lwai_1181": lwai_1181, "wdi": wdi,
            "veg_mask_out": veg_out,
            "ndwi_veg": ndwi_veg, "msavi_veg": msavi_veg, "wdi_veg": wdi_veg, "gndvi_veg": gndvi_veg,
        },
        "paths": paths,
    }
