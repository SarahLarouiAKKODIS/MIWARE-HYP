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
    band_depth
)

# ============================================================
# SCRIPT EnMAP : indices + masque végétation + WDI + VII (Zhang et al. 2012)
#
# Ajout : VII_Zhang2012 (vrai VII de l'article)
#   - a = somme(reflectance 497–635 nm)
#   - b = somme(reflectance 700–1200 nm)
#   - Na = a / mean(a)
#   - Nb = b / mean(b)
#   - VII = (Na - Nb)/(Na + Nb) * 100
# ============================================================

# =========================
# PARAMÈTRES
# =========================
TIF_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"
WAVELENGTHS_CSV = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full.csv"

OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs"
PREFIX = "enmap_salsigne"

AUTO_CONVERT_UM_TO_NM = True
USE_AUTO_TOL = True
TOL_NM_FIXED = 12.0

# Longueurs d'onde (nm) pour indices "classiques"
RED_NM = 665.0
GREEN_NM = 560.0
REDEDGE_NM = 705.0
NIR_NM = 865.0
SWIR1_NM = 1610.0

# Plages VII (Zhang et al. 2012)
VII_GREEN_LO, VII_GREEN_HI = 497.0, 635.0
VII_NIR_LO,   VII_NIR_HI   = 700.0, 1200.0

# Pour WDI (Zhang et al. 2012) : deux vallées d’absorption d’eau
W1_CENTER = 968.0
W1_LEFT   = 940.0
W1_RIGHT  = 990.0

W2_CENTER = 1181.0
W2_LEFT   = 1140.0
W2_RIGHT  = 1240.0

# Masque végétation
NDVI_TH = 0.3

# Nodata float logique
NODATA_F32 = -9999.0

# Scale + clipping
scale = 10000.0
MIN, MAX = 0.0, 1.2

# =========================
# HELPERS
# =========================
def band_indices_in_range(wv_nm, lo, hi):
    idx = np.where((wv_nm >= lo) & (wv_nm <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"Aucune bande trouvée dans [{lo}, {hi}] nm")
    return idx

def scale_clip_to_reflectance(arr, scale, min_val, max_val):
    """
    Convertit en réflectance (arr/scale) puis clip.
    Gère NaN / inf.
    """
    out = arr.astype(np.float32) / float(scale)
    out = np.where(np.isfinite(out), out, np.nan).astype(np.float32)
    out = np.clip(out, min_val, max_val).astype(np.float32)
    return out

def compute_vii_zhang2012_blockwise(src, wv_nm, scale, min_val, max_val,
                                   green_lo, green_hi, nir_lo, nir_hi):
    """
    Calcule VII (Zhang et al. 2012) en mode block/window :
      a = somme reflectance sur [green_lo, green_hi]
      b = somme reflectance sur [nir_lo, nir_hi]
      Na = a / mean(a)
      Nb = b / mean(b)
      VII = (Na - Nb)/(Na + Nb)*100
    """
    # Indices 0-based -> rasterio 1-based
    idx_green = band_indices_in_range(wv_nm, green_lo, green_hi)
    idx_nir   = band_indices_in_range(wv_nm, nir_lo, nir_hi)

    bands_green = (idx_green + 1).tolist()
    bands_nir   = (idx_nir + 1).tolist()

    h, w = src.height, src.width
    a = np.full((h, w), np.nan, dtype=np.float32)
    b = np.full((h, w), np.nan, dtype=np.float32)

    nd = src.nodata
    nd = float(nd) if nd is not None else None

    # 1) calcule a et b (somme des bandes) par bloc
    for (ji, window) in src.block_windows(1):
        # Lire les bandes de la fenêtre : (nbands, win_h, win_w)
        g_raw = src.read(indexes=bands_green, window=window)
        n_raw = src.read(indexes=bands_nir, window=window)

        g = scale_clip_to_reflectance(g_raw, scale, min_val, max_val)
        n = scale_clip_to_reflectance(n_raw, scale, min_val, max_val)

        # Si nodata défini, on masque les pixels où une bande vaut nodata (avant scale)
        # (option "strict": si une bande est nodata -> pixel nan)
        if nd is not None:
            g_mask = np.any(g_raw == nd, axis=0)
            n_mask = np.any(n_raw == nd, axis=0)
            bad = g_mask | n_mask
        else:
            bad = np.zeros((g.shape[1], g.shape[2]), dtype=bool)

        # Sommes
        a_win = np.nansum(g, axis=0).astype(np.float32)
        b_win = np.nansum(n, axis=0).astype(np.float32)

        # Pixels invalides -> nan
        a_win = np.where(bad, np.nan, a_win).astype(np.float32)
        b_win = np.where(bad, np.nan, b_win).astype(np.float32)

        r0 = window.row_off
        c0 = window.col_off
        a[r0:r0 + window.height, c0:c0 + window.width] = a_win
        b[r0:r0 + window.height, c0:c0 + window.width] = b_win

    # 2) normalisation image-wide
    a_mean = np.nanmean(a)
    b_mean = np.nanmean(b)

    Na = a / a_mean
    Nb = b / b_mean

    denom = (Na + Nb)
    vii = (Na - Nb) / denom * 100.0
    vii = np.where(np.isfinite(vii) & np.isfinite(denom) & (denom != 0), vii, np.nan).astype(np.float32)

    return vii, len(bands_green), len(bands_nir)

# =========================
# MAIN
# =========================
os.makedirs(OUTDIR, exist_ok=True)

# 1) longueurs d'onde
wv = load_wavelengths_from_csv(WAVELENGTHS_CSV)
if AUTO_CONVERT_UM_TO_NM and wv.max() < 50:
    wv = wv * 1000.0

tol_nm = compute_auto_tol_nm(wv) if USE_AUTO_TOL else TOL_NM_FIXED

with rasterio.open(TIF_PATH) as src:
    nb = src.count
    if len(wv) != nb:
        raise SystemExit(f"len(wavelengths)={len(wv)} != nb_bandes_tif={nb}")

    # ---------- VII (Zhang 2012) : calcul hyperspectral par plages ----------
    vii_zhang, n_green_bands, n_nir_bands = compute_vii_zhang2012_blockwise(
        src=src,
        wv_nm=wv,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        green_lo=VII_GREEN_LO,
        green_hi=VII_GREEN_HI,
        nir_lo=VII_NIR_LO,
        nir_hi=VII_NIR_HI
    )

    # 2) match bandes (0-based) -> rasterio (1-based) pour les indices "classiques" + WDI
    b_red   = nearest_band_index(wv, RED_NM, tol_nm) + 1
    b_green = nearest_band_index(wv, GREEN_NM, tol_nm) + 1
    b_nir   = nearest_band_index(wv, NIR_NM, tol_nm) + 1
    b_re    = nearest_band_index(wv, REDEDGE_NM, tol_nm) + 1
    b_swir1 = nearest_band_index(wv, SWIR1_NM, tol_nm) + 1

    b_w1c = nearest_band_index(wv, W1_CENTER, tol_nm) + 1
    b_w1l = nearest_band_index(wv, W1_LEFT,   tol_nm) + 1
    b_w1r = nearest_band_index(wv, W1_RIGHT,  tol_nm) + 1

    b_w2c = nearest_band_index(wv, W2_CENTER, tol_nm) + 1
    b_w2l = nearest_band_index(wv, W2_LEFT,   tol_nm) + 1
    b_w2r = nearest_band_index(wv, W2_RIGHT,  tol_nm) + 1

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

    # 3) lecture + scale + clip (indices "classiques" + WDI)
    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    red     = bands["RED"]
    green   = bands["GREEN"]
    nir     = bands["NIR"]
    rededge = bands["REDEDGE"]
    swir1   = bands["SWIR1"]

    w1c = bands["W1C"]
    w1l = bands["W1L"]
    w1r = bands["W1R"]

    w2c = bands["W2C"]
    w2l = bands["W2L"]
    w2r = bands["W2R"]

    # 4) gestion nodata / invalid (inclut aussi les bandes WDI)
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

    # 5) indices principaux
    ndvi = safe_norm_diff(nir, red, invalid)
    ndvi[ndvi == NODATA_F32] = np.nan

    # GNDVI (utile, mais ce n'est PAS le VII de l'article)
    gndvi = safe_norm_diff(nir, green, invalid)
    gndvi[gndvi == NODATA_F32] = np.nan

    ndre = safe_norm_diff(nir, rededge, invalid)
    ndre[ndre == NODATA_F32] = np.nan

    ndwi_gao = safe_norm_diff(nir, swir1, invalid)
    ndwi_gao[ndwi_gao == NODATA_F32] = np.nan

    msavi_idx = msavi(nir, red, invalid)
    msavi_idx[msavi_idx == NODATA_F32] = np.nan

    # 6) masque végétation
    veg = np.full(ndvi.shape, np.nan, dtype=np.float32)
    veg[np.isfinite(ndvi) & (ndvi <= NDVI_TH)] = 0.0
    veg[np.isfinite(ndvi) & (ndvi >  NDVI_TH)] = 1.0

    veg_out = np.full(veg.shape, -1, dtype=np.int16)
    veg_out[veg == 0] = 0
    veg_out[veg == 1] = 255

    # 6bis) masque VII sur végétation
    vii_veg = np.full(vii_zhang.shape, np.nan, dtype=np.float32)
    vii_veg[veg == 1] = vii_zhang[veg == 1]
    vii_veg[veg == 0] = 0.0

    # 7) LWAI (band depth) + WDI (Zhang et al.)
    lam_w1l = float(wv[b_w1l - 1])
    lam_w1c = float(wv[b_w1c - 1])
    lam_w1r = float(wv[b_w1r - 1])

    lam_w2l = float(wv[b_w2l - 1])
    lam_w2c = float(wv[b_w2c - 1])
    lam_w2r = float(wv[b_w2r - 1])

    lwai_968  = band_depth(w1l, w1c, w1r, lam_w1l, lam_w1c, lam_w1r)
    lwai_1181 = band_depth(w2l, w2c, w2r, lam_w2l, lam_w2c, lam_w2r)

    lwai_968  = np.where((lwai_968  < 0) | (lwai_968  > 1), np.nan, lwai_968)
    lwai_1181 = np.where((lwai_1181 < 0) | (lwai_1181 > 1), np.nan, lwai_1181)

    mask_k = (veg == 1) & np.isfinite(lwai_968) & np.isfinite(lwai_1181) \
         & (ndvi > 0.45) & (lwai_968 > 0.02) & (lwai_1181 > 0.02)

    k = np.nanmean(lwai_1181[mask_k]) / np.nanmean(lwai_968[mask_k])
    wdi = k * lwai_968 - lwai_1181

    # 8) indices masqués végétation (0 ailleurs)
    def mask_to_veg(arr):
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        out[veg == 1] = arr[veg == 1]
        out[veg == 0] = 0.0
        return out

    ndwi_veg  = mask_to_veg(ndwi_gao)
    msavi_veg = mask_to_veg(msavi_idx)
    gndvi_veg = mask_to_veg(gndvi)
    wdi_veg   = mask_to_veg(wdi)

# =========================
# ÉCRITURE
# =========================
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDVI.tiff"), ndvi, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_GNDVI.tiff"), gndvi, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDRE.tiff"), ndre, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDWI_Gao.tiff"), ndwi_gao, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_MSAVI.tiff"), msavi_idx, dtype="float32")

# --- nouveaux outputs VII (article) ---
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VII_Zhang2012.tiff"), vii_zhang, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VII_Zhang2012_veg.tiff"), vii_veg, dtype="float32")

write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_LWAI_968.tiff"), lwai_968, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_LWAI_1181.tiff"), lwai_1181, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WDI_article.tiff"), wdi, dtype="float32")

write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VEG_MASK.tiff"), veg_out, dtype="int16", nodata=-1)
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDWI_veg.tiff"), ndwi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_MSAVI_veg.tiff"), msavi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WDI_veg.tiff"), wdi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_GNDVI_veg.tiff"), gndvi_veg, dtype="float32")

# Visuel RGB du masque veg
rgb = np.zeros((*veg_out.shape, 3), dtype=np.uint8)
rgb[veg_out == -1]  = [255, 0, 0]       # rouge = pixels exclus (-1)
rgb[veg_out == 255] = [0, 255, 0]       # vert = végétation
rgb[veg_out == 0]   = [160, 160, 160]   # gris = non-végétation

tifffile.imwrite(
    os.path.join(OUTDIR, f"{PREFIX}_VEG_MASK_VISUAL.tiff"),
    rgb,
    photometric="rgb",
    imagej=True
)

# =========================
# LOGS
# =========================
print("OK.")
print(f"Tolérance utilisée: {tol_nm:.1f} nm")
print("Bandes utilisées (cible -> bande 1-based -> nm réel) :")
print(f"  RED      {RED_NM}       -> band {b_red}   -> {float(wv[b_red-1]):.1f} nm")
print(f"  GREEN    {GREEN_NM}     -> band {b_green} -> {float(wv[b_green-1]):.1f} nm")
print(f"  NIR      {NIR_NM}       -> band {b_nir}   -> {float(wv[b_nir-1]):.1f} nm")
print(f"  REDEDGE  {REDEDGE_NM}   -> band {b_re}    -> {float(wv[b_re-1]):.1f} nm")
print(f"  SWIR1    {SWIR1_NM}     -> band {b_swir1} -> {float(wv[b_swir1-1]):.1f} nm")

print(f"VII_Zhang2012: green[{VII_GREEN_LO}-{VII_GREEN_HI}] nm -> {n_green_bands} bandes | "
      f"nir[{VII_NIR_LO}-{VII_NIR_HI}] nm -> {n_nir_bands} bandes")
print("WDI water bands :")
print(f"  W1_LEFT   {W1_LEFT}   -> band {b_w1l} -> {float(wv[b_w1l-1]):.1f} nm")
print(f"  W1_CENTER {W1_CENTER} -> band {b_w1c} -> {float(wv[b_w1c-1]):.1f} nm")
print(f"  W1_RIGHT  {W1_RIGHT}  -> band {b_w1r} -> {float(wv[b_w1r-1]):.1f} nm")
print(f"  W2_LEFT   {W2_LEFT}   -> band {b_w2l} -> {float(wv[b_w2l-1]):.1f} nm")
print(f"  W2_CENTER {W2_CENTER} -> band {b_w2c} -> {float(wv[b_w2c-1]):.1f} nm")
print(f"  W2_RIGHT  {W2_RIGHT}  -> band {b_w2r} -> {float(wv[b_w2r-1]):.1f} nm")

print("VII_Zhang2012 min/max/mean:",
      np.nanmin(vii_zhang), np.nanmax(vii_zhang), np.nanmean(vii_zhang))
print("LWAI_968  min/max/mean:",
      np.nanmin(lwai_968), np.nanmax(lwai_968), np.nanmean(lwai_968))
print("LWAI_1181 min/max/mean:",
      np.nanmin(lwai_1181), np.nanmax(lwai_1181), np.nanmean(lwai_1181))
print("WDI min/max/mean:",
      np.nanmin(wdi), np.nanmax(wdi), np.nanmean(wdi))
print("k =", k)
