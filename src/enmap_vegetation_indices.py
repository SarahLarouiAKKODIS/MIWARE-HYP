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
)

# ============================================================
# SCRIPT EnMAP : indices + masque végétation + WAI/WDI (absorption eau + décorrélation)
#
# Entrées :
#   - GeoTIFF hyperspectral multibande (EnMAP)
#   - fichier longueurs d'onde (1 valeur par bande, même ordre que le tif)
#
# Sorties :
#   - NDVI
#   - VII (ici: GNDVI = (NIR-GREEN)/(NIR+GREEN))
#   - NDRE
#   - NDWI (Gao) = (NIR-SWIR1)/(NIR+SWIR1)
#   - MSAVI
#   - WAI_900_970 = (REFW - WABS)/(REFW + WABS)
#   - WDI_decorrelated = WAI - (a*NDVI + b) appris sur pixels végétalisés
#   - masque végétation (int16: -1 nodata, 0 non-veg, 255 veg)
#   - NDWI/MSAVI/WDI masqués sur zones végétalisées (0 ailleurs)
#   - visuel RGB du masque
# ============================================================

# =========================
# PARAMÈTRES "EN DUR"
# =========================
TIF_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"
WAVELENGTHS_CSV = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full.csv"

OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs"
PREFIX = "enmap_salsigne"

AUTO_CONVERT_UM_TO_NM = True

USE_AUTO_TOL = True
TOL_NM_FIXED = 12.0

# Longueurs d'onde (nm)
RED_NM = 665.0
GREEN_NM = 560.0
REDEDGE_NM = 705.0
NIR_NM = 865.0
SWIR1_NM = 1610.0

# Pour WAI/WDI (absorption eau autour de 970 nm)
REF_WATER_NM = 900.0
WATER_ABS_NM = 970.0

# Seuil NDVI "végétation"
NDVI_TH = 0.3
# Seuil NDVI pour apprendre la décorrélation WDI (souvent même valeur)
NDVI_TH_WDI = 0.3

# Nodata float logique (les fonctions utilitaires peuvent l'utiliser)
NODATA_F32 = -9999.0

# Scale + clipping
scale = 10000.0
MIN, MAX = 0.0, 1.2


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

    # 2) match bandes (0-based) -> rasterio (1-based)
    b_red = nearest_band_index(wv, RED_NM, tol_nm) + 1
    b_green = nearest_band_index(wv, GREEN_NM, tol_nm) + 1
    b_nir = nearest_band_index(wv, NIR_NM, tol_nm) + 1
    b_re = nearest_band_index(wv, REDEDGE_NM, tol_nm) + 1
    b_swir1 = nearest_band_index(wv, SWIR1_NM, tol_nm) + 1

    b_refw = nearest_band_index(wv, REF_WATER_NM, tol_nm) + 1
    b_wabs = nearest_band_index(wv, WATER_ABS_NM, tol_nm) + 1

    bands_idx = {
        "RED": b_red,
        "GREEN": b_green,
        "NIR": b_nir,
        "REDEDGE": b_re,
        "SWIR1": b_swir1,
        "REFW": b_refw,   # ~900
        "WABS": b_wabs,   # ~970
    }

    # 3) lecture + scale + clip
    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    red = bands["RED"]
    green = bands["GREEN"]
    nir = bands["NIR"]
    rededge = bands["REDEDGE"]
    swir1 = bands["SWIR1"]
    refw = bands["REFW"]
    wabs = bands["WABS"]

    # 4) gestion nodata/invalid
    nd = src.nodata
    invalid = (
        ~np.isfinite(red) | ~np.isfinite(green) | ~np.isfinite(nir) |
        ~np.isfinite(rededge) | ~np.isfinite(swir1) |
        ~np.isfinite(refw) | ~np.isfinite(wabs)
    )
    if nd is not None:
        nd = float(nd)
        invalid |= (
            (red == nd) | (green == nd) | (nir == nd) |
            (rededge == nd) | (swir1 == nd) |
            (refw == nd) | (wabs == nd)
        )

    # 5) indices principaux
    ndvi = safe_norm_diff(nir, red, invalid)
    ndvi[ndvi == NODATA_F32] = np.nan

    # VII -> ici GNDVI (Green NDVI)
    vii_gndvi = safe_norm_diff(nir, green, invalid)
    vii_gndvi[vii_gndvi == NODATA_F32] = np.nan

    ndre = safe_norm_diff(nir, rededge, invalid)
    ndre[ndre == NODATA_F32] = np.nan

    ndwi_gao = safe_norm_diff(nir, swir1, invalid)
    ndwi_gao[ndwi_gao == NODATA_F32] = np.nan

    msavi_idx = msavi(nir, red, invalid)
    msavi_idx[msavi_idx == NODATA_F32] = np.nan

    # 6) masque végétation (int16 : -1 nodata, 0 non-veg, 255 veg)
    veg = np.full(ndvi.shape, np.nan, dtype=np.float32)
    veg[np.isfinite(ndvi) & (ndvi <= NDVI_TH)] = 0
    veg[np.isfinite(ndvi) & (ndvi > NDVI_TH)] = 1

    veg_out = np.full(veg.shape, -1, dtype=np.int16)
    veg_out[veg == 0] = 0
    veg_out[veg == 1] = 255

    # 7) indices masqués végétation
    ndwi_veg = np.full(ndwi_gao.shape, np.nan, dtype=np.float32)
    ndwi_veg[veg == 1] = ndwi_gao[veg == 1]
    ndwi_veg[veg == 0] = 0.0

    msavi_veg = np.full(msavi_idx.shape, np.nan, dtype=np.float32)
    msavi_veg[veg == 1] = msavi_idx[veg == 1]
    msavi_veg[veg == 0] = 0.0

    vii_veg = np.full(vii_gndvi.shape, np.nan, dtype=np.float32)
    vii_veg[veg == 1] = vii_gndvi[veg == 1]
    vii_veg[veg == 0] = 0.0

    # ============================================================
    # 8) WAI + WDI (absorption d'eau + "decorrelated" de NDVI)
    # WAI_900_970 = (REFW - WABS)/(REFW + WABS)
    # WDI = résidu : WAI - (a*NDVI + b) appris sur pixels végétation
    # ============================================================
    wai = safe_norm_diff(refw, wabs, invalid)
    wai[wai == NODATA_F32] = np.nan

    wdi = np.full(wai.shape, np.nan, dtype=np.float32)

    fit_mask = np.isfinite(wai) & np.isfinite(ndvi) & (ndvi > NDVI_TH_WDI)

    if int(np.sum(fit_mask)) >= 100:
        x = ndvi[fit_mask].astype(np.float32)
        y = wai[fit_mask].astype(np.float32)
        a, b = np.polyfit(x, y, deg=1)  # y ≈ a*x + b
        wdi = (wai - (a * ndvi + b)).astype(np.float32)
        print(f"WDI fit: a={a:.6f}, b={b:.6f}, fit_pixels={int(np.sum(fit_mask))}")
    else:
        print(f"⚠️ Pas assez de pixels (fit_pixels={int(np.sum(fit_mask))}) pour estimer WDI. WDI restera NaN.")

    wdi_veg = np.full(wdi.shape, np.nan, dtype=np.float32)
    wdi_veg[veg == 1] = wdi[veg == 1]
    wdi_veg[veg == 0] = 0.0

# =========================
# 9) ÉCRITURE
# =========================
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDVI.tiff"), ndvi, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VII_GNDVI.tiff"), vii_gndvi, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDRE.tiff"), ndre, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDWI_Gao.tiff"), ndwi_gao, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_MSAVI.tiff"), msavi_idx, dtype="float32")

write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WAI_{int(REF_WATER_NM)}_{int(WATER_ABS_NM)}.tiff"), wai, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WDI_decorrelated.tiff"), wdi, dtype="float32")

write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VEG_MASK.tiff"), veg_out, dtype="int16", nodata=-1)
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_NDWI_veg.tiff"), ndwi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_MSAVI_veg.tiff"), msavi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WDI_veg.tiff"), wdi_veg, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_VII_veg.tiff"), vii_veg, dtype="float32")


# Visuel RGB du masque
rgb = np.zeros((*veg_out.shape, 3), dtype=np.uint8)
rgb[veg_out == -1] = [255, 0, 0]       # rouge = pixels exclus (-1)
rgb[veg_out == 255] = [0, 255, 0]      # vert = végétation (255)
rgb[veg_out == 0] = [160, 160, 160]    # gris = non-végétation (0)

tifffile.imwrite(
    os.path.join(OUTDIR, f"{PREFIX}_VEG_MASK_VISUAL.tiff"),
    rgb,
    photometric="rgb",
    imagej=True
)

# =========================
# 10) LOGS
# =========================
print("OK.")
print(f"Tolérance utilisée: {tol_nm:.1f} nm")
print("Bandes utilisées (cible -> bande 1-based -> nm réel) :")
print(f"  RED      {RED_NM}       -> band {b_red}   -> {float(wv[b_red-1]):.1f} nm")
print(f"  GREEN    {GREEN_NM}     -> band {b_green} -> {float(wv[b_green-1]):.1f} nm")
print(f"  NIR      {NIR_NM}       -> band {b_nir}   -> {float(wv[b_nir-1]):.1f} nm")
print(f"  REDEDGE  {REDEDGE_NM}   -> band {b_re}    -> {float(wv[b_re-1]):.1f} nm")
print(f"  SWIR1    {SWIR1_NM}     -> band {b_swir1} -> {float(wv[b_swir1-1]):.1f} nm")
print(f"  REFW     {REF_WATER_NM} -> band {b_refw}  -> {float(wv[b_refw-1]):.1f} nm")
print(f"  WABS     {WATER_ABS_NM} -> band {b_wabs}  -> {float(wv[b_wabs-1]):.1f} nm")
