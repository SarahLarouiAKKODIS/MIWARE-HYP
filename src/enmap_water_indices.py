import os
import numpy as np
import rasterio
import tifffile
from enmap_indices_calculation_utils import load_wavelengths_from_csv, compute_auto_tol_nm, nearest_band_index, safe_norm_diff, write_imagej_tiff, msavi, read_scale_and_clip_bands


# ============================================================
# SCRIPT EnMAP : indices + masque EAU à partir d'un GeoTIFF
# Sorties :
#   - NDVI
#   - GNDVI
#   - NDRE
#   - NDWI (Gao) = (NIR-SWIR1)/(NIR+SWIR1)  [humidité végétation]
#   - MNDWI (Xu) = (GREEN-SWIR1)/(GREEN+SWIR1) [détection eau]
#   - masque eau (0/255) basé sur MNDWI
#   - indices masqués sur zones eau (optionnel)
# ============================================================

# =========================
# PARAMÈTRES "EN DUR"
# =========================
TIF_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"
WAVELENGTHS_CSV = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full.csv"

OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs"
PREFIX = "enmap_salsigne"

AUTO_CONVERT_UM_TO_NM = True
USE_AUTO_TOL = True

# Longueurs d'onde cibles (nm)
RED_NM = 665.0
GREEN_NM = 560.0
REDEDGE_NM = 705.0
NIR_NM = 865.0
SWIR1_NM = 1610.0

# Seuil MNDWI pour "eau" (à ajuster)
MNDWI_TH = 0.55

NODATA_F32 = -9999.0
TOL_NM_FIXED = 12.0

scale = 10000.0

# bornes physiques raisonnables
MIN, MAX = 0.0, 1.2
# =========================
# MAIN
# =========================
os.makedirs(OUTDIR, exist_ok=True)

wv = load_wavelengths_from_csv(WAVELENGTHS_CSV)
if AUTO_CONVERT_UM_TO_NM and wv.max() < 50:
    wv = wv * 1000.0

tol_nm = compute_auto_tol_nm(wv) if USE_AUTO_TOL else TOL_NM_FIXED

with rasterio.open(TIF_PATH) as src:
    profile = src.profile.copy()
    nb = src.count
    if len(wv) != nb:
        raise SystemExit(f"len(wavelengths)={len(wv)} != nb_bandes_tif={nb}")

    # Match bandes (0-based) -> rasterio (1-based)
    b_green = nearest_band_index(wv, GREEN_NM, tol_nm) + 1
    b_swir1 = nearest_band_index(wv, SWIR1_NM, tol_nm) + 1

    bands_idx = {
    "GREEN": b_green,
    "SWIR1": b_swir1,
    }

    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    # accès direct
    green = bands["GREEN"]
    swir1 = bands["SWIR1"]


    # Invalid (attention: pour l'eau, on ne dépend pas de rededge/ndvi si tu veux simplifier)
    nd = src.nodata
    invalid = (
         ~np.isfinite(green) | ~np.isfinite(swir1)
    )
    if nd is not None:
        nd = float(nd)
        invalid |= (green == nd) | (swir1 == nd)
   


    print("GREEN:", np.nanmin(green), np.nanmax(green))
    print("swir1:", np.nanmin(swir1), np.nanmax(swir1))

    # MNDWI (Xu) (détection eau)
    mndwi = safe_norm_diff(green, swir1, invalid);  mndwi[mndwi == NODATA_F32] = np.nan

    # =========================
    # MASQUE EAU (0 / 255 / -1)
    # =========================

    water = np.full(mndwi.shape, np.nan, dtype=np.float32)
    water[np.isfinite(mndwi) & (mndwi <= MNDWI_TH)] = 0
    water[np.isfinite(mndwi) & (mndwi >  MNDWI_TH)] = 1

    water_out = np.full(water.shape, -1, dtype=np.int16)  # -1 = NoData (pixels exclus)
    water_out[water == 0] = 0
    water_out[water == 1] = 255

 

# =========================
# ÉCRITURE
# =========================

write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_MNDWI_Xu.tiff"), mndwi, dtype="float32")
write_imagej_tiff(os.path.join(OUTDIR, f"{PREFIX}_WATER_MASK.tiff"), water_out, dtype="int16", nodata=-1)


# Visual RGB (rouge=exclu, bleu=eau, gris=non-eau)
rgb = np.zeros((*water_out.shape, 3), dtype=np.uint8)
rgb[water_out == -1] = [255, 0, 0]      # exclu = rouge
rgb[water_out == 255] = [0, 0, 255]     # eau = bleu
rgb[water_out == 0]   = [160, 160, 160] # non-eau = gris

tifffile.imwrite(
    os.path.join(OUTDIR, f"{PREFIX}_WATER_MASK_VISUAL.tiff"),
    rgb,
    photometric="rgb",
    imagej=True
)

# Logs
print("OK.")
print(f"Tolérance utilisée: {tol_nm:.1f} nm")
print("Bandes utilisées (cible -> bande 1-based -> nm réel) :")
print(f"  GREEN    {GREEN_NM} -> band {b_green}  -> {float(wv[b_green-1]):.1f} nm")
print(f"  SWIR1    {SWIR1_NM} -> band {b_swir1}  -> {float(wv[b_swir1-1]):.1f} nm")
print(f"Seuil eau MNDWI_TH = {MNDWI_TH}")
