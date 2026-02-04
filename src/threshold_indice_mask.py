import os
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# SCRIPT GÉNÉRAL : seuillage d'une image indice + masque labelisé + RGB + légende
#
# Méthodes de seuil :
#   - "otsu"
#   - "mean_std"  -> mean + K_STD * std (sur valeurs clippées)
#   - "p98" (ou "p95", "p99", etc.) -> percentile sur valeurs NON clippées (robuste)
#
# Convention d'entrée typique (comme tes *_veg) :
#   - 0   = non-végétation / hors analyse
#   - NaN = pixels à exclure
#   - valeurs != 0 = pixels analysables
#
# Option : masque eau
#   - 255 = eau
#   - 0   = pas eau
#   - -1  = exclu
# ============================================================

# =========================
# PARAMÈTRES À MODIFIER
# =========================

IN_TIF = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_WDI_veg.tiff"
WATER_MASK_TIF = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"

OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Threshold_index"
PREFIX = "enmap_salsigne_WDI"

# Nom utilisé dans la légende
INDEX_NAME = "Indice"
# ex: "VII (Zhang 2012)" ou "WDI"

# Méthode de seuil : "otsu", "mean_std", "p98", "p95", ...
THRESH_METHOD = "p98"

# Pour mean_std
K_STD = 2.0

# Clip robuste (utilisé pour otsu et mean_std)
CLIP_PCT_LOW = 1.0
CLIP_PCT_HIGH = 99.0

# Règles de validité (cas typique *_veg)
EXCLUDE_ZERO = True       # 0 = hors analyse (non-végétation)
EPS_ZERO = 0.0            # si >0, exclusion des "quasi zéro" : abs(x) > EPS_ZERO

# Classes
LABEL_NODATA = 255
LABEL_LOW = 0
LABEL_HIGH = 1

# Couleurs RGB
COLOR_NODATA = (255, 255, 255)  # blanc
COLOR_LOW    = (160, 160, 160)  # gris
COLOR_HIGH   = (255, 0, 255)    # magenta

os.makedirs(OUTDIR, exist_ok=True)

# =========================
# OUTILS
# =========================
def otsu_threshold(values, nbins=512):
    """Otsu sur un vecteur 1D (valeurs finies)."""
    values = values.astype(np.float64)
    vmin, vmax = np.min(values), np.max(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float(vmin)

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    omega = np.cumsum(p)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    return float(centers[idx])



def parse_method(method_str: str):
    """
    Retourne ("otsu"|"mean_std"|"percentile", pct)
    pct n'est défini que si percentile.
    """
    m = method_str.strip().lower()
    if m == "otsu":
        return ("otsu", None)
    if m in ("mean_std", "meanstd", "mean+std", "mean2std"):
        return ("mean_std", None)
    if m.startswith("p"):
        try:
            pct = float(m[1:])
            if not (0 < pct < 100):
                raise ValueError
            return ("percentile", pct)
        except:
            raise ValueError('THRESH_METHOD percentile invalide. Ex: "p98", "p95".')
    raise ValueError('THRESH_METHOD invalide. Choix: "otsu", "mean_std", "p98"...')


def build_valid_mask(arr, nodata_value, water_bool=None, exclude_zero=True, eps_zero=0.0):
    """Masque des pixels utilisables pour calculer le seuil."""
    valid = np.isfinite(arr)
    if nodata_value is not None:
        valid &= (arr != nodata_value)

    if exclude_zero:
        if eps_zero > 0:
            valid &= (np.abs(arr) > eps_zero)
        else:
            valid &= (arr != 0)

    if water_bool is not None:
        valid &= (~water_bool)

    return valid


# =========================
# MAIN
# =========================
with rasterio.open(IN_TIF) as src:
    arr = src.read(1).astype(np.float32)
    profile = src.profile.copy()
    nodata = src.nodata

# Masque eau optionnel
water_bool = None
if WATER_MASK_TIF is not None:
    with rasterio.open(WATER_MASK_TIF) as wm:
        water_mask = wm.read(1)

    if water_mask.shape != arr.shape:
        raise SystemExit(f"Masque eau shape {water_mask.shape} != image indice shape {arr.shape}")

    # Ton codage : 255 = eau
    water_bool = (water_mask == 255)

# Pixels valides
valid = build_valid_mask(arr, nodata, water_bool, EXCLUDE_ZERO, EPS_ZERO)
vals = arr[valid]

if vals.size < 100:
    raise SystemExit("Pas assez de pixels valides pour calculer un seuil.")

# Clip robuste (pour otsu et mean_std)
lo = np.percentile(vals, CLIP_PCT_LOW)
hi = np.percentile(vals, CLIP_PCT_HIGH)
vals_clip = np.clip(vals, lo, hi)

mode, pct = parse_method(THRESH_METHOD)

# Calcul seuil
if mode == "otsu":
    thr = otsu_threshold(vals_clip)
elif mode == "mean_std":
    thr = float(np.mean(vals_clip) + K_STD * np.std(vals_clip))
elif mode == "percentile":
    thr = float(np.nanpercentile(vals, pct))  # percentile sur vals non clippées
else:
    raise RuntimeError("Mode inconnu")

pct_high = 100 * np.mean(vals >= thr)
print(f"[INFO] Seuil {INDEX_NAME} ({THRESH_METHOD}) = {thr:.4f}")
print(f"[INFO] Pixels analysés = {vals.size}")
print(f"[INFO] % pixels HIGH ≈ {pct_high:.2f}%")

# Masque labelisé
mask = np.full(arr.shape, LABEL_NODATA, dtype=np.uint8)
mask[valid & (arr < thr)]  = LABEL_LOW
mask[valid & (arr >= thr)] = LABEL_HIGH

# RGB
rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
rgb[mask == LABEL_NODATA] = COLOR_NODATA
rgb[mask == LABEL_LOW]    = COLOR_LOW
rgb[mask == LABEL_HIGH]   = COLOR_HIGH

# Écriture masque GeoTIFF
mask_profile = profile.copy()
mask_profile.update(dtype=rasterio.uint8, count=1, nodata=LABEL_NODATA, compress="lzw")

mask_path = os.path.join(OUTDIR, f"{PREFIX}_mask.tif")
with rasterio.open(mask_path, "w", **mask_profile) as dst:
    dst.write(mask, 1)

# Écriture RGB GeoTIFF (3 bandes) - écriture d'un coup
rgb_profile = profile.copy()
rgb_profile.update(driver="GTiff", dtype=rasterio.uint8, count=3, compress="lzw")
rgb_profile.pop("nodata", None)
rgb_profile["photometric"] = "RGB"

rgb_path = os.path.join(OUTDIR, f"{PREFIX}_mask_rgb.tif")
with rasterio.open(rgb_path, "w", **rgb_profile) as dst:
    dst.write(np.transpose(rgb, (2, 0, 1)).astype(np.uint8))

print("[OK] Fichiers générés :")
print(" -", mask_path)
print(" -", rgb_path)
