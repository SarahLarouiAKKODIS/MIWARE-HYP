import numpy as np
import rasterio
import tifffile
import os

# ============================================================
# Script général : classification "santé végétation" à partir d’un indice
# - Lit une image d’indice (WDI / MSAVI / VII / NDVI / NDRE / NDWI…)
# - Utilise un masque végétation (255 = veg) pour ne classifier que la végétation
# - Ignore les NaN
# - Calcule des seuils par percentiles (par défaut 10/30/70)
# - Produit :
#     * classes.tif (uint8) : 0=non-veg/noData, 1..4 classes
#     * classes_RGB.tif (RGB)
# ============================================================

# ======================
# CONFIG A MODIFIER
# ======================
INDEX_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VII_GNDVI.tiff"
VEG_MASK_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VEG_MASK.tiff"  # 255 = végétation
OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegetation_health"
PREFIX = "VII_vegetation_health"

# Percentiles pour les seuils (4 classes -> 3 seuils)
PCTS = (10, 30, 70)

# Si True : valeurs élevées = meilleure santé (MSAVI, VII, NDVI…)
# Si False : valeurs élevées = pire santé (rare ; mets True dans la plupart des cas)
HIGH_IS_GOOD = True

# Option utile pour WDI : HIGH_IS_GOOD=True fonctionne aussi
# (WDI faible => stress, WDI élevé => plus humide/vigoureux)
# donc tu peux laisser HIGH_IS_GOOD=True

os.makedirs(OUTDIR, exist_ok=True)

# ======================
# FUNCTIONS
# ======================
def classify_health_from_index(
    index_arr: np.ndarray,
    veg_mask_arr: np.ndarray,
    pcts=(10, 30, 70),
    high_is_good=True,
    min_veg_pixels=100,
):
    """
    Retourne :
      classes (uint8) : 0 non-veg/noData ; 1..4 classes
      thresholds (tuple) : (p1,p2,p3)
    """
    index_arr = index_arr.astype(np.float32)
    veg_bool = (veg_mask_arr == 255)

    valid = veg_bool & np.isfinite(index_arr)
    vals = index_arr[valid]

    if vals.size < min_veg_pixels:
        raise RuntimeError(
            f"Pas assez de pixels végétation valides pour calculer les seuils: {vals.size} < {min_veg_pixels}"
        )

    p1, p2, p3 = np.percentile(vals, list(pcts))

    classes = np.zeros(index_arr.shape, dtype=np.uint8)

    if high_is_good:
        # faible -> stressé ; élevé -> bonne santé
        classes[veg_bool & (index_arr < p1)] = 1
        classes[veg_bool & (index_arr >= p1) & (index_arr < p2)] = 2
        classes[veg_bool & (index_arr >= p2) & (index_arr <= p3)] = 3
        classes[veg_bool & (index_arr > p3)] = 4
    else:
        # élevé -> stressé ; faible -> bonne santé (rare)
        classes[veg_bool & (index_arr > p3)] = 1
        classes[veg_bool & (index_arr > p2) & (index_arr <= p3)] = 2
        classes[veg_bool & (index_arr >= p1) & (index_arr <= p2)] = 3
        classes[veg_bool & (index_arr < p1)] = 4

    return classes, (float(p1), float(p2), float(p3))


def classes_to_rgb(classes: np.ndarray) -> np.ndarray:
    """
    Palette :
      0 -> noir (non-veg/noData)
      1 -> rouge foncé (très stressée)
      2 -> rouge (stress modéré)
      3 -> jaune (normal)
      4 -> vert (bonne santé)
    """
    rgb = np.zeros((classes.shape[0], classes.shape[1], 3), dtype=np.uint8)
    rgb[classes == 1] = [165, 0, 38]
    rgb[classes == 2] = [215, 48, 39]
    rgb[classes == 3] = [254, 224, 139]
    rgb[classes == 4] = [26, 150, 65]
    return rgb


# ======================
# READ
# ======================
with rasterio.open(INDEX_PATH) as src:
    index_img = src.read(1).astype(np.float32)
    profile = src.profile.copy()

with rasterio.open(VEG_MASK_PATH) as src:
    veg = src.read(1)

# (optionnel) vérif taille
if index_img.shape != veg.shape:
    raise ValueError("INDEX et VEG_MASK n'ont pas la même taille. Il faut les aligner/reprojeter avant.")

# ======================
# CLASSIFY
# ======================
classes, (t1, t2, t3) = classify_health_from_index(
    index_img,
    veg,
    pcts=PCTS,
    high_is_good=HIGH_IS_GOOD,
)

print(f"Seuils (percentiles {PCTS}) sur pixels végétation:")
print(f"  T1={t1:.6f}, T2={t2:.6f}, T3={t3:.6f}")

# ======================
# WRITE CLASSES (GeoTIFF)
# ======================
profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")

out_classes = os.path.join(OUTDIR, f"{PREFIX}_classes.tif")
with rasterio.open(out_classes, "w", **profile) as dst:
    dst.write(classes, 1)

# ======================
# WRITE RGB VISUAL
# ======================
rgb = classes_to_rgb(classes)
out_rgb = os.path.join(OUTDIR, f"{PREFIX}_classes_RGB.tif")
tifffile.imwrite(out_rgb, rgb, photometric="rgb", imagej=True)

print("✅ Fichiers créés :")
print(" -", out_classes)
print(" -", out_rgb)
