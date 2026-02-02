import numpy as np
import rasterio
import tifffile
import os

# ============================================================
# Script combiné WDI x VII (GNDVI) sur pixels végétation
# - Utilise des seuils percentiles (10/30/70) calculés sur la végétation
# - Produit :
#   1) 2 cartes de classes (WDI et VII) (uint8)
#   2) 1 carte combinée (uint8) basée sur la logique croisée
#   3) 1 RGB de la carte combinée
#
# Classes WDI / VII :
#   0 = non-veg / NoData
#   1 = très stressé (bas)
#   2 = stress modéré
#   3 = normal
#   4 = bon / haut
#
# Carte combinée (0..4) :
#   0 = non-veg / NoData
#   1 = Stress combiné (hydrique + chlorophylle) : WDI bas ET VII bas
#   2 = Stress hydrique dominant : WDI bas ET VII normal/haut
#   3 = Stress chlorophyllien dominant : VII bas ET WDI normal/haut
#   4 = Bonne santé : WDI normal/haut ET VII normal/haut
# ============================================================

# ======================
# PATHS (à adapter)
# ======================
WDI_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_WDI_decorrelated.tiff"
VII_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VII_GNDVI.tiff"
VEG_MASK_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VEG_MASK.tiff"  # 255 = veg

OUTDIR = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegetation_health"
PREFIX = "veg_health_WDIxVII"

os.makedirs(OUTDIR, exist_ok=True)

# Percentiles
PCTS = (10, 30, 70)

# Combinaison : on définit "bas" = classe 1 ou 2 (<= T2), "normal/haut" = classe 3 ou 4 (> T2)
LOW_CUTOFF_CLASS = 2  # <=2 => "low/stressed", >=3 => "normal/high"


def classify_4classes(arr: np.ndarray, veg: np.ndarray, pcts=(10, 30, 70), min_pixels=100):
    arr = arr.astype(np.float32)
    veg_bool = (veg == 255)
    valid = veg_bool & np.isfinite(arr)
    vals = arr[valid]
    if vals.size < min_pixels:
        raise RuntimeError(f"Pas assez de pixels végétation valides: {vals.size} < {min_pixels}")

    t1, t2, t3 = np.percentile(vals, list(pcts))
    classes = np.zeros(arr.shape, dtype=np.uint8)

    # faible -> stress ; élevé -> bon
    classes[veg_bool & (arr < t1)] = 1
    classes[veg_bool & (arr >= t1) & (arr < t2)] = 2
    classes[veg_bool & (arr >= t2) & (arr <= t3)] = 3
    classes[veg_bool & (arr > t3)] = 4

    return classes, (float(t1), float(t2), float(t3))


def write_u8_geotiff(path, arr_u8, profile):
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr_u8, 1)


# ======================
# READ
# ======================
with rasterio.open(WDI_PATH) as src:
    wdi = src.read(1).astype(np.float32)
    profile = src.profile.copy()

with rasterio.open(VII_PATH) as src:
    vii = src.read(1).astype(np.float32)

with rasterio.open(VEG_MASK_PATH) as src:
    veg = src.read(1)

if wdi.shape != vii.shape or wdi.shape != veg.shape:
    raise ValueError("WDI, VII et VEG_MASK doivent avoir la même shape. Aligne/reprojette avant.")

# ======================
# CLASSIFY INDIVIDUAL
# ======================
wdi_cls, wdi_thr = classify_4classes(wdi, veg, pcts=PCTS)
vii_cls, vii_thr = classify_4classes(vii, veg, pcts=PCTS)

print("Seuils WDI (veg only): T1=%.6f T2=%.6f T3=%.6f" % wdi_thr)
print("Seuils VII (veg only): T1=%.6f T2=%.6f T3=%.6f" % vii_thr)

# ======================
# COMBINE LOGIC
# ======================
# low = classes 1-2 ; normal/high = classes 3-4
veg_bool = (veg == 255) & np.isfinite(wdi) & np.isfinite(vii)

wdi_low = (wdi_cls > 0) & (wdi_cls <= LOW_CUTOFF_CLASS)
vii_low = (vii_cls > 0) & (vii_cls <= LOW_CUTOFF_CLASS)

wdi_ok = (wdi_cls >= 3)
vii_ok = (vii_cls >= 3)

combined = np.zeros(wdi.shape, dtype=np.uint8)

# 1) Stress combiné (hydrique + chlorophylle)
combined[veg_bool & wdi_low & vii_low] = 1

# 2) Stress hydrique dominant (WDI bas, VII ok)
combined[veg_bool & wdi_low & vii_ok] = 2

# 3) Stress chlorophyllien dominant (VII bas, WDI ok)
combined[veg_bool & wdi_ok & vii_low] = 3

# 4) Bonne santé (WDI ok, VII ok)
combined[veg_bool & wdi_ok & vii_ok] = 4

# (option) Cas "intermédiaires" : ex. un indice normal (classe 3) et l'autre modéré (classe 2)
# Ici, ils tomberont dans 2 ou 3 si l'un est "low" (<=2) et l'autre "ok"(>=3).
# S'ils sont tous les deux en classe 3, c'est "Bonne santé" (4).

# ======================
# WRITE OUTPUTS
# ======================
out_wdi_cls = os.path.join(OUTDIR, f"{PREFIX}_WDI_classes.tif")
out_vii_cls = os.path.join(OUTDIR, f"{PREFIX}_VII_classes.tif")
out_combined = os.path.join(OUTDIR, f"{PREFIX}_combined_classes.tif")

write_u8_geotiff(out_wdi_cls, wdi_cls, profile)
write_u8_geotiff(out_vii_cls, vii_cls, profile)
write_u8_geotiff(out_combined, combined, profile)

# ======================
# RGB VISUAL (combined)
# ======================
rgb = np.zeros((combined.shape[0], combined.shape[1], 3), dtype=np.uint8)
# 0 = noir
rgb[combined == 1] = [165, 0, 38]     # rouge foncé : stress combiné
rgb[combined == 2] = [215, 48, 39]    # rouge : stress hydrique
rgb[combined == 3] = [253, 174, 97]   # orange : stress chlorophyllien
rgb[combined == 4] = [26, 150, 65]    # vert : bonne santé

out_rgb = os.path.join(OUTDIR, f"{PREFIX}_combined_RGB.tif")
tifffile.imwrite(out_rgb, rgb, photometric="rgb", imagej=True)

# ======================
# STATS
# ======================
def pct(n): 
    return 100.0 * n / max(1, int(np.sum(veg == 255)))

print("✅ Fichiers créés :")
print(" -", out_wdi_cls)
print(" -", out_vii_cls)
print(" -", out_combined)
print(" -", out_rgb)

print("\nStats (sur pixels veg==255, sans filtrer NaN):")
for k, name in [(1, "Stress combiné"), (2, "Stress hydrique"), (3, "Stress chlorophyllien"), (4, "Bonne santé")]:
    n = int(np.sum(combined == k))
    print(f"  {name:22s}: {n:10d} pixels  ({pct(n):6.2f}%)")
