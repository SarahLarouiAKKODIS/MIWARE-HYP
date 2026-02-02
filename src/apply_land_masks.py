import numpy as np
import rasterio

# ======================
# PATHS
# ======================
veg_mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VEG_MASK.tiff"
water_mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"

mineral = "micas"
Path_mineral = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/" + mineral
mineral_mask_path = Path_mineral + "/" + mineral + "_mask.tif"
mineral_score_path = Path_mineral + "/" + mineral + "_probability.tif"

out_mask_path = Path_mineral + "/" + mineral + "_mask_land_only.tif"
out_score_path = Path_mineral + "/" + mineral + "_probability_land_only.tif"

# ======================
# READ
# ======================
with rasterio.open(mineral_mask_path) as src:
    mineral_mask = src.read(1)
    profile_mask = src.profile.copy()

with rasterio.open(mineral_score_path) as src:
    mineral_score = src.read(1)
    profile_score = src.profile.copy()

with rasterio.open(veg_mask_path) as src:
    veg_mask = src.read(1)

with rasterio.open(water_mask_path) as src:
    water_mask = src.read(1)

# (optionnel mais recommandé) vérif tailles
if mineral_mask.shape != mineral_score.shape or mineral_mask.shape != veg_mask.shape or mineral_mask.shape != water_mask.shape:
    raise ValueError("Les rasters n'ont pas la même taille (shape). Il faut les aligner/reprojeter avant.")

# ======================
# EXCLUDE = veg ou eau
# ======================
exclude = (veg_mask > 0) | (water_mask > 0)

# ======================
# MASK (binaire)
# ======================
mineral_land_only = np.where(exclude, 0, (mineral_mask > 0).astype(np.uint8) * 255).astype(np.uint8)

profile_mask.update(dtype="uint8", count=1, nodata=0, compress="lzw")

with rasterio.open(out_mask_path, "w", **profile_mask) as dst:
    dst.write(mineral_land_only, 1)

# ======================
# SCORE / PROBABILITY (float, conserve les valeurs)
# ======================
# Ici on garde le score tel quel, et on met 0 dans les pixels exclus.
# Si tu préfères NaN pour les exclus, remplace 0.0 par np.nan et mets nodata=np.nan.
mineral_score = mineral_score.astype(np.float32)

mineral_score_land_only = np.where(exclude, 0.0, mineral_score).astype(np.float32)
profile_score.update(dtype="float32", count=1, nodata=0.0, compress="lzw")

# # Si on veut que les zones exclues soient clairement “NoData” :
# mineral_score_land_only = np.where(exclude, np.nan, mineral_score).astype(np.float32)
# profile_score.update(nodata=np.nan)


with rasterio.open(out_score_path, "w", **profile_score) as dst:
    dst.write(mineral_score_land_only, 1)

print("✅ Masque final écrit :", out_mask_path)
print("✅ Image score final écrit :", out_score_path)
print("Score (min/mean/max) sur pixels non exclus:",
      float(np.nanmin(mineral_score_land_only[~exclude])),
      float(np.nanmean(mineral_score_land_only[~exclude])),
      float(np.nanmax(mineral_score_land_only[~exclude])))
