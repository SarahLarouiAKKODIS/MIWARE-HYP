import numpy as np
import rasterio

ref_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE_0/RGB_from_hyperspectral_crop.tif"

path_to_fix = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Spectral_mineral_detection/chalcopyrite/combo_sam_mf_chalcopyrite.tif"


# lire l'image "cassée" (valeurs OK, géoréf manquante)
with rasterio.open(path_to_fix) as bad:
    arr = bad.read(1)
    bad_nodata = bad.nodata

# lire la référence (géoréf OK)
with rasterio.open(ref_path) as ref:
    profile = ref.profile.copy()

profile.update(
    driver="GTiff",
    count=1,
    dtype=arr.dtype,
    nodata=bad_nodata,
    compress="deflate",
)

with rasterio.open(path_to_fix, "w", **profile) as dst:
    dst.write(arr, 1)

print("Écrit:", path_to_fix)