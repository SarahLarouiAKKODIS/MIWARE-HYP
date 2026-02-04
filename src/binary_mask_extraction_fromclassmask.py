import numpy as np
import rasterio

# =========================
# PARAMÈTRES À MODIFIER
# =========================
input_tiff = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLASSES.TIF"# image TIFF de labels
out_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"

output_tiff = out_path + "mask_enmap_allimage_water.tiff"    # masque de sortie

label_value = 2
# =========================

with rasterio.open(input_tiff) as src:
    # Lecture de la première bande
    labels = src.read(1)

    profile = src.profile

# Création du masque
mask = (labels == label_value).astype(np.uint8) * 255

# Mise à jour du profil pour le masque
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw"
)

# Écriture du masque
with rasterio.open(output_tiff, "w", **profile) as dst:
    dst.write(mask, 1)

print("Masque extrait avec rasterio ✔")
