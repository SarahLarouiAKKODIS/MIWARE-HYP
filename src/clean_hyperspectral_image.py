
import rasterio
import numpy as np

Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/"

cube_tif = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"
cloud_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUD.TIF"
haze_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_HAZE.TIF"
cirrus_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CIRRUS.TIF"
cloudshadow_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUDSHADOW.TIF"
snow_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_SNOW.TIF"
testflags = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_TESTFLAGS.TIF"

cube_masked = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean.tif"

# --- Lire le cube hyperspectral ---
with rasterio.open(cube_tif) as src:
    cube = src.read().astype("float32")  # (bands, y, x)
    profile = src.profile

# --- Lire les masques ---
mask_files = [
    cloud_mask,
    haze_mask,
    cirrus_mask,
    cloudshadow_mask,
    snow_mask,
    testflags
]

masks = []
for mf in mask_files:
    with rasterio.open(mf) as src:
        masks.append(src.read(1))

# --- Combiner les masques ---
# True = pixel à exclure
combined_mask = np.any([m > 0 for m in masks], axis=0)
cube[:, combined_mask] = np.nan

# --- Mettre à jour le profil ---
profile.update(
    dtype="float32",
    nodata=np.nan
)

# --- Écriture du cube masqué ---
with rasterio.open(cube_masked, "w", **profile) as dst:
    dst.write(cube)
