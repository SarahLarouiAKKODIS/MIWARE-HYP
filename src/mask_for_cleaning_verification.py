import numpy as np
import rasterio
import tifffile
import os

Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/"

cloud_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUD.TIF"
haze_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_HAZE.TIF"
cirrus_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CIRRUS.TIF"
cloudshadow_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUDSHADOW.TIF"
snow_mask = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_SNOW.TIF"
testflags = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_TESTFLAGS.TIF"

out_label_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/mask_for cleaning_verification.tif"
output_mask_croped = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/mask_for cleaning_verification_crop.tif"

# --- À ADAPTER ---
mask_files = [
    (cloud_mask, 1),     # (fichier, label)
    (haze_mask, 2),
    (cirrus_mask, 3),
    (cloudshadow_mask, 4),
    (snow_mask, 5),
    (testflags, 6),
]

MULTI_LABEL = 9
VALID_LABEL = 0
UNDEF_LABEL = 255  # au cas où (pixels hors emprise/no data dans les masques)

# --- Lecture du premier masque pour récupérer géométrie/profil ---
with rasterio.open(mask_files[0][0]) as src0:
    ref_profile = src0.profile.copy()
    ref_shape = (src0.height, src0.width)

# --- Initialisation ---
label_map = np.full(ref_shape, VALID_LABEL, dtype=np.uint8)  # défaut = valide
hit_count = np.zeros(ref_shape, dtype=np.uint8)              # combien de masques excluent ce pixel ?

# --- Lecture/accumulation ---
for mf, lab in mask_files:
    with rasterio.open(mf) as src:
        m = src.read(1)

    if m.shape != ref_shape:
        raise ValueError(f"Masque {mf} n'a pas la même taille que le masque de référence.")

    excluded = (m > 0) & np.isfinite(m)  # >0 = exclu (ajuste si besoin)

    # pixels exclus par ce masque
    hit_count[excluded] = np.clip(hit_count[excluded] + 1, 0, 255)

    # si c'est la première fois qu'on exclut ce pixel, on stocke l'origine
    first_time = excluded & (hit_count == 1)
    label_map[first_time] = lab

# --- Gérer les pixels exclus par plusieurs masques ---
label_map[hit_count > 1] = MULTI_LABEL

# (Optionnel) marquer les pixels "indéfinis" si tu as des NoData dans les masques
# Ici on ne peut le deviner que si tu as une convention; sinon tu peux ignorer.
# Exemple si tes masques ont un nodata explicite:
# label_map[~np.isfinite(m)] = UNDEF_LABEL

# --- Écriture GeoTIFF (1 bande, palette) ---
ref_profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=UNDEF_LABEL  # facultatif; ici UNDEF_LABEL sert de nodata
)

# Palette (RGBA) : adapte les couleurs comme tu veux
colormap = {
    VALID_LABEL: (0, 0, 0, 0),         # transparent
    1: (255, 255, 255, 255),           # cloud : blanc
    2: (255, 165, 0, 255),             # haze  : orange
    3: (0, 255, 255, 255),             # cirrus: cyan
    4: (255, 0, 255, 255),             # cloudshadow_mask: magenta
    5: (128, 0, 128, 255),             # snow_mask: violet foncé
    6: (255, 255, 0, 255),             # testflags: jaune vif

    MULTI_LABEL: (255, 0, 0, 255),     # multi: rouge
    UNDEF_LABEL: (128, 128, 128, 255), # undefined/nodata: gris
}


print("Valeurs présentes :", np.unique(label_map))


rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)

# même code couleur que ci-dessus
rgb[label_map == 1] = [255, 255, 255]   # cloud
rgb[label_map == 2] = [255, 165, 0]     # haze
rgb[label_map == 3] = [0, 255, 255]     # cirrus
rgb[label_map == 4] = [255, 0, 255]     # testflags
rgb[label_map == 9] = [255, 0, 0]       # multi
# valid reste noir (0,0,0)

tifffile.imwrite(out_label_tif, rgb, photometric="rgb", imagej=True)
print("✅ RGB écrit : excluded_source_labels_rgb.tiff")


x_min, y_min = 756, 646
x_max, y_max = 945, 780

cropped = rgb[y_min:y_max, x_min:x_max]

tifffile.imwrite(output_mask_croped, cropped, imagej=True)

print("✅ Masque croppé enregistré :", output_mask_croped)
print("Taille :", cropped.shape)


