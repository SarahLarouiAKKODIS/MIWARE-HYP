import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

# 1) Entrées
VECTOR_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BDFORET_2-0__SHP_LAMB93_D011_2018-11-20/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D011/FORMATION_VEGETALE.shp"   # ou .shp
LAYER_NAME  = None                # mets le nom si c'est un GPKG multi-couches
#REF_RASTER  = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegetation_indice_outputs/enmap_salsigne_MSAVI_veg.tiff"      # ton raster (orthophoto / S2 / EnMAP...)
REF_RASTER  = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"      # ton raster (orthophoto / S2 / EnMAP...)
OUT_MASK    = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Forest/mask_foret_classes.tif"

# 2) Choisis le champ qui porte la classe
CLASS_FIELD = "ESSENCE"  # <-- à adapter (ex: "TYPE", "ESSENCE", "CLASSE", etc.)

# 3) Mapping classe 
CLASS_MAP = {
    "Sapin, épicéa": 1,
    "Châtaignier": 2,
    "Chênes décidus": 3,
    "Chênes sempervirents" :4,
    "Conifères":5,
    "Douglas":6,
    "Feuillus":7,
    "Hêtre":8,
    "Mixte":9,
    "NC":10,
    "NR":11,
    "Peuplier":12,
    "Pin autre":13,
    "Pin d'Alep":14,
    "Pin laricio, pin noir":15,
    "Pin sylvestre":16,
    "Pins mélangés":17,
    "Robinier":18,
}

BACKGROUND = 0
NODATA = 255  # pratique si tu veux 8-bit avec une valeur nodata distincte

# ---- Charger la référence raster (grille cible)
with rasterio.open(REF_RASTER) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)
    ref_bounds = ref.bounds

# ---- Charger le vecteur
gdf = gpd.read_file(VECTOR_PATH, layer=LAYER_NAME) if LAYER_NAME else gpd.read_file(VECTOR_PATH)

# ---- Reprojeter vers CRS du raster
if gdf.crs != ref_crs:
    gdf = gdf.to_crs(ref_crs)

# ---- Optionnel: clip sur l'emprise du raster (accélère beaucoup)
# (on construit un polygon bbox)
from shapely.geometry import box
bbox = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
gdf = gdf[gdf.intersects(bbox)].copy()

# ---- Nettoyage: retirer géométries vides
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

# ---- Construire les "shapes" (geom, value)
# Ici on encode via CLASS_MAP. Les valeurs non reconnues => ignore ou code "AUTRE"
shapes = []
for geom, cls in zip(gdf.geometry, gdf[CLASS_FIELD].astype(str)):
    if cls in CLASS_MAP:
        shapes.append((geom, CLASS_MAP[cls]))

# ---- Rasteriser
mask = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=BACKGROUND,
    dtype=np.uint8,
    all_touched=False  # True si tu veux inclure tous pixels touchés (souvent pour routes)
)

# ---- Écrire GeoTIFF
# ---- Écrire GeoTIFF
profile = {
    "driver": "GTiff",
    "height": ref_shape[0],
    "width": ref_shape[1],
    "count": 1,
    "dtype": rasterio.uint8,
    "crs": ref_crs,
    "transform": ref_transform,
    "nodata": NODATA,
    "compress" : None,         # <-- IMPORTANT : pas de compression
    "tiled" : False,           # <-- IMPORTANT : pas de tiling
}

# Si tu veux nodata distinct, tu peux mettre les zones hors emprise à NODATA,
# mais ici on garde BACKGROUND=0 partout et nodata optionnel.
with rasterio.open(OUT_MASK, "w", **profile) as dst:
    dst.write(mask, 1)

print("Masque écrit :", OUT_MASK, "shape:", mask.shape, "classes:", np.unique(mask))