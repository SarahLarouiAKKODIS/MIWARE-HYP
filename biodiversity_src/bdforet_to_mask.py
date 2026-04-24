import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pathlib import Path
from commun_functions import load_config


# CONFIG
config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/salsigne.json"
config = load_config(config_path)

Path_res = Path(config["Path_res"]) 
Path_res.mkdir(parents=True, exist_ok=True)
Path_res = Path_res / "Forest"
Path_res.mkdir(parents=True, exist_ok=True)
REF_RASTER = Path(config["image_hyp"])

# 1) Entrées
LAYER_NAME  = None # mets le nom si c'est un GPKG multi-couches
OUT_MASK    = Path_res / "mask_foret_classes.tif"
VECTOR_PATH = config["forest_species_mask"]


# 3) Mapping classe 
## CLASSES DES ESSENCES ET DES ETAT DE STRESSE   
config_forest_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/forest_classes.json"
config_forest = load_config(config_forest_path)

# --- 1) Classes d'essences ---
CLASS_MAP = config_forest["CLASS_MAP"]                 # nom -> id

# 2) Choisis le champ qui porte la classe
CLASS_FIELD = config_forest["CLASS_FIELD"] # <-- à adapter (ex: "TYPE", "ESSENCE", "CLASSE", etc.)

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
    all_touched=False  # True si tu veux inclure tous pixels touchés 
)

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