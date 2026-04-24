import geopandas as gpd
import pandas as pd

# Lire les deux shapefiles
gdf1 = gpd.read_file("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BDFORET_v2/Haute-Vienne/BDFORET_2-0__SHP_LAMB93_D087_2016-02-16/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D087/FORMATION_VEGETALE.shp")
gdf2 = gpd.read_file("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BDFORET_v2/Corrèze/BDFORET_2-0__SHP_LAMB93_D019_2016-02-16/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D019/FORMATION_VEGETALE.shp")

gdf_fusion_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BDFORET_v2/Haute-Vienne_Corrèze_fusion/FORMATION_VEGETALE.shp"
# Ajouter un champ de classe si besoin
gdf1["classe"] = 1
gdf2["classe"] = 2

# Vérifier que les projections sont identiques
gdf2 = gdf2.to_crs(gdf1.crs)

# Fusionner
gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True), crs=gdf1.crs)

# Sauvegarder
gdf.to_file(gdf_fusion_path)