from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils import load_config
import rasterio
from rasterio.warp import transform_bounds
from feature_extraction.environmental_feature_extraction import extract_topographic_features, extract_chelsa_features, extract_climate_features

# Remplacer par votre clé API OpenTopography
API_KEY = "c7b4a4fc64d1dd33fc8b0efbf4a6bf1b"

# ============================================================
# 12) MAIN
# ============================================================
# "salsigne_3.json", "salsigne_mars_2.json", "salsigne_mars_3.json","salsigne.json","salau_1.json","salau_2.json", "cma_1.json", #6
#    "cma.json", "abbaretz.json","abbaretz_3.json" 
if __name__ == "__main__":

    config_site_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/abbaretz_3.json"
    config_site = load_config(config_site_path)

    Path_res = Path(config_site["Path_res"])
    image_hyperspectrale_cleanbands = Path_res / "image_hyperspectrale_cleanbands.tif"

    topo_cube, topo_feature_names = extract_topographic_features(
        image_tif=image_hyperspectrale_cleanbands,
        path_res=Path_res,
        api_key=API_KEY,
        output_tif=Path_res / "Feature_extraction/topographic_features.tif",
        tpi_window_size=9,
    )

    # climate_cube, climate_feature_names = extract_chelsa_features(
    #     image_tif=image_hyperspectrale_cleanbands,
    #     path_res=Path_res,
    #     output_tif=Path_res / "Feature_extraction" / "chelsa_features.tif",
    # )

    climate_cube, climate_feature_names = extract_climate_features(
        image_tif=image_hyperspectrale_cleanbands,
        path_res=Path_res.parents[1] / "Data",
        end_year=2025,
        climate_window=25,
        output_tif=Path_res / "Feature_extraction" / "climate_features_2000_2024.tif",
    )