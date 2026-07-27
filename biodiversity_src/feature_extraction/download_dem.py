from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import requests
from utils import load_config
import rasterio
from rasterio.warp import transform_bounds

# Remplacer par votre clé API OpenTopography
API_KEY = "c7b4a4fc64d1dd33fc8b0efbf4a6bf1b"


# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":

    config_site_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/cma.json"
    config_site = load_config(config_site_path)

    Path_res = Path(config_site["Path_res"])
    image_hyperspectrale_cleanbands = Path_res / "image_hyperspectrale_cleanbands.tif"

    out_dataset_dir = Path_res / "Copernicus_dem_glo30"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_hyperspectrale_cleanbands) as src:
        # Bounding box dans le système de projection de l'image
        bounds = src.bounds
        crs = src.crs

        print("CRS image :", crs)
        print("Bounds image :", bounds)

        # Conversion en coordonnées GPS WGS84 : longitude/latitude
        west, south, east, north = transform_bounds(
            crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top
        )


        url = (
            "https://portal.opentopography.org/API/globaldem"
            f"?demtype=COP30"
            f"&west={west}"
            f"&south={south}"
            f"&east={east}"
            f"&north={north}"
            f"&outputFormat=GTiff"
            f"&API_Key={API_KEY}"
        )

        response = requests.get(url)
        response.raise_for_status()

        with open(out_dataset_dir / "copernicus_dem_glo30.tif", "wb") as f:
            f.write(response.content)

        print("Téléchargement terminé !")