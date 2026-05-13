
import json
import rasterio


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)