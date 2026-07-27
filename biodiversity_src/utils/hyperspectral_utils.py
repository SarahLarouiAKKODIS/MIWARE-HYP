import numpy as np
import rasterio
import pandas as pd
import json

# ============================================================
# 2) LECTURE DES DONNÉES
# ============================================================

def read_hyperspectral_raster(
    image_path,
    scale_factor=None,
    nodata_value=-32768
):
    with rasterio.open(image_path) as src:
        cube = src.read().astype(np.float32)
        profile = src.profile

    cube[cube == nodata_value] = np.nan

    if scale_factor is not None:
        cube /= scale_factor

    return cube, profile


def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)
    

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def read_wavelengths_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "wavelength_nm" not in df.columns:
        raise ValueError("La colonne 'wavelength_nm' est absente du CSV")
    return df["wavelength_nm"].to_numpy(dtype=np.float32)


def save_mask(output_path, mask, reference_profile, nodata=0):
    profile = reference_profile.copy()
    profile.update(count=1, dtype=rasterio.uint8, nodata=nodata)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


# ============================================================
# 3) MASQUE DE VALIDITÉ ENMAP
# ============================================================

def build_enmap_valid_mask(
    cirrus_mask,
    cloud_mask,
    haze_mask,
    cloudshadow_mask,
    snow_mask,
    testflags_mask,
    scene_mask,
):
    return (
        (scene_mask == 1) &
        (cirrus_mask == 0) &
        (cloud_mask == 0) &
        (haze_mask == 0) &
        (cloudshadow_mask == 0) &
        (snow_mask == 0) &
        (testflags_mask == 0)
    )


def find_nearest_band(wavelengths, target_nm):
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def save_valid_mask(valid_mask, output_path, profile):
    """
    Sauvegarde un masque booléen en GeoTIFF.

    Parameters
    ----------
    valid_mask : ndarray (bool)
        True = pixel valide
        False = pixel invalide
    output_path : str ou Path
        Chemin du GeoTIFF de sortie.
    profile : dict
        Profil rasterio de l'image de référence.
    """

    profile_out = profile.copy()
    profile_out.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile_out) as dst:
        dst.write(valid_mask.astype(np.uint8), 1)

    print(f"Masque sauvegardé : {output_path}")


from pathlib import Path
from datetime import datetime
import re
import xml.etree.ElementTree as ET


def get_enmap_acquisition_date(xml_path):
    xml_path = Path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Recherche de toutes les balises <name>
    for elem in root.iter("name"):
        text = elem.text

        if text and "ENMAP" in text and "METADATA.XML" in text:
            match = re.search(r"_(\d{8}T\d{6}Z)_", text)

            if match:
                date_str = match.group(1)
                return datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")

    raise ValueError("Date d'acquisition EnMAP introuvable dans le XML.")