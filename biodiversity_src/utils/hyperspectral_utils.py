import numpy as np
import rasterio
import pandas as pd

# ============================================================
# 2) LECTURE DES DONNÉES
# ============================================================

def read_hyperspectral_raster(image_path, scale_factor=10000.0, nodata_value=-32768):
    with rasterio.open(image_path) as src:
        cube = src.read().astype(np.float32)
        profile = src.profile

    cube[cube == nodata_value] = np.nan
    cube /= scale_factor
    return cube, profile


def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


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