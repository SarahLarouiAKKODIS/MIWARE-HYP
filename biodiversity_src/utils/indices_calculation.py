import numpy as np
from utils.hyperspectral_utils import find_nearest_band



# ============================================================
# 4) INDICES
# ============================================================

def safe_divide(num, den):
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)

    shape = np.broadcast_shapes(num.shape, den.shape)
    out = np.full(shape, np.nan, dtype=np.float32)

    np.divide(
        num,
        den,
        out=out,
        where=np.abs(den) > 1e-10
    )

    return out


def compute_spectral_indices(cube, wavelengths):
    i_green = find_nearest_band(wavelengths, 550)
    i_red = find_nearest_band(wavelengths, 670)
    i_rededge = find_nearest_band(wavelengths, 720)
    i_nir = find_nearest_band(wavelengths, 800)
    i_swir = find_nearest_band(wavelengths, 1240)
    i_531 = find_nearest_band(wavelengths, 531)
    i_570 = find_nearest_band(wavelengths, 570)
    i_700 = find_nearest_band(wavelengths, 700)

    print("Bandes utilisées :")
    print(f"  Green   -> {wavelengths[i_green]:.2f} nm")
    print(f"  Red     -> {wavelengths[i_red]:.2f} nm")
    print(f"  RedEdge -> {wavelengths[i_rededge]:.2f} nm")
    print(f"  NIR     -> {wavelengths[i_nir]:.2f} nm")
    print(f"  SWIR    -> {wavelengths[i_swir]:.2f} nm")
    print(f"  R531    -> {wavelengths[i_531]:.2f} nm")
    print(f"  R570    -> {wavelengths[i_570]:.2f} nm")
    print(f"  R700    -> {wavelengths[i_700]:.2f} nm")


    green = cube[i_green]
    red = cube[i_red]
    rededge = cube[i_rededge]
    nir = cube[i_nir]
    swir = cube[i_swir]
    r531 = cube[i_531]
    r570 = cube[i_570]
    r700 = cube[i_700]

    ndvi = safe_divide(nir - red, nir + red)
    ndre = safe_divide(nir - rededge, nir + rededge)
    ndwi = safe_divide(nir - swir, nir + swir)  # plutôt NDMI-like
    pri = safe_divide(r531 - r570, r531 + r570)
    ari = safe_divide(1, green) - safe_divide(1, r700)
    #ari2 = nir * (safe_divide(1, green) - safe_divide(1, r700))
    nbr = safe_divide(nir - swir, nir + swir)

    return {
        "ndvi": ndvi,
        "ndre": ndre,
        "ndwi": ndwi,
        "pri": pri,
        "ari": ari,
        "nbr": nbr
    }

