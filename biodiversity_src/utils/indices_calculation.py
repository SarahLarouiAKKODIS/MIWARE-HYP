import numpy as np
from utils.hyperspectral_utils import find_nearest_band



# ============================================================
# 4) INDICES
# ============================================================

def safe_divide(num, den, eps=1e-10):
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)

    shape = np.broadcast_shapes(num.shape, den.shape)
    out = np.full(shape, np.nan, dtype=np.float32)

    np.divide(
        num,
        den,
        out=out,
        where=np.abs(den) > eps
    )

    return out


def compute_spectral_indices(cube, wavelengths):

    print("MIN CUBE", np.nanmin(cube))
    print("MEAN CUBE", np.nanmean(cube))
    print("MAX CUBE", np.nanmax(cube))

    # Bandes
    i_blue = find_nearest_band(wavelengths, 490)
    i_green = find_nearest_band(wavelengths, 550)
    i_red = find_nearest_band(wavelengths, 670)
    i_rededge = find_nearest_band(wavelengths, 720)
    i_nir = find_nearest_band(wavelengths, 800)

    i_swir1 = find_nearest_band(wavelengths, 1240)  # eau / NDWI-NDMI
    i_swir2 = find_nearest_band(wavelengths, 2200)  # brûlure / NBR

    i_531 = find_nearest_band(wavelengths, 531)
    i_570 = find_nearest_band(wavelengths, 570)
    i_700 = find_nearest_band(wavelengths, 700)

    print("Bandes utilisées :")
    print(f"  Blue   -> {wavelengths[i_blue]:.2f} nm")
    print(f"  Green  -> {wavelengths[i_green]:.2f} nm")
    print(f"  Red    -> {wavelengths[i_red]:.2f} nm")
    print(f"  RedEdge-> {wavelengths[i_rededge]:.2f} nm")
    print(f"  NIR    -> {wavelengths[i_nir]:.2f} nm")
    print(f"  SWIR1  -> {wavelengths[i_swir1]:.2f} nm")
    print(f"  SWIR2  -> {wavelengths[i_swir2]:.2f} nm")
    print(f"  R531   -> {wavelengths[i_531]:.2f} nm")
    print(f"  R570   -> {wavelengths[i_570]:.2f} nm")
    print(f"  R700   -> {wavelengths[i_700]:.2f} nm")


    # Extraction
    blue = cube[i_blue].astype(np.float32)
    green = cube[i_green].astype(np.float32)
    red = cube[i_red].astype(np.float32)
    rededge = cube[i_rededge].astype(np.float32)
    nir = cube[i_nir].astype(np.float32)
    swir1 = cube[i_swir1].astype(np.float32)
    swir2 = cube[i_swir2].astype(np.float32)
    r531 = cube[i_531].astype(np.float32)
    r570 = cube[i_570].astype(np.float32)
    r700 = cube[i_700].astype(np.float32)



    # Indices
    ndvi = safe_divide(nir - red, nir + red)
    ndre = safe_divide(nir - rededge, nir + rededge)
    # Ici c'est plutôt NDMI / NDWI Gao, sensible à l'eau foliaire
    ndwi = safe_divide(nir - swir1, nir + swir1)
    pri = safe_divide(r531 - r570, r531 + r570)
    ari = safe_divide(1.0, np.maximum(green, 1e-6)) - \
        safe_divide(1.0, np.maximum(r700, 1e-6))
    evi = safe_divide(
        2.5 * (nir - red),
        nir + 6.0 * red - 7.5 * blue + 1.0
    )
    # NBR : utiliser SWIR2
    nbr = safe_divide(nir - swir2, nir + swir2)

    print("blue", np.nanmin(blue), np.nanmean(blue), np.nanmax(blue))
    print("red ", np.nanmin(red),  np.nanmean(red),  np.nanmax(red))
    print("nir ", np.nanmin(nir),  np.nanmean(nir),  np.nanmax(nir))
    print("r550 ", np.nanmin(green),  np.nanmean(green),  np.nanmax(green))
    print("r700 ", np.nanmin(r700),  np.nanmean(r700),  np.nanmax(r700))
    

    num = 2.5 * (nir - red)
    den = nir + 6.0 * red - 7.5 * blue + 1.0

    print("EVI numerator  ", np.nanmin(num), np.nanmean(num), np.nanmax(num))
    print("EVI denominator", np.nanmin(den), np.nanmean(den), np.nanmax(den))

    evi = safe_divide(num, den)

    print("EVI raw", np.nanmin(evi), np.nanmean(evi), np.nanmax(evi))


    return {
        "ndvi": ndvi,
        "ndre": ndre,
        "ndwi": ndwi,
        "pri": pri,
        "ari": ari,
        "evi": evi,
        "nbr": nbr
    }

