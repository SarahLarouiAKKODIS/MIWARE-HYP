import numpy as np
import rasterio
from scipy.signal import savgol_filter

def savgol_smooth_and_normalize(
    img_path: str,
    output_path: str,
    window_length: int = 9,   # doit être impair
    polyorder: int = 2,
    normalize: str = "l2",    # "l2", "max", ou None
    eps: float = 1e-12,
    dtype="float32"
):
    """
    Lissage spectral Savitzky–Golay + normalisation par pixel.

    Hypothèse : les pixels invalides/eau/végétation sont NaN sur toutes les bandes
               (comme après apply_water_veg_mask).
    """

    if window_length % 2 == 0:
        raise ValueError("window_length doit être impair (ex: 7, 9, 11).")

    with rasterio.open(img_path) as src:
        img = src.read().astype(dtype)  # (bands, rows, cols)
        profile = src.profile

    bands, rows, cols = img.shape

    if window_length > bands:
        raise ValueError(f"window_length={window_length} > nb_bandes={bands}. Réduis window_length.")

    # Pixels valides = pas de NaN sur le spectre
    valid = ~np.isnan(img).any(axis=0)  # (rows, cols)

    # --- Lissage Savitzky–Golay (axis=0 = spectral) ---
    # Remplissage des NaN (pixels invalides) à 0 pour ne pas planter (on remet NaN ensuite)
    img_filled = np.nan_to_num(img, nan=0.0)

    # Appliquer le filtre sur toute l'image (rapide), puis ré-appliquer NaN sur pixels invalides
    img_smooth = savgol_filter(
        img_filled,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    ).astype(dtype)

    # Remettre NaN pour les pixels invalides
    img_smooth[:, ~valid] = np.nan

    # --- Normalisation ---
    if normalize is None:
        img_out = img_smooth

    elif normalize.lower() == "l2":
        # norme L2 par pixel
        # shape: (rows, cols)
        norm = np.sqrt(np.nansum(img_smooth ** 2, axis=0)).astype(dtype)
        norm = np.where(norm < eps, np.nan, norm)
        img_out = img_smooth / norm

    elif normalize.lower() == "max":
        # normalisation par max (par pixel)
        mx = np.nanmax(np.abs(img_smooth), axis=0).astype(dtype)
        mx = np.where(mx < eps, np.nan, mx)
        img_out = img_smooth / mx

    else:
        raise ValueError("normalize doit être 'l2', 'max' ou None.")

    profile.update(dtype=dtype, nodata=np.nan, count=img_out.shape[0])

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(img_out.astype(dtype))

    print("Lissage + normalisation terminés ✔")
    print(f"Entrée : {img_path}")
    print(f"Sortie : {output_path}")
    print(f"Bandes : {bands}, window_length={window_length}, polyorder={polyorder}, normalize={normalize}")

