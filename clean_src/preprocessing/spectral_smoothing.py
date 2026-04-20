import numpy as np
import rasterio
from scipy.signal import savgol_filter
from pathlib import Path


def savgol_smooth_and_normalize(
    img_path: str | Path,
    output_path: str | Path,
    window_length: int = 9,   # doit être impair
    polyorder: int = 2,
    normalize: str | None = None,   # None, "l2", "max"
    eps: float = 1e-12,
    dtype="float32",
    nodata_out: float | None = None,   # ex: -9999.0 ou None pour garder NaN
    verbose: bool = True,
):
    """
    Lissage spectral Savitzky–Golay + normalisation optionnelle par pixel.

    Parameters
    ----------
    normalize : None | "l2" | "max"
        None = seulement lissage (recommandé pour band depth / minéraux)
        "l2" = normalisation L2 (pour SAM/MF)
        "max" = normalisation par max
    nodata_out : float | None
        Si None → conserve NaN
        Sinon → remplace NaN par cette valeur dans le GeoTIFF
    """

    img_path = Path(img_path)
    output_path = Path(output_path)

    if window_length % 2 == 0:
        raise ValueError("window_length doit être impair (ex: 7, 9, 11).")

    with rasterio.open(img_path) as src:
        img = src.read().astype(dtype)  # (bands, rows, cols)
        profile = src.profile.copy()

    bands, rows, cols = img.shape

    if window_length > bands:
        raise ValueError(f"window_length={window_length} > nb_bandes={bands}.")

    # Pixels valides = pas de NaN sur toutes les bandes
    valid = ~np.isnan(img).any(axis=0)

    # --- Lissage Savitzky–Golay ---
    img_filled = np.nan_to_num(img, nan=0.0)

    img_smooth = savgol_filter(
        img_filled,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    ).astype(dtype)

    # remettre NaN sur pixels invalides
    img_smooth[:, ~valid] = np.nan

    # --- Normalisation ---
    if normalize is None:
        img_out = img_smooth

    elif normalize.lower() == "l2":
        norm = np.sqrt(np.nansum(img_smooth ** 2, axis=0)).astype(dtype)
        norm = np.where(norm < eps, np.nan, norm)
        img_out = img_smooth / norm

    elif normalize.lower() == "max":
        mx = np.nanmax(np.abs(img_smooth), axis=0).astype(dtype)
        mx = np.where(mx < eps, np.nan, mx)
        img_out = img_smooth / mx

    else:
        raise ValueError("normalize doit être None, 'l2' ou 'max'.")

    # --- Gestion nodata ---
    if nodata_out is not None:
        img_out = np.where(np.isnan(img_out), nodata_out, img_out)
        profile.update(nodata=nodata_out)
    else:
        profile.pop("nodata", None)  # évite nodata=np.nan

    profile.update(dtype=dtype, count=bands)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(img_out.astype(dtype))

    if verbose:
        print("✔ Savitzky–Golay terminé")
        print(f"Entrée : {img_path}")
        print(f"Sortie : {output_path}")
        print(f"Bandes : {bands}")
        print(f"window_length={window_length}, polyorder={polyorder}")
        print(f"normalize={normalize}")