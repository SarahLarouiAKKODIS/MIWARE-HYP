import os
import numpy as np
import csv
import tifffile

TOL_NM_FIXED = 12.0

NODATA_F32 = -9999.0

print("utils.py loaded, np =", np.__name__)

import numpy as np

def read_scale_and_clip_bands(
    src,
    bands: dict,
    scale: float,
    min_val: float,
    max_val: float,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Lit plusieurs bandes raster, applique le scale factor uniquement
    sur les pixels valides (non-NaN), puis seuillage -> NaN.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Dataset ouvert avec rasterio.open(...)
    bands : dict
        Dictionnaire {nom_bande: index_bande} (index 1-based)
        ex: {"RED": 12, "GREEN": 8, "NIR": 34}
    scale : float
        Facteur d'échelle (ex: 10000.0)
    min_val, max_val : float
        Bornes physiques après scaling (ex: 0.0, 1.2)
    verbose : bool
        Si True, affiche les stats par bande

    Returns
    -------
    out : dict[str, np.ndarray]
        Dictionnaire {nom_bande: tableau 2D float32 avec NaN}
    """

    out = {}

    for name, band_index in bands.items():

        # 1) lecture
        band = src.read(band_index).astype(np.float32)

        if verbose:
            print(f"\n{name}")
            print("  nanmin/nanmax (avant scale):",
                  np.nanmin(band), np.nanmax(band))

        # 2) masque validité
        valid = np.isfinite(band)

        # 3) scale uniquement sur pixels valides
        band[valid] /= float(scale)

        if verbose:
            print("  nanmin/nanmax (après scale):",
                  np.nanmin(band), np.nanmax(band))

        # 4) seuillage physique
        band[(band < min_val) | (band > max_val)] = np.nan

        if verbose:
            finite = np.isfinite(band)
            print("  nanmin/nanmax (après seuil):",
                  np.nanmin(band), np.nanmax(band))
            print("  pixels valides:",
                  finite.sum(), "/", finite.size)

        out[name] = band

    return out

# def read_scale_and_clip_band(
#     src,
#     band_index: int,
#     scale: float,
#     min_val: float,
#     max_val: float,
#     name: str = "BAND",
#     verbose: bool = True,
# ) -> np.ndarray:
#     """
#     Lit une bande raster (rasterio.DatasetReader), applique le scale factor
#     uniquement sur les pixels valides (non-NaN), puis seuillage -> NaN.

#     Parameters
#     ----------
#     src : rasterio.DatasetReader
#         Dataset ouvert avec rasterio.open(...)
#     band_index : int
#         Index de bande (1-based) pour src.read(...)
#     scale : float
#         Facteur d'échelle (ex: 10000.0)
#     min_val, max_val : float
#         Bornes physiques après scaling (ex: 0.0, 1.2)
#     name : str
#         Nom affiché pour les prints
#     verbose : bool
#         Si True, affiche les stats

#     Returns
#     -------
#     band : np.ndarray (float32)
#         Tableau 2D avec NaN sur pixels invalides
#     """
#     # 1) lecture
#     band = src.read(band_index).astype(np.float32)

#     if verbose:
#         print(f"{name} nanmin/nanmax (avant scale):", np.nanmin(band), np.nanmax(band))

#     # 2) masque validité
#     valid = np.isfinite(band)

#     # 3) scale uniquement sur valides
#     band[valid] /= float(scale)

#     if verbose:
#         print(f"{name} nanmin/nanmax (après scale):", np.nanmin(band), np.nanmax(band))

#     # 4) seuillage (après scaling)
#     band[(band < min_val) | (band > max_val)] = np.nan

#     if verbose:
#         finite = np.isfinite(band)
#         print(f"{name} nanmin/nanmax (après seuil):", np.nanmin(band), np.nanmax(band))
#         print(f"{name} pixels valides:", finite.sum(), "/", finite.size)

#     return band


def load_wavelengths_from_csv(csv_path: str) -> np.ndarray:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "wavelength_nm" not in reader.fieldnames:
            raise ValueError("Colonne 'wavelength_nm' introuvable dans le CSV.")
        if "band_id" not in reader.fieldnames:
            raise ValueError("Colonne 'band_id' introuvable dans le CSV.")
        for r in reader:
            rows.append((int(r["band_id"]), float(r["wavelength_nm"])))
    if not rows:
        raise ValueError("CSV vide ou illisible.")
    rows.sort(key=lambda x: x[0])
    return np.array([r[1] for r in rows], dtype=np.float32)


def compute_auto_tol_nm(wv_nm: np.ndarray) -> float:
    wv_sorted = np.sort(wv_nm)
    diffs = np.diff(wv_sorted)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return TOL_NM_FIXED
    step = float(np.median(diffs))
    return max(8.0, 1.5 * step)


def nearest_band_index(wavelengths_nm: np.ndarray, target_nm: float, tol_nm: float) -> int:
    diffs = np.abs(wavelengths_nm - target_nm)
    idx0 = int(np.argmin(diffs))
    if float(diffs[idx0]) > tol_nm:
        raise ValueError(
            f"Aucune bande dans la tolérance: cible={target_nm} nm, "
            f"plus proche={float(wavelengths_nm[idx0])} nm (écart {float(diffs[idx0]):.1f} nm) "
            f"> tol {tol_nm:.1f} nm."
        )
    return idx0


def safe_norm_diff(a: np.ndarray, b: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full(a.shape, NODATA_F32, dtype=np.float32)
    valid = (~invalid) & np.isfinite(denom) & (denom != 0)
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out



def write_imagej_tiff(path: str, arr: np.ndarray, dtype: str = "float32", nodata=None):
    arr = np.asarray(arr)
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    is_int = np.issubdtype(np.dtype(dtype), np.integer)

    if is_float:
        arr = arr.astype(dtype, copy=False)
    elif is_int:
        if nodata is None:
            raise ValueError("Pour un dtype entier, nodata doit être défini (ex: -1).")
        arr = np.where(np.isnan(arr), nodata, arr).astype(dtype, copy=False)
    else:
        raise TypeError("dtype non supporté")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    tifffile.imwrite(path, arr, photometric="minisblack", imagej=True)


def msavi(nir: np.ndarray, red: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    out = np.full(nir.shape, NODATA_F32, dtype=np.float32)
    term = (2.0 * nir + 1.0)
    discr = term**2 - 8.0 * (nir - red)
    valid = (~invalid) & (discr >= 0)
    out[valid] = (term[valid] - np.sqrt(discr[valid])) / 2.0
    return out


def read_band(src, band_1based):
    arr = src.read(band_1based).astype(np.float32)
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr 


def band_depth(RL, RC, RR, lamL, lamC, lamR):
    # continuum au centre
    Rcont = RL + ((lamC - lamL) / (lamR - lamL)) * (RR - RL)

    out = np.full(RC.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(RL) & np.isfinite(RC) & np.isfinite(RR) & np.isfinite(Rcont) & (Rcont > 0)

    out[valid] = (1.0 - (RC[valid] / Rcont[valid])).astype(np.float32)
    return out