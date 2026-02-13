from __future__ import annotations

from pathlib import Path
import rasterio
import numpy as np


def mask_enmap_hyperspectral_cube(
    cube_tif: str | Path,
    mask_files: list[str | Path],
    out_tif: str | Path,
    *,
    nodata_value: float = np.nan,
    dtype: str = "float32"
) -> None:
    """
    Applique plusieurs masques de qualité EnMAP à un cube hyperspectral.

    Tous les pixels où AU MOINS un masque > 0 sont mis à nodata (NaN par défaut).

    Parameters
    ----------
    cube_tif : str | Path
        Chemin vers le cube hyperspectral EnMAP (*-SPECTRAL_IMAGE.TIF).
    mask_files : list[str | Path]
        Liste des fichiers de masques qualité (*-QL_QUALITY_*.TIF).
    out_tif : str | Path
        Chemin de sortie du cube masqué.
    nodata_value : float, default np.nan
        Valeur nodata à appliquer aux pixels masqués.
    dtype : str, default "float32"
        Type de données du cube de sortie.
    """
    cube_tif = Path(cube_tif)
    out_tif = Path(out_tif)
    mask_files = [Path(m) for m in mask_files]

    # --- Lire le cube hyperspectral ---
    with rasterio.open(cube_tif) as src:
        cube = src.read().astype(dtype)  # (bands, y, x)
        profile = src.profile.copy()

    # --- Lire les masques ---
    masks = []
    for mf in mask_files:
        with rasterio.open(mf) as src:
            masks.append(src.read(1))

    # --- Combiner les masques ---
    # True = pixel à exclure
    combined_mask = np.any([m > 0 for m in masks], axis=0)

    cube[:, combined_mask] = nodata_value

    # --- Mettre à jour le profil ---
    profile.update(
        dtype=dtype,
        nodata=nodata_value
    )

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # --- Écriture du cube masqué ---
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(cube)
