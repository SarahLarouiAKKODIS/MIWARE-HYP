from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError as e:
    raise ImportError(
        "Il faut installer rasterio : pip install rasterio"
    ) from e


@dataclass
class HyperCube:
    cube: np.ndarray                  # shape: (bands, height, width)
    crs: object                       # rasterio CRS
    transform: object                 # affine transform
    nodata: Optional[float]
    band_names: List[str]             # ex: ["file1_b1", "file2_b1", ...]


def tiffs_to_hyperspectral_cube(
    paths: Sequence[str],
    *,
    reference: Union[str, int] = 0,
    out_dtype: Optional[np.dtype] = None,
    allow_resample: bool = False,
    resampling: str = "bilinear",
    fill_value: Optional[float] = None,
    return_format: str = "bands_first",  # "bands_first" or "bands_last"
) -> HyperCube:
    """
    Empile une liste de TIFF/GeoTIFF en cube hyperspectral.

    Parameters
    ----------
    paths : list[str]
        Chemins vers les TIFF. Chaque TIFF peut contenir 1+ bandes.
    reference : str|int
        Chemin (ou index dans paths) du raster de référence définissant la grille.
    out_dtype : np.dtype|None
        Type de sortie. Si None, on conserve le dtype du raster de référence.
    allow_resample : bool
        Si False : erreur si tailles/CRS/transform différents.
        Si True : reprojection/rééchantillonnage vers la grille de référence.
    resampling : str
        "nearest", "bilinear", "cubic", "average", etc.
    fill_value : float|None
        Valeur utilisée pour remplir les pixels hors emprise lors du reproject.
        Si None : utilise nodata si disponible, sinon 0.
    return_format : str
        "bands_first" -> (B,H,W), "bands_last" -> (H,W,B)

    Returns
    -------
    HyperCube
        cube + métadonnées (CRS, transform, nodata, noms de bandes).
    """
    if not paths:
        raise ValueError("paths est vide.")

    # Résout la référence
    if isinstance(reference, int):
        ref_path = paths[reference]
    else:
        ref_path = reference

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }
    if resampling not in resampling_map:
        raise ValueError(f"resampling invalide: {resampling}. Choix: {list(resampling_map)}")

    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height, ref_width = ref.height, ref.width
        ref_nodata = ref.nodata
        ref_dtype = ref.dtypes[0]

    if out_dtype is None:
        out_dtype = np.dtype(ref_dtype)

    if fill_value is None:
        fill_value = ref_nodata if ref_nodata is not None else 0

    stacked_bands: List[np.ndarray] = []
    band_names: List[str] = []

    for p in paths:
        with rasterio.open(p) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_h, src_w = src.height, src.width
            src_nodata = src.nodata

            # Pour chaque bande du fichier
            for b in range(1, src.count + 1):
                if (src_crs == ref_crs) and (src_transform == ref_transform) and (src_h == ref_height) and (src_w == ref_width):
                    # Grille identique: lecture directe
                    arr = src.read(b).astype(out_dtype, copy=False)
                else:
                    if not allow_resample:
                        raise ValueError(
                            "Les rasters n'ont pas la même grille (CRS/transform/shape). "
                            "Active allow_resample=True pour reprojeter/rééchantillonner."
                        )
                    # Reproject vers la grille de référence
                    arr = np.full((ref_height, ref_width), fill_value, dtype=out_dtype)
                    reproject(
                        source=rasterio.band(src, b),
                        destination=arr,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        src_nodata=src_nodata,
                        dst_nodata=fill_value,
                        resampling=resampling_map[resampling],
                    )

                stacked_bands.append(arr)
                band_names.append(f"{_basename(p)}_b{b}")

    cube = np.stack(stacked_bands, axis=0)  # (bands, H, W)

    if return_format == "bands_last":
        cube_out = np.moveaxis(cube, 0, -1)  # (H, W, bands)
    elif return_format == "bands_first":
        cube_out = cube
    else:
        raise ValueError("return_format doit être 'bands_first' ou 'bands_last'.")

    return HyperCube(
        cube=cube_out,
        crs=ref_crs,
        transform=ref_transform,
        nodata=ref_nodata,
        band_names=band_names,
    )


def save_cube_as_geotiff(
    cube: np.ndarray,
    out_path: str,
    *,
    crs,
    transform,
    nodata: Optional[float] = None,
) -> None:
    """
    Sauvegarde un cube en GeoTIFF multi-bandes.
    Attend cube au format (bands, height, width).
    """
    if cube.ndim != 3:
        raise ValueError("cube doit être 3D (bands, height, width).")

    bands, height, width = cube.shape
    dtype = cube.dtype

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        compress="deflate",
        predictor=2 if np.issubdtype(dtype, np.floating) else 1,
    ) as dst:
        for i in range(bands):
            dst.write(cube[i, :, :], i + 1)


def _basename(path: str) -> str:
    # évite d'importer pathlib/os pour rester simple
    name = path.replace("\\", "/").split("/")[-1]
    if "." in name:
        name = ".".join(name.split(".")[:-1])
    return name


RGB_image = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE_0/RGB_from_hyperspectral_crop.tif"
quality_image = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE_0/quality_layer_colored_crop.tif"
vegetation_MSAVI = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegetation_indice_outputs/enmap_salsigne_MSAVI_veg.tiff"
vegetation_mask = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegetation_indice_outputs/enmap_salsigne_VEG_MASK.tiff"
water_mask = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"

output_image = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/stack_tiffs.tif" 

paths = [
    RGB_image, quality_image, vegetation_MSAVI, vegetation_mask, water_mask]

hc = tiffs_to_hyperspectral_cube(paths, allow_resample=True, resampling="bilinear")
print(hc.cube.shape)  # (bands, H, W)

save_cube_as_geotiff(hc.cube if hc.cube.ndim == 3 else np.moveaxis(hc.cube, -1, 0),
                     output_image,
                     crs=hc.crs,
                     transform=hc.transform,
                     nodata=hc.nodata)