from pathlib import Path
import numpy as np
import rasterio


def rescale_enmap_cube_simple(
    input_tif: str | Path,
    output_tif: str | Path,
    *,
    scale_factor: float = 10000.0,
    dtype: str = "float32"
) -> None:
    """
    Applique un rescaling simple (division) à un cube hyperspectral
    en conservant correctement le nodata source.
    """
    input_tif = Path(input_tif)
    output_tif = Path(output_tif)

    if not input_tif.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_tif}")

    if scale_factor == 0:
        raise ValueError("scale_factor ne peut pas être égal à 0.")

    with rasterio.open(input_tif) as src:
        raw = src.read().astype(np.float32)
        profile = src.profile.copy()
        src_nodata = src.nodata

    if src_nodata is not None:
        raw = np.where(raw == float(src_nodata), np.nan, raw)

    with rasterio.open(input_tif) as src:
        print("src.nodata =", src.nodata)
        cube_raw = src.read().astype(np.float32)

    print("min brut =", np.min(cube_raw))
    print("max brut =", np.max(cube_raw))
    print("contains -32768 ?", np.any(cube_raw == -32768))

    cube = raw / scale_factor

    finite = np.isfinite(cube)
    if finite.any():
        print(
            f"[INFO] Stats après rescaling : "
            f"min={np.nanmin(cube):.4f}, "
            f"max={np.nanmax(cube):.4f}, "
            f"mean={np.nanmean(cube):.4f}"
        )
    else:
        print("[INFO] Aucun pixel valide après rescaling.")

    profile.update(dtype=dtype)
    profile.pop("nodata", None)

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(cube.astype(dtype))