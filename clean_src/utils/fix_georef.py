import rasterio
from pathlib import Path


def fix_raster_georef(
    raster_to_fix: str,
    reference_raster: str,
    output_path: str | None = None,
    compress: str = "deflate",
    overwrite: bool = True,
) -> str:
    """
    Corrige la géoréférence (CRS, transform, etc.) d’un raster en copiant celle
    d’un raster de référence.

    Parameters
    ----------
    raster_to_fix : str
        Raster sans géoréférence correcte.
    reference_raster : str
        Raster de référence avec géoréférence valide.
    output_path : str | None
        Chemin de sortie. Si None → écrase le fichier original (si overwrite=True).
    compress : str
        Compression GeoTIFF (ex: "deflate", "lzw").
    overwrite : bool
        Autorise l'écrasement du fichier d'origine.

    Returns
    -------
    str
        Chemin du raster corrigé.
    """

    raster_to_fix = Path(raster_to_fix)

    if output_path is None:
        if not overwrite:
            raise ValueError("output_path doit être défini si overwrite=False")
        output_path = raster_to_fix
    else:
        output_path = Path(output_path)

    # -----------------------------
    # Lire raster à corriger
    # -----------------------------
    with rasterio.open(raster_to_fix) as bad:
        arr = bad.read()
        bad_nodata = bad.nodata

    # -----------------------------
    # Lire référence géoréférencée
    # -----------------------------
    with rasterio.open(reference_raster) as ref:
        profile = ref.profile.copy()

    # -----------------------------
    # Mise à jour du profil
    # -----------------------------
    profile.update(
        driver="GTiff",
        count=arr.shape[0],
        dtype=arr.dtype,
        nodata=bad_nodata,
        compress=compress,
    )

    # -----------------------------
    # Écriture
    # -----------------------------
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr)

    print(f"[OK] Raster corrigé écrit : {output_path}")

    return str(output_path)