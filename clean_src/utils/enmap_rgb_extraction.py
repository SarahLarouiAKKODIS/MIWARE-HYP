import numpy as np
import pandas as pd
import rasterio


def closest_band(df, target_nm: float) -> tuple[int, float]:
    """Retourne (band_id_1based, wavelength_nm_reelle) le plus proche de target_nm."""
    idx = (df["wavelength_nm"] - target_nm).abs().idxmin()
    return int(df.loc[idx, "band_id"]), float(df.loc[idx, "wavelength_nm"])


def read_band(src, band_1based: int) -> np.ndarray:
    arr = src.read(band_1based).astype("float32")
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def stretch_2_98(arr: np.ndarray) -> np.ndarray:
    """Etirement robuste pour affichage (2–98 percentiles) -> uint8."""
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    p2, p98 = np.percentile(v, [2, 98])
    if p98 <= p2:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - p2) / (p98 - p2)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    out[~np.isfinite(arr)] = 0
    return out

def apply_gamma(arr, gamma=2.2):
    arr = arr.astype(np.float32) / 255.0
    arr = np.power(arr, 1/gamma)
    return (arr * 255).astype(np.uint8)

def hyperspectral_to_rgb(
    cube_tif: str,
    bands_csv: str,
    out_rgb_tif: str,
    target_R: float = 650,
    target_G: float = 560,
    target_B: float = 480,
    scale: float | None = None,
):
    """
    Convertit une image hyperspectrale en RGB en choisissant les bandes les plus proches
    des longueurs d'onde cibles.

    Args:
        cube_tif: chemin vers le raster hyperspectral
        bands_csv: CSV contenant band_id et wavelength_nm
        out_rgb_tif: chemin de sortie RGB
        target_R, target_G, target_B: longueurs d'onde cibles (nm)
        scale: facteur de normalisation (ex: 10000) ou None
    """

    # --- lecture des bandes ---
    df = pd.read_csv(bands_csv)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

    bR, lamR = closest_band(df, target_R)
    bG, lamG = closest_band(df, target_G)
    bB, lamB = closest_band(df, target_B)

    print("✅ Bandes RGB choisies (1-based):")
    print(f"R: cible {target_R} nm -> bande {bR} (λ={lamR:.3f} nm)")
    print(f"G: cible {target_G} nm -> bande {bG} (λ={lamG:.3f} nm)")
    print(f"B: cible {target_B} nm -> bande {bB} (λ={lamB:.3f} nm)")

    # --- lecture raster ---
    with rasterio.open(cube_tif) as src:
        R = read_band(src, bR)
        G = read_band(src, bG)
        B = read_band(src, bB)

        # scaling optionnel
        if scale is not None:
            R, G, B = R / scale, G / scale, B / scale

        # stretch
        stack = np.stack([R, G, B])
        v = stack[np.isfinite(stack)]

        p2, p98 = np.percentile(v, [2, 98])

        def stretch_shared(arr):
            out = (arr - p2) / (p98 - p2)
            out = np.clip(out, 0, 1)
            out = (out * 255).astype(np.uint8)
            out[~np.isfinite(arr)] = 0
            return out

        R8 = stretch_shared(R)
        G8 = stretch_shared(G)
        B8 = stretch_shared(B)

        R8 = apply_gamma(R8)
        G8 = apply_gamma(G8)
        B8 = apply_gamma(B8)

        profile = src.profile.copy()
        profile.update(count=3, dtype="uint8", nodata=0)

        with rasterio.open(out_rgb_tif, "w", **profile) as dst:
            dst.write(R8, 1)
            dst.write(G8, 2)
            dst.write(B8, 3)

    print("✅ RGB exporté :", out_rgb_tif)