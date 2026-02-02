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

# ======================
# A MODIFIER
# ======================
Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/"
image_name = "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"
# cube_tif = Path_data + "SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/" + image_name
# out_rgb_tif = Path_data + "RGB_from_hyperspectral_crop.tif"

cube_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"
out_rgb_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/RGB_from_hyperspectral_clean_crop.tif"

bands_csv = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full.csv"

# Cibles RGB "naturelles" (nm) - tu peux changer
target_R = 650
target_G = 560
target_B = 480

# ======================

df = pd.read_csv(bands_csv)
df["wavelength_nm"] = df["wavelength_nm"].astype(float)

bR, lamR = closest_band(df, target_R)
bG, lamG = closest_band(df, target_G)
bB, lamB = closest_band(df, target_B)

print("✅ Bandes RGB choisies (1-based):")
print(f"R: cible {target_R} nm -> bande {bR} (λ={lamR:.3f} nm)")
print(f"G: cible {target_G} nm -> bande {bG} (λ={lamG:.3f} nm)")
print(f"B: cible {target_B} nm -> bande {bB} (λ={lamB:.3f} nm)")

with rasterio.open(cube_tif) as src:
    R = read_band(src, bR)
    G = read_band(src, bG)
    B = read_band(src, bB)

    # Si ton L2A est en entiers scalés (0..10000), décommente :
    # scale = 10000.0
    # R, G, B = R/scale, G/scale, B/scale

    R8 = stretch_2_98(R)
    G8 = stretch_2_98(G)
    B8 = stretch_2_98(B)

    profile = src.profile.copy()
    profile.update(count=3, dtype="uint8", nodata=0)

    with rasterio.open(out_rgb_tif, "w", **profile) as dst:
        dst.write(R8, 1)  # bande 1 = R
        dst.write(G8, 2)  # bande 2 = G
        dst.write(B8, 3)  # bande 3 = B

print("✅ RGB exporté :", out_rgb_tif)
