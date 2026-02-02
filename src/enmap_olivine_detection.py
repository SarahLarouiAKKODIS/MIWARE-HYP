import numpy as np
import rasterio
import os
from enmap_mineral_detection_utils import load_mineral_targets
from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth


# ======================
tif_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"

targets = load_mineral_targets(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_selected_bands_by_mineral.csv",
    "olivine"
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/olivine/"
os.makedirs(Path_outdata, exist_ok=True)

b860,  lam860  = targets[860.0]
b1050, lam1050 = targets[1050.0]
b1280, lam1280 = targets[1280.0]

b1800, lam1800 = targets[1800.0]
b2000, lam2000 = targets[2000.0]
b2300, lam2300 = targets[2300.0]

out_bd1050 = Path_outdata + "BD1050_olivine.tif"
out_bd2000 = Path_outdata + "BD2000_control.tif"
out_mask   = Path_outdata + "olivine_mask.tif"
out_prob   = Path_outdata + "olivine_probability.tif"   # <-- NOUVEAU

# Seuils de départ (à ajuster)
bd1050_thresh = 0.05   # olivine si bande ~1µm marquée
bd2000_max    = 0.02   # et bande ~2µm faible (pyroxènes -> souvent plus fort)

# Pour la "probabilité" (score) : normalisation en [0..1]
# Ici : score olivine = AND doux entre "bd1050 fort" et "bd2000 faible"
bd1050_score_min, bd1050_score_max = 0.00, 0.15  # à ajuster selon la scène
bd2000_good_min,  bd2000_good_max  = 0.00, 0.06  # plus bd2000 est petit, mieux c'est

scale = 10000.0
MIN, MAX = 0.0, 1.2

# ======================
# UTILS
# ======================
def normalize01(x, vmin, vmax):
    """Normalise x dans [0,1] avec clamp, en float32."""
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:

    bands_idx = {
        "b860": b860,
        "b1050": b1050,
        "b1280": b1280,
        "b1800": b1800,
        "b2000": b2000,
        "b2300": b2300,
    }

    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    # accès direct
    R860  = bands["b860"]
    R1050 = bands["b1050"]
    R1280 = bands["b1280"]
    R1800 = bands["b1800"]
    R2000 = bands["b2000"]
    R2300 = bands["b2300"]

    bd1050 = band_depth(R860, R1050, R1280, lam860, lam1050, lam1280).astype("float32")
    bd2000 = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype("float32")

    # ----------------------
    # 1) Masque binaire (0 / 255)
    # ----------------------
    mask_bool = (bd1050 > bd1050_thresh) & (bd2000 < bd2000_max)
    mask = (mask_bool.astype("uint8") * 255).astype("uint8")

    # ----------------------
    # 2) NOUVEAU : "probabilité" / score [0..1]
    #    - s1050 : plus bd1050 est grand -> plus olivine probable
    #    - s2000 : plus bd2000 est petit -> plus olivine probable (donc score inversé)
    #    Combinaison : sqrt(s1050 * s2000) (AND doux)
    #    (option) prob = 0 hors masque, comme pour micas
    # ----------------------
    s1050 = normalize01(bd1050, bd1050_score_min, bd1050_score_max)

    # Score "faible bd2000" : on normalise puis on inverse
    s2000_bad = normalize01(bd2000, bd2000_good_min, bd2000_good_max)
    s2000 = 1.0 - s2000_bad

    prob = np.sqrt(s1050 * s2000).astype("float32")
    prob = np.where(mask_bool, prob, 0.0).astype("float32")  # optionnel

    # ======================
    # ECRITURE
    # ======================
    prof_f = src.profile.copy()
    prof_f.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    prof_u8 = src.profile.copy()
    prof_u8.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    with rasterio.open(out_bd1050, "w", **prof_f) as dst:
        dst.write(bd1050, 1)

    with rasterio.open(out_bd2000, "w", **prof_f) as dst:
        dst.write(bd2000, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    # NOUVEAU : enregistre le raster "probabilité"
    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd1050, out_bd2000, out_mask, out_prob)

if np.any(mask == 255):
    print("Il existe au moins un pixel à 255 dans le masque, détection d'olivine !")
else:
    print("Aucun pixel à 255")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

# (option) stats rapides sur la "probabilité" sur pixels détectés
if n_pixels > 0:
    p = prob[mask == 255]
    print("Probability score on detected pixels (min/mean/max):",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
