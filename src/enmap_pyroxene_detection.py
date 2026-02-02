import numpy as np
import os
import rasterio
from enmap_mineral_detection_utils import load_mineral_targets
from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth

# ======================
# A MODIFIER
# ======================
tif_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"

targets = load_mineral_targets(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_selected_bands_by_mineral.csv",
    "pyroxene"
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/pyroxene/"
os.makedirs(Path_outdata, exist_ok=True)

b900,  lam900  = targets[900.0]
b1000, lam1000 = targets[1000.0]
b1200, lam1200 = targets[1200.0]
b1800, lam1800 = targets[1800.0]
b2000, lam2000 = targets[2000.0]
b2300, lam2300 = targets[2300.0]

out_bd1um  = Path_outdata + "BD1um_pyroxene.tif"
out_bd2um  = Path_outdata + "BD2um_pyroxene.tif"
out_mask   = Path_outdata + "pyroxene_mask.tif"
out_prob   = Path_outdata + "pyroxene_probability.tif"  # <-- NOUVEAU

# ======================
# Seuils de départ (à ajuster) — PYROXENE
# ======================
bd1um_thresh = 0.05     # bande ~1 µm marquée
bd2um_thresh = 0.03     # bande ~2 µm marquée

# Normalisation score [0..1] (à ajuster selon tes scènes)
bd1um_score_min, bd1um_score_max = 0.00, 0.15
bd2um_score_min, bd2um_score_max = 0.00, 0.12

scale = 10000.0
MIN, MAX = 0.0, 1.2

# ======================
# UTILS
# ======================
def normalize01(x, vmin, vmax):
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:

    bands_idx = {
        "b900": b900,
        "b1000": b1000,
        "b1200": b1200,
        "b1800": b1800,
        "b2000": b2000,
        "b2300": b2300,
    }

    bands = read_scale_and_clip_bands(
        src, bands=bands_idx, scale=scale, min_val=MIN, max_val=MAX, verbose=True
    )

    R900  = bands["b900"]
    R1000 = bands["b1000"]
    R1200 = bands["b1200"]
    R1800 = bands["b1800"]
    R2000 = bands["b2000"]
    R2300 = bands["b2300"]

    # Band depth ~1 µm (pyroxènes)
    bd1um = band_depth(R900, R1000, R1200, lam900, lam1000, lam1200).astype("float32")

    # Band depth ~2 µm (pyroxènes)
    bd2um = band_depth(R1800, R2000, R2300, lam1800, lam2000, lam2300).astype("float32")

    # ----------------------
    # 1) Masque binaire pyroxène (0 / 255)
    # ----------------------
    mask_bool = (bd1um > bd1um_thresh) & (bd2um > bd2um_thresh)
    mask = (mask_bool.astype("uint8") * 255).astype("uint8")

    # ----------------------
    # 2) Score "probabilité" [0..1] SUR PIXELS DÉTECTÉS
    #    - s1 = norm(bd1um)
    #    - s2 = norm(bd2um)
    #    - prob = sqrt(s1 * s2) (AND doux)
    #    - prob = 0 hors détection (comme tu le veux)
    # ----------------------
    s1 = normalize01(bd1um, bd1um_score_min, bd1um_score_max)
    s2 = normalize01(bd2um, bd2um_score_min, bd2um_score_max)
    prob = np.sqrt(s1 * s2).astype("float32")
    prob = np.where(mask_bool, prob, 0.0).astype("float32")  # <-- seulement sur pixels détectés

    # ----------------------
    # Profils de sortie
    # ----------------------
    prof_f = src.profile.copy()
    prof_f.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    prof_u8 = src.profile.copy()
    prof_u8.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    # ----------------------
    # Écriture
    # ----------------------
    with rasterio.open(out_bd1um, "w", **prof_f) as dst:
        dst.write(bd1um, 1)

    with rasterio.open(out_bd2um, "w", **prof_f) as dst:
        dst.write(bd2um, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd1um, out_bd2um, out_mask, out_prob)

if np.any(mask == 255):
    print("Détection pyroxène : au moins un pixel détecté ✅")
else:
    print("Aucun pixel détecté (seuils trop stricts ou pas de signature pyroxène)")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

if n_pixels > 0:
    p = prob[mask == 255]
    print("Probability score on detected pixels (min/mean/max):",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
