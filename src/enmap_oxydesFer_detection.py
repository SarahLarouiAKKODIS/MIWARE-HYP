import numpy as np
import rasterio
import os
from enmap_mineral_detection_utils import load_mineral_targets
from enmap_indices_calculation_utils import read_band, band_depth, read_scale_and_clip_bands


def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    (a-b)/(a+b) avec protection denom==0
    """
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (denom != 0)
    out[valid] = ((a[valid] - b[valid]) / denom[valid]).astype(np.float32)
    return out


def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalise x dans [0,1] avec clamp, en float32."""
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")


tif_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"

targets = load_mineral_targets(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_selected_bands_by_mineral.csv",
    "iron_oxides"
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/iron_oxides/"
os.makedirs(Path_outdata, exist_ok=True)

# ======================
# BANDES IRON OXIDES (nm)
# ======================
# Absorption ferrique ~0.9 µm : triplet 860 / 900 / 940
b860, lam860 = targets[860.0]
b900, lam900 = targets[900.0]
b940, lam940 = targets[940.0]

# Rougeur (redness) : 650 / 550
b650, lam650 = targets[650.0]
b550, lam550 = targets[550.0]

# ======================
# OUTPUT
# ======================
out_bd900   = Path_outdata + "BD900_iron_oxides.tif"
out_redness = Path_outdata + "REDNESS_iron_oxides.tif"
out_mask    = Path_outdata + "iron_oxides_mask.tif"
out_prob    = Path_outdata + "iron_oxides_probability.tif"   # <-- NOUVEAU

scale = 10000.0
MIN, MAX = 0.0, 1.2

# ======================
# SEUILS de départ (à ajuster)
# ======================
bd900_thresh    = 0.04   # profondeur absorption ~900 nm (Fe3+)
redness_thresh  = 0.05   # rougeur (red > green)

# ======================
# Paramètres "probabilité" (score) [0..1] (à ajuster selon tes scènes)
# - bd900 : plus grand = mieux
# - redness : plus grand = mieux (attention: indices dans [-1,1])
# ======================
bd900_score_min, bd900_score_max = 0.00, 0.12
red_score_min,   red_score_max   = 0.00, 0.20   # score pour redness positive (0..0.2 souvent suffisant)

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:
    bands_idx = {
        "b860": b860,
        "b900": b900,
        "b940": b940,
        "b650": b650,
        "b550": b550,
    }

    bands = read_scale_and_clip_bands(
        src, bands=bands_idx, scale=scale, min_val=MIN, max_val=MAX, verbose=True
    )

    # accès direct
    R860 = bands["b860"]
    R900 = bands["b900"]
    R940 = bands["b940"]
    R650 = bands["b650"]
    R550 = bands["b550"]

    # 1) band depth ~900 nm
    bd900 = band_depth(R860, R900, R940, lam860, lam900, lam940).astype("float32")

    # 2) redness index : (Red - Green)/(Red + Green) ; ici 650=red, 550=green
    redness = safe_norm_diff(R650, R550).astype(np.float32)
    redness = np.clip(redness, -1.0, 1.0)

    # optionnel (recommandé) : ignorer les pixels très sombres (ratio instable même si denom != 0)
    den = R650 + R550
    dark = ~np.isfinite(den) | (den <= 1e-3)   # si réflectance en [0..1]
    # si tes données sont en 0..10000, mets plutôt: (den <= 50) ou (den <= 100)
    redness[dark] = np.nan

    print("redness min/max:", np.nanmin(redness), np.nanmax(redness))
    print("percentiles:", np.nanpercentile(redness, [1, 5, 50, 95, 99]))
    print("redness_thresh:", redness_thresh)
    print("bd900_thresh:", bd900_thresh)

    # ----------------------
    # 1) Masque binaire iron oxides (0 / 255)
    # ----------------------
    mask_bool = (bd900 > bd900_thresh) & (redness > redness_thresh)

    print("pixels détectés:", int(mask_bool.sum()), "sur", int(np.isfinite(bd900).sum()))
    if np.isfinite(bd900).sum() > 0:
        print("ratio (%):", 100.0 * float(mask_bool.sum()) / float(np.isfinite(bd900).sum()))

    mask = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

    # ----------------------
    # 2) NOUVEAU : raster "probabilité" / score [0..1]
    #    - s_bd900 : absorption ferrique forte -> score élevé
    #    - s_red   : rougeur forte -> score élevé (on ne garde que la partie positive via min/max)
    #    Combinaison : sqrt(s_bd900 * s_red) (AND doux)
    #    (option) prob = 0 hors masque, comme pour micas/olivine
    # ----------------------
    s_bd900 = normalize01(bd900, bd900_score_min, bd900_score_max)

    # Si redness est négatif, normalize01 donnera 0 après clamp (vu red_score_min=0), ce qui est ok
    s_red = normalize01(redness, red_score_min, red_score_max)

    prob = np.sqrt(s_bd900 * s_red).astype("float32")
    prob = np.where(mask_bool, prob, 0.0).astype("float32")  # optionnel

    # ----------------------
    # Profils d'écriture
    # ----------------------
    prof_f = src.profile.copy()
    prof_f.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    prof_u8 = src.profile.copy()
    prof_u8.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    # ----------------------
    # Écriture
    # ----------------------
    with rasterio.open(out_bd900, "w", **prof_f) as dst:
        dst.write(bd900, 1)

    with rasterio.open(out_redness, "w", **prof_f) as dst:
        dst.write(redness, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    # NOUVEAU : enregistre le raster "probabilité"
    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd900, out_redness, out_mask, out_prob)

if np.any(mask == 255):
    print("Il existe au moins un pixel à 255 dans le masque, détection des oxydes de fer !")
else:
    print("Aucun pixel à 255")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

# (option) stats rapides sur la "probabilité" sur pixels détectés
if n_pixels > 0:
    p = prob[mask == 255]
    print("Probability score on detected pixels (min/mean/max):",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
