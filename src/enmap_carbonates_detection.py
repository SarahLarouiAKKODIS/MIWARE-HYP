import numpy as np
import os
import rasterio
from enmap_mineral_detection_utils import load_mineral_targets
from enmap_indices_calculation_utils import read_scale_and_clip_bands, band_depth

# ======================
# A MODIFIER
# ======================
tif_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"

# Cibles (bandes 1-based + longueurs d’onde réelles) depuis ton CSV
targets = load_mineral_targets(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_selected_bands_by_mineral.csv",
    "carbonates"  # <-- nom de la classe dans ton CSV
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/carbonates/"
os.makedirs(Path_outdata, exist_ok=True)

# ======================
# PARAMS / SEUILS
# ======================
scale = 10000.0
MIN, MAX = 0.0, 1.2  # bornes physiques post-scale

# Seuils de départ (à ajuster selon ta scène)
# Carbonates : absorption ~2.33–2.35 µm souvent marquée
bd2330_thresh = 0.03      # band depth principal CO3
bd2500_thresh = 0.02      # optionnel si 2.50 µm dispo

# Pour la "probabilité" (score) : bornes min/max des band depths pour normaliser en [0..1]
# (à ajuster selon tes scènes ; valeurs par défaut pour un rendu lisible)
bd2330_score_min, bd2330_score_max = 0.00, 0.10
bd2500_score_min, bd2500_score_max = 0.00, 0.08  # si utilisé

# ======================
# UTILS
# ======================
def get_target(targets_dict, wl):
    """Retourne (band_index, wavelength_nm) pour une clé wl (float) présente dans targets."""
    if float(wl) not in targets_dict:
        raise KeyError(
            f"Longueur d’onde {wl} nm absente de targets. "
            f"Clés dispo: {sorted(targets_dict.keys())[:10]} ..."
        )
    return targets_dict[float(wl)]

def normalize01(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalise x dans [0,1] avec clamp, en float32."""
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")

# ======================
# Choix des longueurs d’onde pour carbonates
# ======================
# 1) Principal CO3 autour de 2330-2350 nm
b2200, lam2200 = get_target(targets, 2200.0)  # épaule gauche
b2330, lam2330 = get_target(targets, 2330.0)  # centre absorption carbonate
b2450, lam2450 = get_target(targets, 2450.0)  # épaule droite

# 2) Optionnel : absorption ~2.50 µm (si ton CSV la fournit)
has_bd2500 = True
try:
    b2400, lam2400 = get_target(targets, 2400.0)  # épaule gauche 2.50
    b2500, lam2500 = get_target(targets, 2500.0)  # centre 2.50
    b2600, lam2600 = get_target(targets, 2600.0)  # épaule droite 2.50
except KeyError:
    has_bd2500 = False

out_bd2330 = Path_outdata + "BD2330_carbonates.tif"
out_bd2500 = Path_outdata + "BD2500_carbonates.tif"
out_mask   = Path_outdata + "carbonates_mask.tif"
out_prob   = Path_outdata + "carbonates_probability.tif"  # <-- NOUVEAU

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:

    # Lire + scale + clip en une fois (toutes les bandes nécessaires)
    bands_idx = {
        "b2200": b2200,
        "b2330": b2330,
        "b2450": b2450,
    }

    if has_bd2500:
        bands_idx.update({
            "b2400": b2400,
            "b2500": b2500,
            "b2600": b2600,
        })

    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    R2200 = bands["b2200"]
    R2330 = bands["b2330"]
    R2450 = bands["b2450"]

    # Band depth carbonate principal ~2330 nm
    bd2330 = band_depth(R2200, R2330, R2450, lam2200, lam2330, lam2450).astype("float32")

    # Optionnel : band depth ~2500 nm
    if has_bd2500:
        R2400 = bands["b2400"]
        R2500 = bands["b2500"]
        R2600 = bands["b2600"]
        bd2500 = band_depth(R2400, R2500, R2600, lam2400, lam2500, lam2600).astype("float32")
    else:
        bd2500 = None

    # ----------------------
    # 1) Masque binaire carbonates (0 / 255)
    # ----------------------
    if has_bd2500:
        mask_bool = (bd2330 > bd2330_thresh) & (bd2500 > bd2500_thresh)
    else:
        mask_bool = (bd2330 > bd2330_thresh)

    mask = (mask_bool.astype("uint8") * 255).astype("uint8")

    # ----------------------
    # 2) NOUVEAU : "probabilité" / score [0..1]
    #    - Sans bd2500 : score = norm(bd2330)
    #    - Avec bd2500 : score = sqrt( norm2330 * norm2500 ) (AND doux)
    #    Et on met à 0 hors masque (optionnel).
    # ----------------------
    s2330 = normalize01(bd2330, bd2330_score_min, bd2330_score_max)

    if has_bd2500:
        s2500 = normalize01(bd2500, bd2500_score_min, bd2500_score_max)
        prob = np.sqrt(s2330 * s2500).astype("float32")
    else:
        prob = s2330.astype("float32")

    prob = np.where(mask_bool, prob, 0.0).astype("float32")  # optionnel

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
    with rasterio.open(out_bd2330, "w", **prof_f) as dst:
        dst.write(bd2330, 1)

    if has_bd2500:
        with rasterio.open(out_bd2500, "w", **prof_f) as dst:
            dst.write(bd2500, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    # NOUVEAU : enregistre le raster "probabilité"
    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd2330, (out_bd2500 if has_bd2500 else "(BD2500 non calculé)"), out_mask, out_prob)

if np.any(mask == 255):
    print("Détection carbonates : au moins un pixel détecté ✅")
else:
    print("Aucun pixel détecté (seuils trop stricts ou pas de signature carbonate)")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

# (option) stats rapides sur la "probabilité" sur pixels détectés
if n_pixels > 0:
    p = prob[mask == 255]
    print("Probability score on detected pixels (min/mean/max):",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
