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
    "amphiboles"  # <-- nom de la classe dans ton CSV
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/amphiboles/"
os.makedirs(Path_outdata, exist_ok=True)

# ======================
# PARAMS / SEUILS
# ======================
scale = 10000.0
MIN, MAX = 0.0, 1.2

# Amphiboles : absorption ~2.30–2.35 µm (Mg/Fe-OH) souvent marquée
# On utilise un triplet ~2250 / 2320 / 2390 (à adapter à ton CSV)
bd2320_thresh = 0.03   # critère principal amphiboles

# Optionnel : contrôle ~2.00 µm (pyroxènes, etc.) pour éviter confusion (si dispo)
bd2000_thresh = 0.02

# Paramètres score [0..1] (normalisation)
bd2320_score_min, bd2320_score_max = 0.00, 0.10
bd2000_score_min, bd2000_score_max = 0.00, 0.08

# ======================
# UTILS
# ======================
def get_target(targets_dict, wl):
    wl = float(wl)
    if wl not in targets_dict:
        raise KeyError(
            f"Longueur d’onde {wl} nm absente de targets. "
            f"Exemples de clés dispo: {sorted(targets_dict.keys())[:15]} ..."
        )
    return targets_dict[wl]

def normalize01(x, vmin, vmax):
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")

# ======================
# CHOIX DES BANDES (amphiboles)
# ======================
# 1) Absorption diagnostique ~2320 nm (amphiboles)
# Triplet typique : 2250 (épaule gauche) / 2320 (centre) / 2390 (épaule droite)
b2250, lam2250 = get_target(targets, 2250.0)
b2320, lam2320 = get_target(targets, 2320.0)
b2390, lam2390 = get_target(targets, 2390.0)

# 2) Optionnel : absorption ~2000 nm (contrôle), triplet 1900 / 2000 / 2100
has_bd2000 = True
try:
    b1900, lam1900 = get_target(targets, 1900.0)
    b2000, lam2000 = get_target(targets, 2000.0)
    b2100, lam2100 = get_target(targets, 2100.0)
except KeyError:
    has_bd2000 = False

out_bd2320 = Path_outdata + "BD2320_amphiboles.tif"
out_bd2000 = Path_outdata + "BD2000_control.tif"
out_mask   = Path_outdata + "amphiboles_mask.tif"
out_prob   = Path_outdata + "amphiboles_probability.tif"

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:

    bands_idx = {
        "b2250": b2250,
        "b2320": b2320,
        "b2390": b2390,
    }
    if has_bd2000:
        bands_idx.update({
            "b1900": b1900,
            "b2000": b2000,
            "b2100": b2100,
        })

    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    R2250 = bands["b2250"]
    R2320 = bands["b2320"]
    R2390 = bands["b2390"]

    bd2320 = band_depth(R2250, R2320, R2390, lam2250, lam2320, lam2390).astype("float32")

    if has_bd2000:
        R1900 = bands["b1900"]
        R2000 = bands["b2000"]
        R2100 = bands["b2100"]
        bd2000 = band_depth(R1900, R2000, R2100, lam1900, lam2000, lam2100).astype("float32")
    else:
        bd2000 = None

    # ----------------------
    # 1) Masque binaire amphiboles (0 / 255)
    # ----------------------
    if has_bd2000:
        # condition plus stricte (évite pixels sans OH réel) : les deux absorptions présentes
        mask_bool = (bd2320 > bd2320_thresh) & (bd2000 > bd2000_thresh)
    else:
        mask_bool = (bd2320 > bd2320_thresh)

    mask = (mask_bool.astype("uint8") * 255).astype("uint8")

    # ----------------------
    # 2) Score "probabilité" [0..1] (normalisé, pas %)
    # ----------------------
    s2320 = normalize01(bd2320, bd2320_score_min, bd2320_score_max)

    if has_bd2000:
        s2000 = normalize01(bd2000, bd2000_score_min, bd2000_score_max)
        prob = np.sqrt(s2320 * s2000).astype("float32")
    else:
        prob = s2320.astype("float32")

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
    with rasterio.open(out_bd2320, "w", **prof_f) as dst:
        dst.write(bd2320, 1)

    if has_bd2000:
        with rasterio.open(out_bd2000, "w", **prof_f) as dst:
            dst.write(bd2000, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd2320, (out_bd2000 if has_bd2000 else "(BD2000 non calculé)"), out_mask, out_prob)

if np.any(mask == 255):
    print("Détection amphiboles : au moins un pixel détecté ✅")
else:
    print("Aucun pixel détecté (seuils trop stricts ou pas de signature amphibole)")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

if n_pixels > 0:
    p = prob[mask == 255]
    print("Probability score on detected pixels (min/mean/max):",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
