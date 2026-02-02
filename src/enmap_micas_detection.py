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
    "micas"  # <-- nom de la classe dans ton CSV
)

Path_outdata = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Mineral_detection/micas/"
os.makedirs(Path_outdata, exist_ok=True)

# ======================
# PARAMS / SEUILS
# ======================
scale = 10000.0
MIN, MAX = 0.0, 1.2

# Seuils de départ (à ajuster)
bd2200_thresh = 0.03   # critère principal mica
bd1900_thresh = 0.02   # optionnel (eau/OH), utile pour robustesse

# Pour la "probabilité" (score) : borne min/max du band depth utilisé pour normaliser en [0..1]
# (à ajuster selon tes scènes ; ces valeurs donnent souvent un rendu lisible)
bd2200_score_min, bd2200_score_max = 0.00, 0.10
bd1900_score_min, bd1900_score_max = 0.00, 0.08  # si utilisé

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
    """Normalise x dans [0,1] avec clamp, en float32."""
    x = x.astype("float32")
    y = (x - vmin) / (vmax - vmin + 1e-12)
    return np.clip(y, 0.0, 1.0).astype("float32")

# ======================
# CHOIX DES BANDES (micas)
# ======================
# 1) Absorption diagnostique ~2200 nm (Al-OH)
b2100, lam2100 = get_target(targets, 2100.0)
b2200, lam2200 = get_target(targets, 2200.0)
b2300, lam2300 = get_target(targets, 2300.0)

# 2) Optionnel : absorption ~1900 nm (H2O/OH) pour robustesse
has_bd1900 = True
try:
    b1800, lam1800 = get_target(targets, 1800.0)
    b1900, lam1900 = get_target(targets, 1900.0)
    b2000, lam2000 = get_target(targets, 2000.0)
except KeyError:
    has_bd1900 = False

out_bd2200  = Path_outdata + "BD2200_micas.tif"
out_bd1900  = Path_outdata + "BD1900_control.tif"
out_mask    = Path_outdata + "micas_mask.tif"
out_prob    = Path_outdata + "micas_probability.tif"  # <-- NOUVEAU : masque "probabilité" (score 0..1)

# ======================
# CALCUL
# ======================
with rasterio.open(tif_path) as src:

    bands_idx = {
        "b2100": b2100,
        "b2200": b2200,
        "b2300": b2300,
    }
    if has_bd1900:
        bands_idx.update({
            "b1800": b1800,
            "b1900": b1900,
            "b2000": b2000,
        })

    # Lecture + scale + clipping
    bands = read_scale_and_clip_bands(
        src,
        bands=bands_idx,
        scale=scale,
        min_val=MIN,
        max_val=MAX,
        verbose=True
    )

    R2100 = bands["b2100"]
    R2200 = bands["b2200"]
    R2300 = bands["b2300"]

    # Band depth ~2200 nm (micas)
    bd2200 = band_depth(R2100, R2200, R2300, lam2100, lam2200, lam2300).astype("float32")

    # Optionnel : band depth ~1900 nm
    if has_bd1900:
        R1800 = bands["b1800"]
        R1900 = bands["b1900"]
        R2000 = bands["b2000"]
        bd1900 = band_depth(R1800, R1900, R2000, lam1800, lam1900, lam2000).astype("float32")
    else:
        bd1900 = None

    # ----------------------
    # 1) Masque binaire mica probable (0 / 255)
    # ----------------------
    if has_bd1900:
        mask_bool = (bd2200 > bd2200_thresh) & (bd1900 > bd1900_thresh)
    else:
        mask_bool = (bd2200 > bd2200_thresh)

    mask = (mask_bool.astype("uint8") * 255).astype("uint8")

    # ----------------------
    # 2) NOUVEAU : "probabilité" / score [0..1] (float32)
    #    Idée : normaliser les band depths, puis combiner.
    #    - Sans bd1900 : score = norm(bd2200)
    #    - Avec bd1900 : score = sqrt( norm2200 * norm1900 ) (AND doux)
    #    Et on remet à 0 hors masque binaire (optionnel mais souvent souhaité).
    # ----------------------
    s2200 = normalize01(bd2200, bd2200_score_min, bd2200_score_max)

    if has_bd1900:
        s1900 = normalize01(bd1900, bd1900_score_min, bd1900_score_max)
        prob = np.sqrt(s2200 * s1900).astype("float32")
    else:
        prob = s2200

    # (option) Probabilité seulement là où le masque détecte
    prob = np.where(mask_bool, prob, 0.0).astype("float32")

    p = prob[mask == 255]

    p_min = np.nanmin(p)
    p_max = np.nanmax(p)

    print("Probabilité min :", p_min)
    print("Probabilité max :", p_max)


    # ======================
    # ECRITURE
    # ======================
    prof_f = src.profile.copy()
    prof_f.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    prof_u8 = src.profile.copy()
    prof_u8.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    with rasterio.open(out_bd2200, "w", **prof_f) as dst:
        dst.write(bd2200, 1)

    if has_bd1900:
        with rasterio.open(out_bd1900, "w", **prof_f) as dst:
            dst.write(bd1900, 1)

    with rasterio.open(out_mask, "w", **prof_u8) as dst:
        dst.write(mask, 1)

    # NOUVEAU : enregistre le raster "probabilité"
    with rasterio.open(out_prob, "w", **prof_f) as dst:
        dst.write(prob, 1)

print("✅ Créés :", out_bd2200, (out_bd1900 if has_bd1900 else "(BD1900 non calculé)"), out_mask, out_prob)

if np.any(mask == 255):
    print("Détection micas : au moins un pixel détecté ✅")
else:
    print("Aucun pixel détecté (seuils trop stricts ou pas de signature mica)")

n_pixels = int(np.sum(mask == 255))
print("Nombre de pixels à 255 :", n_pixels)

# (option) stats rapides sur la "probabilité"
if n_pixels > 0:
    p = prob[mask == 255]
    print("Probabilité (score) sur pixels détectés -> min/mean/max:",
          float(np.nanmin(p)), float(np.nanmean(p)), float(np.nanmax(p)))
