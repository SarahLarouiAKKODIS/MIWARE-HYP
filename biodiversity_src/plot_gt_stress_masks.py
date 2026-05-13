#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from utils.commun_functions import load_config, read_mask
from utils.classification_functions import reclassify_gt_to_stress_classes
from utils.display import print_stress_class_distribution

# ============================================================
# 4) COULEURS
# ============================================================

# Couleurs pour les 21 classes GT (0 = fond / no data)
GT_COLORS = [
    "#ffffff",  # 0 = fond / no data

    "#1b9e77",  # 1 Sapin, épicéa
    "#d95f02",  # 2 Châtaignier
    "#7570b3",  # 3 Chênes décidus
    "#66a61e",  # 4 Chênes sempervirents
    "#e7298a",  # 5 Conifères
    "#a6761d",  # 6 Douglas
    "#666666",  # 7 Feuillus
    "#1f78b4",  # 8 Hêtre
    "#b2df8a",  # 9 Mixte
    "#fb9a99",  # 10 NC
    "#cab2d6",  # 11 NR
    "#fdbf6f",  # 12 Peuplier
    "#ff7f00",  # 13 Pin autre
    "#6a3d9a",  # 14 Pin d'Alep
    "#b15928",  # 15 Pin laricio, pin noir
    "#a6cee3",  # 16 Pin sylvestre
    "#33a02c",  # 17 Pins mélangés
    "#ffcc00",  # 18 Pin maritime (jaune/orangé → pin distinct)
    "#8dd3c7",  # 19 Pin à crochets, pin cembro (bleu-vert clair → alpin)
    "#bc80bd",  # 20 Mélèze (violet clair → caduc conifère, distinct)
    "#e31a1c",  # 21 Robinier (rouge → feuillu particulier)
]
# Couleurs pour les 4 classes stress (0 = blanc)
STRESS_COLORS = [
    "#ffffff",  # 0 = fond / exclu
    "#99d8c9",  # 1 pas stressée
    "#2ca25f",  # 2 peu stressée
    "#fec44f",  # 3 moyennement stressée
    "#de2d26",  # 4 très stressée
]

# STRESS_COLORS = [
#     "#ffffff",  # 0 = fond / exclu
#     "#31a354",  # 1 pas stressée (vert)
#     "#ffd92f",  # 2 peu stressée (jaune)
#     "#fd8d3c",  # 3 moyennement stressée (orange)
#     "#de2d26",  # 4 très stressée (rouge)
# ]

# ============================================================
# 5) FIGURE
# ============================================================

def plot_gt_and_stress_masks(gt_mask, stress_mask, output_path=None):
    gt_cmap = ListedColormap(GT_COLORS)
    stress_cmap = ListedColormap(STRESS_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # -------------------------
    # Masque GT original
    # -------------------------
    axes[0].imshow(gt_mask, cmap=gt_cmap, vmin=0, vmax=max(ID_TO_CLASS.keys()))
    axes[0].set_title("Masque vérité terrain (classes originales)")
    axes[0].axis("off")

    gt_legend_elements = []
    present_gt_labels = sorted(np.unique(gt_mask))
    for lab in present_gt_labels:
        if lab == 0:
            continue
        if lab in ID_TO_CLASS:
            gt_legend_elements.append(
                Patch(facecolor=GT_COLORS[lab], edgecolor="black", label=ID_TO_CLASS[lab])
            )

    axes[0].legend(
        handles=gt_legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
        title="Classes GT"
    )

    # -------------------------
    # Masque 4 classes stress
    # -------------------------
    axes[1].imshow(stress_mask, cmap=stress_cmap, vmin=0, vmax=4)
    axes[1].set_title("Masque reclassé en 4 classes de stress")
    axes[1].axis("off")

    stress_legend_elements = []
    present_stress_labels = sorted(np.unique(stress_mask))
    for lab in present_stress_labels:
        if lab == 0:
            continue
        stress_legend_elements.append(
            Patch(facecolor=STRESS_COLORS[lab], edgecolor="black", label=STRESS_CLASS_NAMES[lab])
        )

    axes[1].legend(
        handles=stress_legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=10,
        title="Classes stress"
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 6) MAIN
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # CONFIG
    # ============================================================

    # MASQUE FOREST ESSENCES
    config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/cma.json"
    config = load_config(config_path)

    Path_res = Path(config["Path_res"]) 
    Path_res = Path_res / "Forest"
    gt_mask_path = Path_res / "mask_foret_classes.tif"
    output_figure_path = Path_res / "figure_gt_vs_stress.png"

    gt_mask = read_mask(gt_mask_path)

    ## CLASSES DES ESSENCES ET DES ETAT DE STRESSE   
    config_forest = load_config(config["config_forest_path"])
   
    # --- 1) Classes d'essences ---
    CLASS_MAP = config_forest["CLASS_MAP"]                 # nom -> id
    ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}     # id -> nom

    # --- 2) Groupes de stress ---
    def keys_to_int(d):
        return {int(k): v for k, v in d.items()}

    STRESS_GROUPS = keys_to_int(config_forest["STRESS_GROUPS"])
    STRESS_CLASS_NAMES = keys_to_int(config_forest["STRESS_CLASS_NAMES"])

    # --- 3) Conversion des espèces en labels numériques ---
    STRESS_GROUPS_LABELS = {
        group: [CLASS_MAP[species] for species in species_list]
        for group, species_list in STRESS_GROUPS.items()
    }
    # ============================================================
    # RECLASSIFICATION DES PIXELS
    # ============================================================

    stress_mask = reclassify_gt_to_stress_classes(
        gt_mask,
        STRESS_GROUPS_LABELS,
        excluded_labels=(0, 10, 11)
    )

    print_stress_class_distribution(
    stress_mask,
    stress_class_names=STRESS_CLASS_NAMES
)

    print('stress_mask', stress_mask.shape)
    print('stress_mask', np.unique(stress_mask))

    # ============================================================
    # PLOT
    # ============================================================

    plot_gt_and_stress_masks(
        gt_mask=gt_mask,
        stress_mask=stress_mask,
        output_path=output_figure_path
    )