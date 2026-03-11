#!/usr/bin/env python3
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# 1) CLASSES ORIGINALES
# ============================================================

CLASS_MAP = {
    "Sapin, épicéa": 1,
    "Châtaignier": 2,
    "Chênes décidus": 3,
    "Chênes sempervirents": 4,
    "Conifères": 5,
    "Douglas": 6,
    "Feuillus": 7,
    "Hêtre": 8,
    "Mixte": 9,
    "NC": 10,
    "NR": 11,
    "Peuplier": 12,
    "Pin autre": 13,
    "Pin d'Alep": 14,
    "Pin laricio, pin noir": 15,
    "Pin sylvestre": 16,
    "Pins mélangés": 17,
    "Robinier": 18,
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}


# ============================================================
# 2) REGROUPEMENT EN 3 CLASSES
# ============================================================

STRESS_GROUPS = {
    1: [  # peu stressée
        CLASS_MAP["Sapin, épicéa"],
        CLASS_MAP["Chênes sempervirents"],
        CLASS_MAP["Douglas"],
        CLASS_MAP["Pin laricio, pin noir"],
        CLASS_MAP["Pin sylvestre"],
    ],
    2: [  # moyennement stressée
        CLASS_MAP["Conifères"],
        CLASS_MAP["Feuillus"],
        CLASS_MAP["Mixte"],
        CLASS_MAP["Pin autre"],
        CLASS_MAP["Pin d'Alep"],
        CLASS_MAP["Pins mélangés"],
        CLASS_MAP["Robinier"],
    ],
    3: [  # très stressée
        CLASS_MAP["Châtaignier"],
        CLASS_MAP["Chênes décidus"],
        CLASS_MAP["Hêtre"],
        CLASS_MAP["Peuplier"],
    ],
}

STRESS_CLASS_NAMES = {
    1: "Peu stressée",
    2: "Moyennement stressée",
    3: "Très stressée",
}


# ============================================================
# 3) OUTILS
# ============================================================

def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def reclassify_gt_to_stress_classes(gt_mask, stress_groups, excluded_labels=(0, 10, 11)):
    """
    Reclassifie le masque GT original en 3 classes de stress.
    Les pixels non mappés ou exclus sont mis à 0.
    """
    out = np.zeros_like(gt_mask, dtype=np.uint8)

    for stress_id, original_ids in stress_groups.items():
        for original_id in original_ids:
            out[gt_mask == original_id] = stress_id

    for lab in excluded_labels:
        out[gt_mask == lab] = 0

    return out


# ============================================================
# 4) COULEURS
# ============================================================

# Couleurs pour les 18 classes GT (0 = blanc)
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
    "#e31a1c",  # 18 Robinier
]

# Couleurs pour les 3 classes stress (0 = blanc)
STRESS_COLORS = [
    "#ffffff",  # 0 = fond / exclu
    "#2ca25f",  # 1 peu stressée
    "#fec44f",  # 2 moyennement stressée
    "#de2d26",  # 3 très stressée
]


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
    # Masque 3 classes stress
    # -------------------------
    axes[1].imshow(stress_mask, cmap=stress_cmap, vmin=0, vmax=3)
    axes[1].set_title("Masque reclassé en 3 classes de stress")
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
    gt_mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Forest/mask_foret_classes.tif"
    output_figure_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Forest/figure_gt_vs_stress.png"

    gt_mask = read_mask(gt_mask_path)

    stress_mask = reclassify_gt_to_stress_classes(
        gt_mask,
        STRESS_GROUPS,
        excluded_labels=(0, 10, 11)
    )

    plot_gt_and_stress_masks(
        gt_mask=gt_mask,
        stress_mask=stress_mask,
        output_path=output_figure_path
    )