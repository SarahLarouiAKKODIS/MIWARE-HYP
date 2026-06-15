import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def print_stress_class_distribution(out, stress_class_names=None):
    """
    Affiche le nombre de pixels par classe de stress.

    Parameters
    ----------
    out : np.ndarray
        Masque reclassifié.
    stress_class_names : dict, optional
        Exemple :
        {
            "1": "Pas stressée",
            "2": "Peu stressée",
            "3": "Moyennement stressée",
            "4": "Très stressée"
        }
    """

    unique, counts = np.unique(out, return_counts=True)

    print("\nDistribution des classes de stress :")
    print("-" * 40)

    total = out.size

    for cls, count in zip(unique, counts):

        if cls == 0:
            class_name = "Fond / exclu"
        elif stress_class_names is not None:
            class_name = stress_class_names.get(str(cls), f"Classe {cls}")
        else:
            class_name = f"Classe {cls}"

        percent = (count / total) * 100

        print(
            f"Classe {cls} ({class_name}) : "
            f"{count:,} pixels ({percent:.2f}%)"
        )

# ============================================================
# 10) BOXPLOTS
# ============================================================

def plot_boxplots(indices_dict, group_mask, STRESS_CLASS_NAMES, valid_pixels_mask=None, ignore_labels=(0,), min_pixels=30):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    ordered_ids = [1, 2, 3, 4]
    ordered_labels = [STRESS_CLASS_NAMES[i] for i in ordered_ids]

    for index_name, index_map in indices_dict.items():
        data = []

        for stress_id in ordered_ids:
            mask = (
                (group_mask == stress_id) &
                valid_pixels_mask &
                np.isfinite(index_map)
            )
            values = index_map[mask]
            if values.size < min_pixels:
                data.append(np.array([np.nan], dtype=np.float32))
            else:
                data.append(values)

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, tick_labels=ordered_labels, showfliers=False)
        plt.title(f"{index_name.upper()} par niveau de stress")
        plt.ylabel(index_name.upper())
        plt.tight_layout()
        plt.show()


def print_class_distribution_named(y, id_to_class, name="dataset"):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\nDistribution des classes dans {name} :")

    for c, n in zip(classes, counts):
        class_name = id_to_class.get(c, str(c))
        pct = 100 * n / total
        print(f"{class_name:25s} : {n:8d} ({pct:5.2f}%)")

def print_spatial_cv_results(cv_results, class_name_map=None):
    print("\n=== Spatial Block CV - Résumé global ===")
    print(f"Nombre de folds       : {cv_results['n_splits']}")
    print(f"Taille des blocs      : {cv_results['block_size']} px")
    print(f"Blocs utilisés        : {cv_results['n_blocks_used']}")
    print(f"Accuracy moyenne      : {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Kappa moyen           : {cv_results['mean_kappa']:.4f} ± {cv_results['std_kappa']:.4f}")

    for fr in cv_results["fold_results"]:
        print(f"\n--- Fold {fr['fold']} ---")
        print(f"Train pixels          : {fr['n_train']}")
        print(f"Test pixels           : {fr['n_test']}")
        print(f"Train blocks          : {fr['n_train_blocks']}")
        print(f"Test blocks           : {fr['n_test_blocks']}")
        print(f"Accuracy              : {fr['accuracy']:.4f}")
        print(f"Kappa                 : {fr['kappa']:.4f}")

        print("\nMatrice de confusion :")
        print(fr["confusion_matrix"])

        print("\nClassification report :")
        print(fr["classification_report"])

        print("\nImportance des variables :")
        feature_names = ["NDVI", "NDRE", "NDWI", "PRI", "ARI", "EVI", "NBR"]
        for name, imp in zip(feature_names, fr["feature_importances"]):
            print(f"{name:6s} : {imp:.4f}")




def plot_spatial_cv_folds(rows, cols, image_shape, block_size=200, n_splits=5):
    """
    Visualise les blocs spatiaux utilisés pour la spatial cross-validation
    et montre quels blocs sont train/test pour chaque fold.
    """

    from matplotlib.colors import ListedColormap

    H, W = image_shape

    # ----------------------------------------------------
    # construire blocs
    # ----------------------------------------------------
    block_row = rows // block_size
    block_col = cols // block_size
    block_ids = block_row * 1000 + block_col

    unique_blocks = np.unique(block_ids)

    print("Nombre de blocs :", len(unique_blocks))

    gkf = GroupKFold(n_splits=n_splits)

    # ----------------------------------------------------
    # figure
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, n_splits, figsize=(5*n_splits, 5))

    for i, (train_idx, test_idx) in enumerate(
        gkf.split(np.zeros(len(rows)), np.zeros(len(rows)), groups=block_ids)
    ):

        ax = axes[i]

        img = np.zeros((H, W))

        # train = 1
        img[rows[train_idx], cols[train_idx]] = 1

        # test = 2
        img[rows[test_idx], cols[test_idx]] = 2

        #cmap = plt.cm.get_cmap("Set1", 3)

        cmap = ListedColormap([
            "white",   # 0 non utilisé
            "blue",    # 1 train
            "red"      # 2 test
        ])

        ax.imshow(img, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f"Fold {i+1}")
        ax.axis("off")

    plt.suptitle("Spatial Cross Validation (train / test)", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_gt_and_stress_masks(gt_mask, stress_mask, class_map, stress_class_names, output_path=None):

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

    gt_cmap = ListedColormap(GT_COLORS)
    stress_cmap = ListedColormap(STRESS_COLORS)

    id_to_class = {v: k for k, v in class_map.items()}  

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # -------------------------
    # Masque GT original
    # -------------------------
    axes[0].imshow(gt_mask, cmap=gt_cmap, vmin=0, vmax=max(id_to_class.keys()))
    axes[0].set_title("Masque vérité terrain (classes originales)")
    axes[0].axis("off")

    gt_legend_elements = []
    present_gt_labels = sorted(np.unique(gt_mask))
    for lab in present_gt_labels:
        if lab == 0:
            continue
        if lab in id_to_class:
            gt_legend_elements.append(
                Patch(facecolor=GT_COLORS[lab], edgecolor="black", label=id_to_class[lab])
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
            Patch(facecolor=STRESS_COLORS[lab], edgecolor="black", label=stress_class_names[lab])
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