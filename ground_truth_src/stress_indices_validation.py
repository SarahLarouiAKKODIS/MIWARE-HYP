#!/usr/bin/env python3
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split


# ============================================================
# 0) CLASSES INITIALES
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
# 1) REGROUPEMENT EN 3 CLASSES DE "STRESS"
#    1 = peu stressée
#    2 = moyennement stressée
#    3 = très stressée
#
#    À AJUSTER selon ton expertise.
# ============================================================

STRESS_GROUPS = {
    1: [  # peu stressée / forte vigueur spectrale
        CLASS_MAP["Sapin, épicéa"],
        CLASS_MAP["Chênes sempervirents"],
        CLASS_MAP["Douglas"],
        CLASS_MAP["Pin laricio, pin noir"],
        CLASS_MAP["Pin sylvestre"],
    ],
    2: [  # intermédiaire
        CLASS_MAP["Conifères"],
        CLASS_MAP["Feuillus"],
        CLASS_MAP["Mixte"],
        CLASS_MAP["Pin autre"],
        CLASS_MAP["Pin d'Alep"],
        CLASS_MAP["Pins mélangés"],
        CLASS_MAP["Robinier"],
    ],
    3: [  # plus stressée / plus faible vigueur spectrale
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

EXCLUDED_ORIGINAL_LABELS = (0, CLASS_MAP["NC"], CLASS_MAP["NR"])


# ============================================================
# 2) LECTURE DES DONNÉES
# ============================================================

def read_hyperspectral_raster(image_path, scale_factor=10000.0, nodata_value=-32768):
    with rasterio.open(image_path) as src:
        cube = src.read().astype(np.float32)
        profile = src.profile

    cube[cube == nodata_value] = np.nan
    cube /= scale_factor
    return cube, profile


def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def read_wavelengths_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "wavelength_nm" not in df.columns:
        raise ValueError("La colonne 'wavelength_nm' est absente du CSV")
    return df["wavelength_nm"].to_numpy(dtype=np.float32)


def save_mask(output_path, mask, reference_profile, nodata=0):
    profile = reference_profile.copy()
    profile.update(count=1, dtype=rasterio.uint8, nodata=nodata)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


# ============================================================
# 3) MASQUE DE VALIDITÉ ENMAP
# ============================================================

def build_enmap_valid_mask(
    cirrus_mask,
    cloud_mask,
    haze_mask,
    cloudshadow_mask,
    snow_mask,
    testflags_mask,
    scene_mask,
):
    return (
        (scene_mask == 1) &
        (cirrus_mask == 0) &
        (cloud_mask == 0) &
        (haze_mask == 0) &
        (cloudshadow_mask == 0) &
        (snow_mask == 0) &
        (testflags_mask == 0)
    )


# ============================================================
# 4) INDICES
# ============================================================

def safe_divide(num, den):
    out = np.full(num.shape, np.nan, dtype=np.float32)
    valid = np.abs(den) > 1e-10
    out[valid] = num[valid] / den[valid]
    return out


def find_nearest_band(wavelengths, target_nm):
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def compute_spectral_indices(cube, wavelengths):
    i_green = find_nearest_band(wavelengths, 550)
    i_red = find_nearest_band(wavelengths, 670)
    i_rededge = find_nearest_band(wavelengths, 720)
    i_nir = find_nearest_band(wavelengths, 800)
    i_swir = find_nearest_band(wavelengths, 1240)

    print("Bandes utilisées :")
    print(f"  Green   -> {wavelengths[i_green]:.2f} nm")
    print(f"  Red     -> {wavelengths[i_red]:.2f} nm")
    print(f"  RedEdge -> {wavelengths[i_rededge]:.2f} nm")
    print(f"  NIR     -> {wavelengths[i_nir]:.2f} nm")
    print(f"  SWIR    -> {wavelengths[i_swir]:.2f} nm")

    green = cube[i_green]
    red = cube[i_red]
    rededge = cube[i_rededge]
    nir = cube[i_nir]
    swir = cube[i_swir]

    ndvi = safe_divide(nir - red, nir + red)
    ndre = safe_divide(nir - rededge, nir + rededge)
    gndvi = safe_divide(nir - green, nir + green)
    ndwi = safe_divide(nir - swir, nir + swir)  # plutôt NDMI-like

    return {
        "ndvi": ndvi,
        "ndre": ndre,
        "gndvi": gndvi,
        "ndwi": ndwi,
    }


# ============================================================
# 5) RECLASSIFICATION DU MASQUE GT EN 3 CLASSES
# ============================================================

def reclassify_gt_to_stress_classes(gt_mask, stress_groups, excluded_labels=()):
    """
    Reclassifie le masque GT espèces -> 3 classes de stress.
    Les pixels non mappés restent à 0.
    """
    out = np.zeros_like(gt_mask, dtype=np.uint8)

    for stress_id, original_ids in stress_groups.items():
        for original_id in original_ids:
            out[gt_mask == original_id] = stress_id

    for lab in excluded_labels:
        out[gt_mask == lab] = 0

    return out


# ============================================================
# 6) EXTRACTION DES PIXELS
# ============================================================


def extract_pixels(indices_dict, target_mask, valid_pixels_mask=None, ignore_labels=(0,)):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(target_mask.shape, dtype=bool)

    valid = (
        valid_pixels_mask &
        np.isfinite(indices_dict["ndvi"]) &
        np.isfinite(indices_dict["ndre"]) &
        np.isfinite(indices_dict["gndvi"]) &
        np.isfinite(indices_dict["ndwi"])
    )

    for lab in ignore_labels:
        valid &= (target_mask != lab)

    # coordonnées pixels
    rows, cols = np.where(valid)

    # features
    X = np.stack([
        indices_dict["ndvi"][valid],
        indices_dict["ndre"][valid],
        indices_dict["gndvi"][valid],
        indices_dict["ndwi"][valid],
    ], axis=1).astype(np.float32)

    # labels
    y = target_mask[valid].astype(np.int32)

    return X, y, rows, cols


# ============================================================
# 7) STATS PAR CLASSE
# ============================================================

def compute_mean_indices_per_group(indices_dict, group_mask, valid_pixels_mask=None, ignore_labels=(0,)):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    stats = {}

    for stress_id, stress_name in STRESS_CLASS_NAMES.items():
        if stress_id in ignore_labels:
            continue

        mask = (
            (group_mask == stress_id) &
            valid_pixels_mask &
            np.isfinite(indices_dict["ndvi"]) &
            np.isfinite(indices_dict["ndre"]) &
            np.isfinite(indices_dict["gndvi"]) &
            np.isfinite(indices_dict["ndwi"])
        )

        n = np.count_nonzero(mask)
        if n == 0:
            continue

        stats[stress_id] = {
            "class_name": stress_name,
            "n_pixels": int(n),
            "ndvi": float(np.nanmean(indices_dict["ndvi"][mask])),
            "ndre": float(np.nanmean(indices_dict["ndre"][mask])),
            "gndvi": float(np.nanmean(indices_dict["gndvi"][mask])),
            "ndwi": float(np.nanmean(indices_dict["ndwi"][mask])),
            "ndvi_std": float(np.nanstd(indices_dict["ndvi"][mask])),
            "ndre_std": float(np.nanstd(indices_dict["ndre"][mask])),
            "gndvi_std": float(np.nanstd(indices_dict["gndvi"][mask])),
            "ndwi_std": float(np.nanstd(indices_dict["ndwi"][mask])),
        }

    return stats


# ============================================================
# 8) COHÉRENCE ATTENDUE DES INDICES
# ============================================================

def check_stress_coherence(stats):
    """
    Vérifie la cohérence attendue :
    - NDVI : peu stressée > moyenne > très stressée
    - NDRE : peu stressée > moyenne > très stressée
    - GNDVI : peu stressée > moyenne > très stressée
    - NDWI (ta formule NIR-SWIR / NIR+SWIR) :
      on attend généralement peu stressée > moyenne > très stressée
      (plus proche de 0 si végétation moins stressée)
    """
    required_ids = [1, 2, 3]
    if not all(i in stats for i in required_ids):
        print("\nImpossible de tester la cohérence : une ou plusieurs classes manquent.")
        return

    print("\n=== Vérification de cohérence des indices ===")

    for index_name in ["ndvi", "ndre", "gndvi", "ndwi"]:
        v1 = stats[1][index_name]
        v2 = stats[2][index_name]
        v3 = stats[3][index_name]

        coherent = (v1 > v2 > v3)

        print(
            f"{index_name.upper():5s} | "
            f"Peu stressée={v1:.4f} | "
            f"Moyennement stressée={v2:.4f} | "
            f"Très stressée={v3:.4f} | "
            f"Cohérent={coherent}"
        )


# ============================================================
# 9) KRUSKAL-WALLIS
# ============================================================

def kruskal_tests(indices_dict, group_mask, valid_pixels_mask=None, ignore_labels=(0,), min_pixels=30):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    results = {}

    for index_name, index_map in indices_dict.items():
        groups = []

        for stress_id in sorted(STRESS_CLASS_NAMES.keys()):
            if stress_id in ignore_labels:
                continue

            mask = (
                (group_mask == stress_id) &
                valid_pixels_mask &
                np.isfinite(index_map)
            )

            values = index_map[mask]
            if values.size >= min_pixels:
                groups.append(values)

        if len(groups) < 2:
            results[index_name] = None
            continue

        stat, pvalue = kruskal(*groups)
        results[index_name] = {
            "statistic": float(stat),
            "pvalue": float(pvalue),
        }

    return results


# ============================================================
# 10) BOXPLOTS
# ============================================================

def plot_boxplots(indices_dict, group_mask, valid_pixels_mask=None, ignore_labels=(0,), min_pixels=30):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    ordered_ids = [1, 2, 3]
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


def train_test_spatial_split(X, y, rows, cols, image_shape):

    H, W = image_shape
    split_row = H // 2

    train_mask = rows < split_row
    test_mask = rows >= split_row

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Le split spatial a produit un ensemble train ou test vide.")
    
    return X_train, X_test, y_train, y_test


# ============================================================
# 11) RANDOM FOREST
# ============================================================

def run_random_forest_classification(X_train, X_test, y_train, y_test):
    

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = [1, 2, 3]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "kappa": cohen_kappa_score(y_test, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=[STRESS_CLASS_NAMES[l] for l in labels],
            zero_division=0,
            output_dict=False
        ),
        "feature_importances": clf.feature_importances_,
    }

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


def spatial_block_cv_random_forest(
    X,
    y,
    rows,
    cols,
    n_splits=5,
    block_size=200,
    min_pixels_per_block=50,
    random_state=42,
    n_estimators=300,
):
    """
    Spatial cross-validation robuste par blocs pour classification raster.

    Paramètres
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Features par pixel.
    y : np.ndarray, shape (n_samples,)
        Labels par pixel.
    rows : np.ndarray, shape (n_samples,)
        Coordonnées image: lignes.
    cols : np.ndarray, shape (n_samples,)
        Coordonnées image: colonnes.
    n_splits : int
        Nombre de folds GroupKFold.
    block_size : int
        Taille des blocs spatiaux en pixels.
    min_pixels_per_block : int
        Nombre minimal de pixels d'un bloc pour être conservé.
    random_state : int
        Graine aléatoire du Random Forest.
    n_estimators : int
        Nombre d'arbres du Random Forest.

    Retour
    ------
    results : dict
        Contient :
        - "fold_results" : liste des résultats par fold
        - "mean_accuracy"
        - "std_accuracy"
        - "mean_kappa"
        - "std_kappa"
        - "n_blocks_used"
        - "block_ids"
    """

    if not (len(X) == len(y) == len(rows) == len(cols)):
        raise ValueError("X, y, rows et cols doivent avoir la même longueur")

    # ------------------------------------------------------------
    # 1) Construction des identifiants de blocs
    # ------------------------------------------------------------
    block_row = rows // block_size
    block_col = cols // block_size

    # identifiant unique de bloc
    block_ids = block_row.astype(np.int64) * 10**6 + block_col.astype(np.int64)

    # ------------------------------------------------------------
    # 2) Filtrage des blocs trop petits
    # ------------------------------------------------------------
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    valid_blocks = unique_blocks[counts >= min_pixels_per_block]

    keep = np.isin(block_ids, valid_blocks)

    X_f = X[keep]
    y_f = y[keep]
    rows_f = rows[keep]
    cols_f = cols[keep]
    groups_f = block_ids[keep]

    unique_groups = np.unique(groups_f)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Pas assez de blocs valides pour {n_splits} folds. "
            f"Blocs disponibles: {len(unique_groups)}"
        )

    # ------------------------------------------------------------
    # 3) Cross-validation par groupes spatiaux
    # ------------------------------------------------------------
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    all_labels = sorted(np.unique(y_f))

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_f, y_f, groups=groups_f), start=1):
        X_train = X_f[train_idx]
        y_train = y_f[train_idx]
        X_test = X_f[test_idx]
        y_test = y_f[test_idx]

        train_groups = np.unique(groups_f[train_idx])
        test_groups = np.unique(groups_f[test_idx])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_result = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_blocks": len(train_groups),
            "n_test_blocks": len(test_groups),
            "train_blocks": train_groups,
            "test_blocks": test_groups,
            "accuracy": accuracy_score(y_test, y_pred),
            "kappa": cohen_kappa_score(y_test, y_pred, labels=all_labels),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=all_labels),
            "labels": all_labels,
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=all_labels,
                zero_division=0,
                output_dict=False,
            ),
            "feature_importances": clf.feature_importances_,
        }

        fold_results.append(fold_result)

    accuracies = np.array([fr["accuracy"] for fr in fold_results], dtype=np.float32)
    kappas = np.array([fr["kappa"] for fr in fold_results], dtype=np.float32)

    results = {
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "n_blocks_used": int(len(unique_groups)),
        "block_ids": unique_groups,
        "block_size": block_size,
        "min_pixels_per_block": min_pixels_per_block,
        "n_splits": n_splits,
    }

    return results

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
        feature_names = ["NDVI", "NDRE", "GNDVI", "NDWI"]
        for name, imp in zip(feature_names, fr["feature_importances"]):
            print(f"{name:6s} : {imp:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold


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
# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":
    
    image_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"
    gt_mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Forest/mask_foret_classes.tif"
    output_group_mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Forest/mask_gt_stress_3classes.tif"
    # 6 masques qualité EnMAP
    path_mask = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/"
    cirrus_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CIRRUS.TIF"
    cloud_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUD.TIF"
    haze_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_HAZE.TIF"
    cloudshadow_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUDSHADOW.TIF"
    snow_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_SNOW.TIF"
    testflags_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_TESTFLAGS.TIF"

    # masque scène : 0 unclassified, 1 land, 2 water, 3 no data
    scene_mask_path = path_mask + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLASSES.TIF"

    wavelengths_csv = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full2.csv"
   

    # Lecture
    cube, profile = read_hyperspectral_raster(image_path)
    wavelengths = read_wavelengths_from_csv(wavelengths_csv)
    gt_mask = read_mask(gt_mask_path)

    cirrus_mask = read_mask(cirrus_mask_path)
    cloud_mask = read_mask(cloud_mask_path)
    haze_mask = read_mask(haze_mask_path)
    cloudshadow_mask = read_mask(cloudshadow_mask_path)
    snow_mask = read_mask(snow_mask_path)
    testflags_mask = read_mask(testflags_mask_path)
    scene_mask = read_mask(scene_mask_path)

    print("cube shape:", cube.shape)
    print("min cube:", np.nanmin(cube))
    print("max cube:", np.nanmax(cube))
    print("Nombre wavelengths:", len(wavelengths))

    if len(wavelengths) != cube.shape[0]:
        raise ValueError(
            f"Incohérence bandes : {len(wavelengths)} wavelengths vs {cube.shape[0]} bandes"
        )

    # Masque de validité EnMAP
    enmap_valid_mask = build_enmap_valid_mask(
        cirrus_mask=cirrus_mask,
        cloud_mask=cloud_mask,
        haze_mask=haze_mask,
        cloudshadow_mask=cloudshadow_mask,
        snow_mask=snow_mask,
        testflags_mask=testflags_mask,
        scene_mask=scene_mask,
    )

    print("Pixels valides :", np.count_nonzero(enmap_valid_mask))
    print("Pixels totaux  :", enmap_valid_mask.size)

    # Indices
    indices = compute_spectral_indices(cube, wavelengths)

    # Reclassification GT -> 3 classes
    stress_mask = reclassify_gt_to_stress_classes(
        gt_mask,
        STRESS_GROUPS,
        excluded_labels=EXCLUDED_ORIGINAL_LABELS
    )

    save_mask(output_group_mask_path, stress_mask, profile)
    print("Masque 3 classes sauvegardé :", output_group_mask_path)

    # Stats par groupe
    stats = compute_mean_indices_per_group(
        indices,
        stress_mask,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=(0,)
    )

    print("\n=== Moyennes des indices par classe de stress ===")
    for stress_id in sorted(stats.keys()):
        vals = stats[stress_id]
        print(
            f"{stress_id} - {vals['class_name']:25s} | "
            f"n={vals['n_pixels']:7d} | "
            f"NDVI={vals['ndvi']:.4f} ± {vals['ndvi_std']:.4f} | "
            f"NDRE={vals['ndre']:.4f} ± {vals['ndre_std']:.4f} | "
            f"GNDVI={vals['gndvi']:.4f} ± {vals['gndvi_std']:.4f} | "
            f"NDWI={vals['ndwi']:.4f} ± {vals['ndwi_std']:.4f}"
        )

    # Vérification de cohérence
    check_stress_coherence(stats)

    # Kruskal-Wallis
    kw_results = kruskal_tests(
        indices,
        stress_mask,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=(0,),
        min_pixels=30
    )

    print("\n=== Tests de Kruskal-Wallis ===")
    for idx_name, res in kw_results.items():
        if res is None:
            print(f"{idx_name.upper()} : test impossible")
            continue
        print(f"{idx_name.upper()} | H={res['statistic']:.4f} | p={res['pvalue']:.4e}")

    # Boxplots
    plot_boxplots(
        indices,
        stress_mask,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=(0,),
        min_pixels=30
    )

    # Random Forest
    X, y, rows, cols = extract_pixels(
    indices_dict=indices,
    target_mask=stress_mask,
    valid_pixels_mask=enmap_valid_mask,
    ignore_labels=(0,)
    )

    print("\nForme de X :", X.shape)
    print("Forme de y :", y.shape)

    ########### Séparation aléatoire stratifée 

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.3,
    #     random_state=42,
    #     stratify=y
    # )

    ########### Spatial split

    # X_train, X_test, y_train, y_test =  train_test_spatial_split(X, y, rows, cols, gt_mask.shape)
    
    # print_class_distribution_named(y_train, STRESS_CLASS_NAMES, "train")
    # print_class_distribution_named(y_test, STRESS_CLASS_NAMES, "test")

    # rf_results = run_random_forest_classification(X_train, X_test, y_train, y_test)

    ############ Spatial cros-validation par blocs

    n_splits = 5
    block_size = 200
    min_pixels_per_block = 100

    plot_spatial_cv_folds(
    rows=rows,
    cols=cols,
    image_shape=gt_mask.shape,
    block_size=block_size,
    n_splits=n_splits
)

    cv_results = spatial_block_cv_random_forest(
    X=X,
    y=y,
    rows=rows,
    cols=cols,
    n_splits=5,
    block_size=200,
    min_pixels_per_block=100,
    random_state=42,
    n_estimators=300,
    )

    print_spatial_cv_results(cv_results)

    # print("\n=== Random Forest sur 3 classes de stress ===")
    # print(f"Accuracy : {rf_results['accuracy']:.4f}")
    # print(f"Kappa    : {rf_results['kappa']:.4f}")

    # print("\nMatrice de confusion :")
    # print(rf_results["confusion_matrix"])

    # print("\nClassification report :")
    # print(rf_results["classification_report"])

    # print("\nImportance des indices :")
    # feature_names = ["NDVI", "NDRE", "GNDVI", "NDWI"]
    # for name, imp in zip(feature_names, rf_results["feature_importances"]):
    #     print(f"{name:6s} : {imp:.4f}")