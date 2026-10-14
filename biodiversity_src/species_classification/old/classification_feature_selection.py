

from pathlib import Path
import numpy as np
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import rasterio

from utils import cube_to_dataset, concatenate_similarity_best_maps, sequential_forward_selection_xgb, sequential_floating_forward_selection_xgb

config_path = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/")

config_files = [
    config_path / "salsigne_3.json", #0
    config_path / "salsigne_mars_2.json", #1
    config_path / "salsigne_mars_3.json", #2
    config_path / "salsigne.json", #3
    config_path / "salau_1.json", #4
    config_path / "salau_2.json", #5
    config_path / "cma_1.json", #6
    config_path / "cma.json", #7
    config_path / "abbaretz.json", #8
    config_path / "abbaretz_3.json" #9
]



# ============================================================
# PARAMÈTRES
# ============================================================

INDICES_FEATURE_NAMES = [
    "NDVI", "NDRE", "NDWI", "PRI", "ARI", "EVI", "NBR"
]

BAND_DEPTH_FEATURE_NAMES = [
    "BD970", "BD1200", "BD1450", "BD1940", "BD2200"
]

EXCLUDED_CLASSES = [10, 11, 7, 5, 9, 13, 17, 16, 19, 20, 21]

TEST_GROUP_IDS = [4, 5, 6, 7]


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def read_multiband_tif(tif_path):
    """Lit un GeoTIFF multibande et récupère les noms des bandes."""
    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        cube = src.read().astype(np.float32)
        names = [
            desc if desc is not None else f"band_{i+1}"
            for i, desc in enumerate(src.descriptions)
        ]

    return cube, names


def print_cube_stats(cube, names, title):
    """Affiche les statistiques min/max/moyenne de chaque bande."""
    print(f"\n--- {title} ---")

    for i, name in enumerate(names):
        print(
            name,
            "min =", np.nanmin(cube[i]),
            "max =", np.nanmax(cube[i]),
            "mean =", np.nanmean(cube[i]),
        )


def check_spatial_shapes(cubes, names):
    """Vérifie que tous les cubes ont les mêmes dimensions spatiales."""
    shapes = [cube.shape[1:] for cube in cubes]

    if len(set(shapes)) != 1:
        raise ValueError(
            "Dimensions spatiales incompatibles :\n"
            + "\n".join([f"{n}: {s}" for n, s in zip(names, shapes)])
        )


# ============================================================
# CONSTRUCTION DU DATASET GLOBAL
# ============================================================

X_list = []
y_list = []
groups = []

feature_names_reference = None

for group_id, config_path in enumerate(config_files):

    print("\n===================================================")
    print("Config :", config_path)
    print("Group ID :", group_id)
    print("===================================================")

    with open(config_path) as f:
        config = json.load(f)

    Path_res = Path(config["Path_res"])
    feature_dir = Path_res / "Feature_extraction"

    # --------------------------------------------------------
    # 1. Indices spectraux
    # --------------------------------------------------------

    data_indices = np.load(feature_dir / "dataset_indices_for_species_classif.npz")
    X_ind = data_indices["X"].astype(np.float32)
    y_mask = data_indices["y"]

    # --------------------------------------------------------
    # 2. Band Depth
    # --------------------------------------------------------

    data_band_depth = np.load(feature_dir / "band_depth_cube.npz")
    X_bd = data_band_depth["X"].astype(np.float32)

    # --------------------------------------------------------
    # 3. Similarité spectrale ACE
    # --------------------------------------------------------

    similarity_cube, similarity_feature_names = concatenate_similarity_best_maps(
        input_dir=Path_res / "Similarity_best_maps",
        output_path=feature_dir / "similarity_best_ace.tif",
        mode="ace",
    )

    similarity_cube = similarity_cube.astype(np.float32)

    # --------------------------------------------------------
    # 4. Topographie
    # --------------------------------------------------------

    topo_cube, topo_feature_names = read_multiband_tif(
        feature_dir / "topographic_features.tif"
    )

    # --------------------------------------------------------
    # 5. Climat
    # --------------------------------------------------------

    climate_cube, climate_feature_names = read_multiband_tif(
        feature_dir / "climate_features_2000_2024.tif"
    )

    print_cube_stats(
        climate_cube,
        climate_feature_names,
        title="Statistiques climatiques",
    )

    # --------------------------------------------------------
    # 6. Vérification dimensions
    # --------------------------------------------------------

    cubes = [
        X_ind,
        X_bd,
        similarity_cube,
        topo_cube,
        climate_cube,
    ]

    cube_names = [
        "indices",
        "band_depth",
        "similarity",
        "topography",
        "climate",
    ]

    check_spatial_shapes(cubes, cube_names)

    # --------------------------------------------------------
    # 7. Construction du cube final
    # --------------------------------------------------------

    cube = np.concatenate(cubes, axis=0)

    feature_names = (
        INDICES_FEATURE_NAMES
        + BAND_DEPTH_FEATURE_NAMES
        + list(similarity_feature_names)
        + list(topo_feature_names)
        + list(climate_feature_names)
    )

    if cube.shape[0] != len(feature_names):
        raise ValueError(
            f"Incohérence features : cube={cube.shape[0]}, "
            f"feature_names={len(feature_names)}"
        )

    # Sécurité : l'ordre et le nombre de features doivent être identiques entre sites
    if feature_names_reference is None:
        feature_names_reference = feature_names.copy()
    else:
        if feature_names != feature_names_reference:
            raise ValueError(
                "Les feature_names ne sont pas identiques entre sites."
            )

    print("Cube final :", cube.shape)
    print("Nombre de features :", len(feature_names))

    # --------------------------------------------------------
    # 8. Conversion cube -> dataset pixels
    # --------------------------------------------------------

    X, y = cube_to_dataset(
        feature_cube=cube,
        label_mask=y_mask,
        valid_mask_path=Path_res / "enmap_valid_mask.tif",
        ignore_labels=(0,),
    )

    X_list.append(X)
    y_list.append(y)
    groups.extend([group_id] * len(y))

    print("X site :", X.shape)
    print("y site :", y.shape)


# ============================================================
# FUSION DES SITES
# ============================================================

X_all = np.concatenate(X_list, axis=0).astype(np.float32)
y_all = np.concatenate(y_list, axis=0)
groups = np.asarray(groups)

feature_names = feature_names_reference

print("\n===================================================")
print("DATASET GLOBAL")
print("===================================================")
print("X_all :", X_all.shape)
print("y_all :", y_all.shape)
print("groups :", groups.shape)
print("Nombre de features :", len(feature_names))


# ============================================================
# SPLIT TRAIN / TEST PAR GROUPES
# ============================================================

is_test = np.isin(groups, TEST_GROUP_IDS)
is_train = ~is_test

print("group_train :", np.unique(groups[is_train]))
print("group_test  :", np.unique(groups[is_test]))

X_train_all = X_all[is_train]
y_train_all = y_all[is_train]

X_test = X_all[is_test]
y_test = y_all[is_test]


# ============================================================
# EXCLUSION DES CLASSES RARES / NON PERTINENTES
# ============================================================

keep_train = ~np.isin(y_train_all, EXCLUDED_CLASSES)
keep_test = ~np.isin(y_test, EXCLUDED_CLASSES)

X_train_all = X_train_all[keep_train]
y_train_all = y_train_all[keep_train]

X_test = X_test[keep_test]
y_test = y_test[keep_test]

print("\nClasses train :", np.unique(y_train_all, return_counts=True))
print("Classes test  :", np.unique(y_test, return_counts=True))


# ============================================================
# SEQUENTIAL FORWARD SELECTION
# ============================================================
max_per_class=30000
min_improvement=1e-6
max_features=None

# ============================================================
# SEQUENTIAL FLOATING FORWARD SELECTION
# ============================================================
(
    best_indices,
    best_features,
    best_score,
    sffs_history,
    sffs_progression,
    sffs_best_by_size,
    sffs_summary,
) = sequential_floating_forward_selection_xgb(
    X_train_all=X_train_all,
    y_train_all=y_train_all,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    max_features=None,
    min_improvement=min_improvement,
    max_per_class=max_per_class,
    random_state=42,
)

# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


out_dir = (
    Path_res.parent
    / f"SFFS_XGBoost_{timestamp}"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

sffs_history.to_csv(
    out_dir / "sffs_history.csv",
    index=False,
)

sffs_progression.to_csv(
    out_dir / "sffs_progression.csv",
    index=False,
)

sffs_best_by_size.to_csv(
    out_dir / "sffs_best_by_size.csv",
    index=False,
)

sffs_summary.to_csv(
    out_dir / "sffs_summary.csv",
    index=False,
)

print("Résultats sauvegardés dans :", out_dir)


params = {
    "test_groups": TEST_GROUP_IDS,
    "excluded_classes": EXCLUDED_CLASSES,
    "max_per_class": max_per_class,
    "min_improvement": min_improvement,
    "max_features": max_features,
}

with open(out_dir / "parameters.json", "w") as f:
    json.dump(params, f, indent=4)