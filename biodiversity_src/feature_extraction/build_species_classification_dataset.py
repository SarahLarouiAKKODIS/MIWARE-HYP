import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from utils import concatenate_similarity_best_maps
from utils import load_config, evaluate_classification_model, concatenate_tifs, concatenate_ace_maps, cube_to_dataset, subsample_by_class, plot_pca_by_class_and_group, concatenate_similarity_best_maps, compute_feature_statistics_by_class


INDICES_FEATURE_NAMES = [
    "NDVI", "NDRE", "NDWI", "PRI", "ARI", "EVI", "NBR"
]

BAND_DEPTH_FEATURE_NAMES = [
    "BD970", "BD1200", "BD1450", "BD1940", "BD2200"
]

DATE_FEATURE_NAMES = [
    "acquisition_day_sin",
    "acquisition_day_cos",
]


def load_raster_features(tif_path):
    """
    Charge un GeoTIFF multibande.

    Returns
    -------
    cube : np.ndarray
        Tableau (features, rows, cols).
    feature_names : list[str]
        Descriptions des bandes.
    """
    with rasterio.open(tif_path) as src:
        cube = src.read().astype(np.float32)
        feature_names = [
            description if description else f"band_{i + 1}"
            for i, description in enumerate(src.descriptions)
        ]

    return cube, feature_names


def acquisition_date_features(acquisition_date, height, width):
    """
    Encode la date d'acquisition en deux variables cycliques.

    La représentation sin/cos permet de tenir compte
    de la continuité entre décembre et janvier.
    """
    acquisition_date = date.fromisoformat(
        str(acquisition_date)[:10]
    )

    day_of_year = acquisition_date.timetuple().tm_yday
    days_in_year = (
        date(acquisition_date.year + 1, 1, 1)
        - date(acquisition_date.year, 1, 1)
    ).days

    angle = 2 * np.pi * (day_of_year - 1) / days_in_year

    day_sin = np.sin(angle)
    day_cos = np.cos(angle)

    date_cube = np.empty(
        (2, height, width),
        dtype=np.float32
    )

    date_cube[0] = day_sin
    date_cube[1] = day_cos

    return date_cube


def build_features_dataset(config_files, Path_config):
    """
    Construit un dataset de classification à partir
    de plusieurs images EnMAP et de leurs configurations.

    Paramètres
    ----------
    config_files : list[str | Path]
        Liste des fichiers de configuration JSON.

    Returns
    -------
    X_all : np.ndarray
        Matrice (n_pixels, n_features).

    y_all : np.ndarray
        Labels de classification.

    groups : np.ndarray
        Identifiant de l'image d'origine pour chaque pixel.

    feature_names : list[str]
        Liste ordonnée des caractéristiques.
    """

    X_list = []
    y_list = []
    groups_list = []

    feature_names = None

    if not config_files:
        raise ValueError("La liste config_files est vide.")

    for group_id, config_path in enumerate(config_files):

        config_path = Path(Path_config) / Path(config_path)

        print(f"\n{'=' * 60}")
        print(f"Configuration : {config_path}")
        print(f"{'=' * 60}")

        # =================================================
        # 1. CONFIGURATION
        # =================================================

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        path_res = Path(config["Path_res"])
        acquisition_date = config["acquisition_date"]

        feature_dir = path_res / "Feature_extraction"

        print(f"Date d'acquisition : {acquisition_date}")

        # =================================================
        # 2. INDICES DE VÉGÉTATION
        # =================================================

        with np.load(
            feature_dir / "dataset_indices_for_species_classif.npz"
        ) as data:

            X_ind = data["X"].astype(np.float32)
            label_mask = data["y"]

        # =================================================
        # 3. PROFONDEURS DES BANDES D'ABSORPTION
        # =================================================

        with np.load(
            feature_dir / "band_depth_cube.npz"
        ) as data:

            X_bd = data["X"].astype(np.float32)

        # =================================================
        # 4. CARTES DE SIMILARITÉ SPECTRALE ACE
        # =================================================

        similarity_cube, similarity_names = (
            concatenate_similarity_best_maps(
                input_dir=path_res / "Similarity_best_maps",
                output_path=feature_dir / "similarity_best_ace.tif",
                mode="ace",
            )
        )

        similarity_cube = similarity_cube.astype(np.float32)

        # print("Caractéristiques similarité spectrale :")
        # for i, name in enumerate(similarity_names, 1):
        #     print(f"{i}. {name}")

        # =================================================
        # 5. CARACTÉRISTIQUES TOPOGRAPHIQUES
        # =================================================

        topo_cube, topo_names = load_raster_features(
            feature_dir / "topographic_features.tif"
        )

        # print("Caractéristiques topographiques :")
        # for i, name in enumerate(topo_names, 1):
        #     print(f"{i}. {name}")

        # =================================================
        # 6. CARACTÉRISTIQUES CLIMATIQUES
        # =================================================

        acquisition_year = date.fromisoformat(
            str(acquisition_date)[:10]
        ).year

        climate_start_year = acquisition_year - 10
        climate_last_year = acquisition_year - 1

        climate_path = (
            feature_dir 
            / f"bioclim_features_{climate_start_year}_"
              f"{climate_last_year}.tif"
        )

        climate_cube, climate_names = load_raster_features(
            climate_path
        )
        # print("Caractéristiques climatiques :")
        # for i, name in enumerate(climate_names, 1):
        #     print(f"{i}. {name}")

        # =================================================
        # 7. CARACTÉRISTIQUES TEMPORELLES
        # =================================================

        _, height, width = X_ind.shape

        date_cube = acquisition_date_features(
            acquisition_date=acquisition_date,
            height=height,
            width=width,
        )

        # =================================================
        # 8. VÉRIFICATION DES DIMENSIONS
        # =================================================

        feature_cubes = {
            "indices": X_ind,
            "band_depth": X_bd,
            "similarity": similarity_cube,
            "topography": topo_cube,
            "climate": climate_cube,
            "date": date_cube,
        }

        for name, feature_cube in feature_cubes.items():

            if feature_cube.ndim != 3:
                raise ValueError(
                    f"{name} : cube attendu en 3 dimensions, "
                    f"reçu {feature_cube.shape}"
                )

            if feature_cube.shape[1:] != (height, width):
                raise ValueError(
                    f"{name} : dimensions incompatibles "
                    f"{feature_cube.shape[1:]} != {(height, width)}"
                )

        if label_mask.shape != (height, width):
            raise ValueError(
                f"Dimensions du masque de labels incorrectes : "
                f"{label_mask.shape} != {(height, width)}"
            )

        # =================================================
        # 9. CONCATÉNATION DES CARACTÉRISTIQUES
        # =================================================

        cube = np.concatenate(
            [
                X_ind,
                X_bd,
                similarity_cube,
                topo_cube,
                climate_cube,
                date_cube,
            ],
            axis=0,
        )

        current_feature_names = (
            INDICES_FEATURE_NAMES
            + BAND_DEPTH_FEATURE_NAMES
            + list(similarity_names)
            + topo_names
            + climate_names
            + DATE_FEATURE_NAMES
        )

        if cube.shape[0] != len(current_feature_names):
            raise ValueError(
                f"Incohérence : {cube.shape[0]} bandes "
                f"pour {len(current_feature_names)} noms."
            )

        # Vérifier la cohérence entre toutes les images
        if feature_names is None:
            feature_names = current_feature_names

        elif current_feature_names != feature_names:
            raise ValueError(
                f"Caractéristiques différentes pour {config_path}.\n"
                "L'ordre et les noms des features doivent être "
                "identiques entre toutes les images."
            )

        # =================================================
        # 10. CONSTRUCTION DU DATASET PAR PIXEL
        # =================================================

        X, y = cube_to_dataset(
            feature_cube=cube,
            label_mask=label_mask,
            valid_mask_path=path_res / "enmap_valid_mask.tif",
            ignore_labels=(0,),
        )

        X_list.append(X)
        y_list.append(y)

        groups_list.append(
            np.full(len(y), group_id, dtype=np.int32)
        )

        # print(f"Cube : {cube.shape}")
        # print(f"Pixels retenus : {len(y)}")
        # print(f"Nombre de caractéristiques : {X.shape[1]}")
        # print(f"Groupe : {group_id}")

    # =====================================================
    # 11. FUSION GLOBALE
    # =====================================================

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)

    print(f"\n{'=' * 60}")
    print("DATASET GLOBAL")
    print(f"{'=' * 60}")

    print(f"X_all : {X_all.shape}")
    print(f"y_all : {y_all.shape}")
    print(f"groups : {groups.shape}")
    print(f"Nombre de caractéristiques : {len(feature_names)}")

    return X_all, y_all, groups, feature_names