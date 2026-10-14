
### Le script `main_biodiv_indice_calculation.py` doit être exécuté
# préalablement afin de générer les indices utilisés pour la classification.
#
# Validation croisée stratifiée en 4 folds :
# - 3/4 des pixels pour l'apprentissage ;
# - 1/4 des pixels pour le test ;
# - les pixels sont mélangés indépendamment de leur image d'origine ;
# - chaque pixel est utilisé une fois dans le test.

from pathlib import Path
import gc
import json
import sys

import numpy as np
import pandas as pd
import rasterio

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ============================================================
# IMPORTS DU PROJET
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils import (
    concatenate_similarity_best_maps,
    cube_to_dataset,
    evaluate_classification_model,
    subsample_by_class,
)


# ============================================================
# PARAMÈTRES
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 4
MAX_SAMPLES_PER_CLASS = 30000

# Mettre False si l'évaluation graphique de chaque fold est trop lourde.
RUN_DETAILED_EVALUATION_EACH_FOLD = True

excluded_classes = [
    5,
    7,
    9,
    10,
    11,
    13,
    16,
    17,
    19,
    20,
    21,
]


# ============================================================
# FICHIERS DE CONFIGURATION
# ============================================================

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
# NOMS DES VARIABLES
# ============================================================

indices_feature_names = [
    "NDVI",
    "NDRE",
    "NDWI",
    "PRI",
    "ARI",
    "EVI",
    "NBR",
]

band_depth_feature_names = [
    "BD970",
    "BD1200",
    "BD1450",
    "BD1940",
    "BD2200",
]

topo_feature_names = [
    "altitude",
    "slope",
    "aspect",
    "tpi",
    "twi",
    "tri",
    "curvature",
]

climate_feature_names = [
    "mean_temperature",
    "annual_precipitation",
    "summer_water_deficit",
    "annual_pet",
    "temperature_seasonality",
    "precipitation_seasonality",
]


# ============================================================
# VARIABLES UTILISÉES PAR XGBOOST
# ============================================================

classifier_feature_indices = [
    0, 1, 3, 4, 5,
    7, 8, 9, 10, 11,
    12,
    14, 15, 16,
    18, 19, 20,
    21,
]

classifier_feature_names = [
    "NDVI",
    "NDRE",
    "PRI",
    "ARI",
    "EVI",
    "BD970",
    "BD1200",
    "BD1450",
    "BD1940",
    "BD2200",
    "ace_max",
    "altitude",
    "slope",
    "aspect",
    "twi",
    "tri",
    "curvature",
    "mean_temperature",
]


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def safe_raster_descriptions(src, fallback_names):
    """
    Utilise les descriptions des bandes lorsqu'elles existent.
    Sinon, utilise les noms de secours.
    """
    descriptions = list(src.descriptions)

    if (
        len(descriptions) == src.count
        and all(name is not None and str(name).strip() for name in descriptions)
    ):
        return [str(name) for name in descriptions]

    if len(fallback_names) != src.count:
        raise ValueError(
            f"Le raster contient {src.count} bandes, mais la liste de secours "
            f"contient {len(fallback_names)} noms."
        )

    return list(fallback_names)


def class_count_table(y_train, y_test):
    """Construit un tableau des effectifs par classe."""
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()

    return (
        pd.DataFrame({
            "train": train_counts,
            "test": test_counts,
        })
        .fillna(0)
        .astype(int)
    )


def image_count_table(groups_train, groups_test):
    """
    Indique combien de pixels de chaque image sont présents dans le train
    et dans le test. Les images ne sont pas utilisées pour le découpage.
    """
    train_counts = pd.Series(groups_train).value_counts().sort_index()
    test_counts = pd.Series(groups_test).value_counts().sort_index()

    return (
        pd.DataFrame({
            "train": train_counts,
            "test": test_counts,
        })
        .fillna(0)
        .astype(int)
    )


def build_xgb_classifier():
    """Crée un nouveau classifieur indépendant pour chaque fold."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )


# ============================================================
# 1. CONSTRUCTION DU DATASET GLOBAL
# ============================================================

X_list = []
y_list = []
groups_list = []
group_metadata = []

reference_feature_names = None

for group_id, current_config_path in enumerate(config_files):

    print("\n" + "=" * 80)
    print("Configuration :", current_config_path)
    print("Identifiant de l'image :", group_id)
    print("=" * 80)

    with open(current_config_path, encoding="utf-8") as file:
        config = json.load(file)

    path_res = Path(config["Path_res"])

    # --------------------------------------------------------
    # Indices de végétation
    # --------------------------------------------------------
    data_indices = np.load(
        path_res / "Feature_extraction/dataset_indices_for_species_classif.npz"
    )
    X_ind = data_indices["X"]
    label_mask = data_indices["y"]

    # --------------------------------------------------------
    # Profondeurs de bandes
    # --------------------------------------------------------
    data_band_depth = np.load(
        path_res / "Feature_extraction/band_depth_cube.npz"
    )
    X_bd = data_band_depth["X"]

    # --------------------------------------------------------
    # Similarité spectrale
    # --------------------------------------------------------
    similarity_cube, similarity_feature_names = (
        concatenate_similarity_best_maps(
            input_dir=path_res / "Similarity_best_maps",
            output_path=(
                path_res / "Feature_extraction/similarity_best_ace.tif"
            ),
            mode="ace",
        )
    )

    # --------------------------------------------------------
    # Topographie
    # --------------------------------------------------------
    with rasterio.open(
        path_res / "Feature_extraction/topographic_features.tif"
    ) as src:
        topo_cube = src.read().astype(np.float32)
        current_topo_feature_names = safe_raster_descriptions(
            src,
            topo_feature_names,
        )

    # --------------------------------------------------------
    # Climat
    # --------------------------------------------------------
    with rasterio.open(
        path_res / "Feature_extraction/climate_features_2000_2024.tif"
    ) as src:
        climate_cube = src.read().astype(np.float32)
        current_climate_feature_names = safe_raster_descriptions(
            src,
            climate_feature_names,
        )

    for i, name in enumerate(current_climate_feature_names):
        print(
            f"{name}: "
            f"min={np.nanmin(climate_cube[i]):.6f}, "
            f"max={np.nanmax(climate_cube[i]):.6f}, "
            f"moyenne={np.nanmean(climate_cube[i]):.6f}"
        )

    # --------------------------------------------------------
    # Vérification des dimensions spatiales
    # --------------------------------------------------------
    spatial_shape = X_ind.shape[1:]

    assert similarity_cube.shape[1:] == spatial_shape
    assert X_bd.shape[1:] == spatial_shape
    assert topo_cube.shape[1:] == spatial_shape
    assert climate_cube.shape[1:] == spatial_shape

    # --------------------------------------------------------
    # Concaténation du cube
    # --------------------------------------------------------
    cube = np.concatenate(
        (
            X_ind,
            X_bd,
            similarity_cube,
            topo_cube,
            climate_cube,
        ),
        axis=0,
    )

    current_feature_names = (
        indices_feature_names
        + band_depth_feature_names
        + list(similarity_feature_names)
        + current_topo_feature_names
        + current_climate_feature_names
    )

    assert cube.shape[0] == len(current_feature_names), (
        f"Le cube contient {cube.shape[0]} variables, mais "
        f"{len(current_feature_names)} noms ont été définis."
    )

    if reference_feature_names is None:
        reference_feature_names = current_feature_names
    else:
        assert current_feature_names == reference_feature_names, (
            "Les noms ou l'ordre des variables diffèrent entre les images."
        )

    print("Dimensions du cube :", cube.shape)

    # --------------------------------------------------------
    # Conversion en pixels × variables
    # --------------------------------------------------------
    X_image, y_image = cube_to_dataset(
        feature_cube=cube,
        label_mask=label_mask,
        valid_mask_path=path_res / "enmap_valid_mask.tif",
        ignore_labels=(0,),
    )

    if len(X_image) == 0:
        raise ValueError(
            f"Aucun pixel valide n'a été extrait pour {current_config_path}."
        )

    X_list.append(X_image)
    y_list.append(y_image)

    # Le groupe est conservé uniquement pour les diagnostics.
    # Il n'intervient pas dans StratifiedKFold.
    groups_list.append(
        np.full(
            len(y_image),
            group_id,
            dtype=np.int32,
        )
    )

    group_metadata.append({
        "group_id": group_id,
        "config_path": str(current_config_path),
        "path_res": str(path_res),
        "n_pixels_before_filtering": len(y_image),
    })

    print("Pixels extraits :", len(y_image))
    print(
        "Classes de l'image :",
        np.unique(y_image, return_counts=True),
    )

    del (
        data_indices,
        data_band_depth,
        X_ind,
        X_bd,
        similarity_cube,
        topo_cube,
        climate_cube,
        cube,
        label_mask,
    )
    gc.collect()


# ============================================================
# 2. FUSION DE TOUS LES PIXELS
# ============================================================

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
groups = np.concatenate(groups_list, axis=0)

feature_names = reference_feature_names
group_metadata_df = pd.DataFrame(group_metadata)

assert len(X_all) == len(y_all) == len(groups)

assert X_all.shape[1] == len(feature_names), (
    f"X_all possède {X_all.shape[1]} variables, mais feature_names "
    f"contient {len(feature_names)} noms."
)

computed_classifier_feature_names = [
    feature_names[index]
    for index in classifier_feature_indices
]

assert computed_classifier_feature_names == classifier_feature_names, (
    "Les indices et les noms des variables du classifieur ne correspondent "
    "pas.\n"
    f"Noms calculés : {computed_classifier_feature_names}\n"
    f"Noms attendus : {classifier_feature_names}"
)

print("\n" + "=" * 80)
print("JEU DE DONNÉES GLOBAL")
print("=" * 80)
print("X_all :", X_all.shape)
print("y_all :", y_all.shape)
print("groups :", groups.shape)
print("Nombre d'images :", len(np.unique(groups)))
print("Classes avant exclusion :", np.unique(y_all, return_counts=True))


# ============================================================
# 3. EXCLUSION GLOBALE DES CLASSES NON RETENUES
# ============================================================

keep_global = ~np.isin(y_all, excluded_classes)

X_all = X_all[keep_global]
y_all = y_all[keep_global]
groups = groups[keep_global]

print("\nClasses exclues :", excluded_classes)
print("X_all après exclusion :", X_all.shape)
print("Classes conservées :", np.unique(y_all, return_counts=True))

classes, class_counts = np.unique(y_all, return_counts=True)

if np.any(class_counts < N_SPLITS):
    problematic_classes = classes[class_counts < N_SPLITS]
    raise ValueError(
        "StratifiedKFold nécessite au moins N_SPLITS pixels par classe. "
        f"Classes insuffisamment représentées : {problematic_classes}"
    )


# ============================================================
# 4. VALIDATION CROISÉE STRATIFIÉE 4-FOLD
# ============================================================
#
# À chaque fold :
# - 75 % des pixels sont utilisés pour l'apprentissage ;
# - 25 % des pixels sont utilisés pour le test ;
# - la proportion des classes est approximativement conservée ;
# - l'image d'origine des pixels n'est pas prise en compte.

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []
class_reports = []
confusion_records = []
importance_records = []
class_distribution_records = []
image_distribution_records = []

all_true_labels = []
all_predicted_labels = []
all_test_groups = []
all_test_indices = []

print("\n" + "=" * 80)
print("VALIDATION CROISÉE STRATIFIÉE 4-FOLD")
print("3/4 DES PIXELS EN TRAIN ET 1/4 EN TEST")
print("=" * 80)

for fold_number, (train_indices, test_indices) in enumerate(
    cv.split(X_all, y_all),
    start=1,
):

    print("\n" + "#" * 80)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("#" * 80)

    X_train_fold = X_all[train_indices]
    y_train_fold = y_all[train_indices]
    groups_train_fold = groups[train_indices]

    X_test_fold = X_all[test_indices]
    y_test_fold = y_all[test_indices]
    groups_test_fold = groups[test_indices]

    assert np.intersect1d(train_indices, test_indices).size == 0

    print("Pixels train :", len(y_train_fold))
    print("Pixels test  :", len(y_test_fold))
    print(
        "Proportion train :",
        f"{100 * len(y_train_fold) / len(y_all):.2f} %",
    )
    print(
        "Proportion test :",
        f"{100 * len(y_test_fold) / len(y_all):.2f} %",
    )

    # --------------------------------------------------------
    # Vérification des classes
    # --------------------------------------------------------
    missing_in_train = np.setdiff1d(
        np.unique(y_test_fold),
        np.unique(y_train_fold),
    )

    if missing_in_train.size > 0:
        raise RuntimeError(
            "Certaines classes du test sont absentes du train : "
            f"{missing_in_train}"
        )

    # --------------------------------------------------------
    # Distribution des images
    # --------------------------------------------------------
    fold_image_summary = image_count_table(
        groups_train_fold,
        groups_test_fold,
    )

    print("\nPixels de chaque image dans le train et le test :")
    print(fold_image_summary)

    for image_id, row in fold_image_summary.iterrows():
        image_distribution_records.append({
            "fold": fold_number,
            "image_id": int(image_id),
            "n_train": int(row["train"]),
            "n_test": int(row["test"]),
        })

    # --------------------------------------------------------
    # Sous-échantillonnage du train uniquement
    # --------------------------------------------------------
    X_train_sampled, y_train_sampled = subsample_by_class(
        X_train_fold,
        y_train_fold,
        max_per_class=MAX_SAMPLES_PER_CLASS,
    )

    fold_class_summary = class_count_table(
        y_train_sampled,
        y_test_fold,
    )

    print("\nEffectifs par classe :")
    print(fold_class_summary)

    for class_value, row in fold_class_summary.iterrows():
        class_distribution_records.append({
            "fold": fold_number,
            "class": class_value,
            "n_train": int(row["train"]),
            "n_test": int(row["test"]),
        })

    # --------------------------------------------------------
    # Sélection des variables
    # --------------------------------------------------------
    X_train_classifier = X_train_sampled[
        :,
        classifier_feature_indices,
    ]

    X_test_classifier = X_test_fold[
        :,
        classifier_feature_indices,
    ]

    # --------------------------------------------------------
    # Encodage des classes
    # --------------------------------------------------------
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train_sampled
    )

    y_test_encoded = label_encoder.transform(
        y_test_fold
    )

    # --------------------------------------------------------
    # Poids équilibrés
    # --------------------------------------------------------
    classifier_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    # --------------------------------------------------------
    # Modèle indépendant du fold
    # --------------------------------------------------------
    species_classifier = build_xgb_classifier()

    species_classifier.fit(
        X_train_classifier,
        y_train_encoded,
        sample_weight=classifier_sample_weights,
    )

    # --------------------------------------------------------
    # Prédictions
    # --------------------------------------------------------
    y_pred_encoded = species_classifier.predict(
        X_test_classifier
    )

    y_pred_encoded = np.asarray(
        y_pred_encoded,
        dtype=np.int64,
    )

    y_pred = label_encoder.inverse_transform(
        y_pred_encoded
    )

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------
    accuracy = accuracy_score(
        y_test_fold,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_fold,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test_fold,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test_fold,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    fold_results.append({
        "fold": fold_number,
        "n_train_before_sampling": len(y_train_fold),
        "n_train_after_sampling": len(y_train_sampled),
        "n_test": len(y_test_fold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    # --------------------------------------------------------
    # Rapport de classification
    # --------------------------------------------------------
    fold_labels = label_encoder.classes_

    report = classification_report(
        y_test_fold,
        y_pred,
        labels=fold_labels,
        output_dict=True,
        zero_division=0,
    )

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            class_reports.append({
                "fold": fold_number,
                "label": label,
                **metrics,
            })

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------
    cm = confusion_matrix(
        y_test_fold,
        y_pred,
        labels=fold_labels,
    )

    for true_position, true_class in enumerate(fold_labels):
        for pred_position, predicted_class in enumerate(fold_labels):
            confusion_records.append({
                "fold": fold_number,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": int(cm[true_position, pred_position]),
            })

    # --------------------------------------------------------
    # Importance des variables
    # --------------------------------------------------------
    fold_importance = pd.DataFrame({
        "feature": classifier_feature_names,
        "importance": species_classifier.feature_importances_,
    })

    fold_importance["fold"] = fold_number
    importance_records.append(fold_importance)

    print("\nImportance des variables du fold :")
    print(
        fold_importance
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Évaluation détaillée existante
    # --------------------------------------------------------
    if RUN_DETAILED_EVALUATION_EACH_FOLD:
        evaluate_classification_model(
            model=species_classifier,
            y_test=y_test_fold,
            y_pred=y_pred,
            feature_names=classifier_feature_names,
        )

    # --------------------------------------------------------
    # Conservation des prédictions hors-échantillon
    # --------------------------------------------------------
    all_true_labels.append(y_test_fold)
    all_predicted_labels.append(y_pred)
    all_test_groups.append(groups_test_fold)
    all_test_indices.append(test_indices)

    del (
        species_classifier,
        X_train_fold,
        y_train_fold,
        X_test_fold,
        y_test_fold,
        X_train_sampled,
        y_train_sampled,
        X_train_classifier,
        X_test_classifier,
        y_train_encoded,
        y_test_encoded,
        y_pred_encoded,
        y_pred,
        classifier_sample_weights,
    )
    gc.collect()


# ============================================================
# 5. VÉRIFICATION DE LA COUVERTURE DES PIXELS
# ============================================================

concatenated_test_indices = np.concatenate(all_test_indices)

assert len(concatenated_test_indices) == len(y_all), (
    "Tous les pixels doivent être utilisés exactement une fois dans le test."
)

assert len(np.unique(concatenated_test_indices)) == len(y_all), (
    "Certains pixels ont été utilisés plusieurs fois dans le test."
)


# ============================================================
# 6. SYNTHÈSE DES FOLDS
# ============================================================

fold_results_df = pd.DataFrame(fold_results)

print("\n" + "=" * 80)
print("RÉSULTATS PAR FOLD")
print("=" * 80)
print(fold_results_df.to_string(index=False))

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

cv_summary_df = (
    fold_results_df[metric_columns]
    .agg(["mean", "std", "min", "max"])
    .T
    .reset_index()
    .rename(columns={"index": "metric"})
)

print("\n" + "=" * 80)
print("SYNTHÈSE DE LA VALIDATION CROISÉE")
print("=" * 80)
print(cv_summary_df.to_string(index=False))


# ============================================================
# 7. MÉTRIQUES GLOBALES HORS-ÉCHANTILLON
# ============================================================

y_true_oof = np.concatenate(all_true_labels)
y_pred_oof = np.concatenate(all_predicted_labels)
groups_oof = np.concatenate(all_test_groups)
indices_oof = np.concatenate(all_test_indices)

global_metrics_df = pd.DataFrame([{
    "n_predictions": len(y_true_oof),
    "accuracy": accuracy_score(y_true_oof, y_pred_oof),
    "balanced_accuracy": balanced_accuracy_score(
        y_true_oof,
        y_pred_oof,
    ),
    "macro_f1": f1_score(
        y_true_oof,
        y_pred_oof,
        average="macro",
        zero_division=0,
    ),
    "weighted_f1": f1_score(
        y_true_oof,
        y_pred_oof,
        average="weighted",
        zero_division=0,
    ),
}])

print("\n" + "=" * 80)
print("MÉTRIQUES GLOBALES HORS-ÉCHANTILLON")
print("=" * 80)
print(global_metrics_df.to_string(index=False))


# ============================================================
# 8. AGRÉGATION DES IMPORTANCES
# ============================================================

importance_by_fold_df = pd.concat(
    importance_records,
    ignore_index=True,
)

importance_summary_df = (
    importance_by_fold_df
    .groupby("feature", as_index=False)
    .agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_min=("importance", "min"),
        importance_max=("importance", "max"),
    )
    .sort_values(
        "importance_mean",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("IMPORTANCE MOYENNE DES VARIABLES")
print("=" * 80)
print(importance_summary_df.to_string(index=False))


# ============================================================
# 9. TABLES COMPLÉMENTAIRES
# ============================================================

class_reports_df = pd.DataFrame(class_reports)
confusion_records_df = pd.DataFrame(confusion_records)
class_distribution_df = pd.DataFrame(class_distribution_records)
image_distribution_df = pd.DataFrame(image_distribution_records)

oof_predictions_df = pd.DataFrame({
    "original_pixel_index": indices_oof,
    "image_id": groups_oof,
    "y_true": y_true_oof,
    "y_pred": y_pred_oof,
    "correct": y_true_oof == y_pred_oof,
})

# Remise dans l'ordre initial des pixels.
oof_predictions_df = (
    oof_predictions_df
    .sort_values("original_pixel_index")
    .reset_index(drop=True)
)


# ============================================================
# 10. EXPORT DES RÉSULTATS
# ============================================================

output_dir = ROOT / "results_pixel_stratified_4fold"
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

fold_results_df.to_csv(
    output_dir / "fold_results.csv",
    index=False,
)

cv_summary_df.to_csv(
    output_dir / "cv_summary.csv",
    index=False,
)

global_metrics_df.to_csv(
    output_dir / "global_oof_metrics.csv",
    index=False,
)

importance_by_fold_df.to_csv(
    output_dir / "feature_importance_by_fold.csv",
    index=False,
)

importance_summary_df.to_csv(
    output_dir / "feature_importance_summary.csv",
    index=False,
)

class_reports_df.to_csv(
    output_dir / "classification_reports_by_fold.csv",
    index=False,
)

confusion_records_df.to_csv(
    output_dir / "confusion_matrices_by_fold.csv",
    index=False,
)

class_distribution_df.to_csv(
    output_dir / "class_distribution_by_fold.csv",
    index=False,
)

image_distribution_df.to_csv(
    output_dir / "image_distribution_by_fold.csv",
    index=False,
)

oof_predictions_df.to_csv(
    output_dir / "out_of_fold_predictions.csv",
    index=False,
)

group_metadata_df.to_csv(
    output_dir / "group_metadata.csv",
    index=False,
)

print("\nRésultats enregistrés dans :")
print(output_dir)