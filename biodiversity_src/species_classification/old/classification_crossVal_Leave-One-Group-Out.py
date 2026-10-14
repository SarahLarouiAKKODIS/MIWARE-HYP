### Le script `main_biodiv_indice_calculation.py` doit être exécuté 
# préalablement afin de générer les indices utilisés pour la classification.

from pathlib import Path
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import rasterio

from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import joblib
from utils import evaluate_classification_model, concatenate_tifs, concatenate_ace_maps, cube_to_dataset, subsample_by_class, plot_pca_by_class_and_group, concatenate_similarity_best_maps, compute_feature_statistics_by_class

config_path = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToTest")

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



indices_feature_names = [
    "NDVI",
    "NDRE",
    "NDWI",
    "PRI",
    "ARI",
    "EVI",
    "NBR"]

band_depth_feature_names = ["BD970", "BD1200", "BD1450", "BD1940", "BD2200"]

topo_feature_names = ["altitude", "slope", "aspect", "tpi", "twi", "tri", "curvature"]

#climate_feature_names = ['mean_annual_temperature', 'temperature_seasonality', 'annual_precipitation', 'precipitation_seasonality']
climate_feature_names = ['mean_temperature', 'annual_precipitation', 'summer_water_deficit', 'annual_pet', 'temperature_seasonality', 'precipitation_seasonality']

X_list = []
y_list = []
groups = []

group_id = 0

for config_path in config_files:

    print('config_path ===========================================', config_path)

    with open(config_path) as f:
        config = json.load(f)

    Path_res = Path(config["Path_res"])

    ## indices biodiversité

    data_indices = np.load(Path_res / "Feature_extraction/dataset_indices_for_species_classif.npz")
    X_ind = data_indices["X"]
    y = data_indices["y"]
    
    ## profondeur de bandes d'absorption discriminantes

    data_band_dept = np.load(Path_res / "Feature_extraction/band_depth_cube.npz")
    X_bd = data_band_dept["X"]

    ## cartes de similarité spectrale band_depth_cube

    similarity_cube, similarity_feature_names = concatenate_similarity_best_maps(
        input_dir=Path_res / "Similarity_best_maps",
        output_path=Path_res / "Feature_extraction/similarity_best_ace.tif",
        mode="ace",
    )

    with rasterio.open(
        Path_res / "Feature_extraction/topographic_features.tif") as src:

        topo_cube = src.read().astype(np.float32)

        topo_feature_names = [
            src.descriptions[i]
            for i in range(src.count)
        ]


    with rasterio.open(
        Path_res / "Feature_extraction/climate_features_2000_2024.tif") as src:

        climate_cube = src.read().astype(np.float32)

        climate_feature_names = [
            src.descriptions[i]
            for i in range(src.count)
        ]


    for i,name in enumerate(climate_feature_names):

        print(name)

        print(np.nanmin(climate_cube[i]))
        print(np.nanmax(climate_cube[i]))
        print(np.nanmean(climate_cube[i]))

    ## varaibles topographie s :  altitude, pente, exposition, TPI, TWI approximatif
    # print('feature_names_sam_ace', len(similarity_feature_names))
    # print('X_ind', X_ind.shape)
    # print('similarity_cube', similarity_cube.shape)
    # print('X_bd', X_bd.shape)
    # print('topo_cube', topo_cube.shape)
    # print('climate_cube', climate_cube.shape)
    assert X_ind.shape[1:] == similarity_cube.shape[1:] == X_bd.shape[1:]== topo_cube.shape[1:] == climate_cube.shape[1:], \
    "Les cubes doivent avoir les mêmes dimensions spatiales."

    cube = np.concatenate((X_ind, X_bd, similarity_cube, topo_cube, climate_cube), axis=0)

    #cube = np.concatenate((X_ind, X_bd, similarity_cube, topo_cube), axis=0)
    #cube = np.concatenate((X_ind, X_bd), axis=0)
    print('cube', cube.shape)

    ######### Construction du dataset IA [X, y] ###################################
    # Conversion du cube de caractéristiques (features × lignes × colonnes)
    # en un dataset (pixels × features) en conservant uniquement les pixels
    # valides d'EnMAP et les classes d'intérêt.

    X, y = cube_to_dataset(
        feature_cube=cube,
        label_mask=y,
        valid_mask_path=Path_res / "enmap_valid_mask.tif",
        ignore_labels=(0,),
    )

    X_list.append(X)
    y_list.append(y)

    # même groupe pour tout le fichier
    groups.extend([group_id] * len(y))

    print('config_path', [group_id, config_path])

    group_id += 1

# ============================================================
# FUSION GLOBALE
# ============================================================

import gc
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_STATE = 42
MAX_PER_CLASS = 30000
RUN_DETAILED_EVALUATION_EACH_FOLD = True

excluded_classes = [10, 11, 7, 5, 9, 13, 17, 16, 19, 20, 21]

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
groups = np.asarray(groups)

feature_names = (
    indices_feature_names
    + band_depth_feature_names
    + similarity_feature_names
    + topo_feature_names
    + climate_feature_names
)

assert X_all.shape[1] == len(feature_names), (
    f"Incohérence : X_all possède {X_all.shape[1]} variables, "
    f"mais feature_names contient {len(feature_names)} noms."
)
assert len(X_all) == len(y_all) == len(groups), (
    "X_all, y_all et groups doivent contenir le même nombre d'observations."
)

print("\nJeu de données complet")
print("X_all :", X_all.shape)
print("y_all :", y_all.shape)
print("groups :", groups.shape)
print("Nombre d'images :", len(np.unique(groups)))


# ============================================================
# EXCLUSION GLOBALE DES CLASSES NON RETENUES
# ============================================================

keep_global = ~np.isin(y_all, excluded_classes)
X_all = X_all[keep_global]
y_all = y_all[keep_global]
groups = groups[keep_global]

print("\nClasses conservées :", np.unique(y_all, return_counts=True))
print("Images conservées :", np.unique(groups))


# ============================================================
# FEATURES UTILISÉES PAR LE CLASSIFIEUR XGBOOST
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

computed_classifier_feature_names = [
    feature_names[i] for i in classifier_feature_indices
]

assert computed_classifier_feature_names == classifier_feature_names, (
    "Les indices et les noms des variables du classifieur ne correspondent pas.\n"
    f"Noms calculés : {computed_classifier_feature_names}\n"
    f"Noms attendus : {classifier_feature_names}"
)


# ============================================================
# VALIDATION CROISÉE LEAVE-ONE-IMAGE-OUT
# ============================================================

logo = LeaveOneGroupOut()

fold_results = []
importance_results = []
classification_reports = []
confusion_results = []

all_y_true = []
all_y_pred = []
all_test_groups = []

n_folds = logo.get_n_splits(X_all, y_all, groups)
print(f"\nNombre de folds Leave-One-Image-Out : {n_folds}")

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X_all, y_all, groups=groups),
    start=1,
):
    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{n_folds}")
    print("=" * 80)

    train_group_ids = np.unique(groups[train_idx])
    test_group_ids = np.unique(groups[test_idx])

    assert len(test_group_ids) == 1, (
        "Chaque fold doit contenir exactement une image de test."
    )

    test_group_id = int(test_group_ids[0])

    print("Images d'entraînement :", train_group_ids)
    print("Image de test :", test_group_id)

    X_train_all = X_all[train_idx]
    y_train_all = y_all[train_idx]

    X_test_all = X_all[test_idx]
    y_test = y_all[test_idx]

    # --------------------------------------------------------
    # Retirer du test les classes absentes de l'entraînement
    # --------------------------------------------------------
    train_classes = np.unique(y_train_all)
    missing_in_train = np.setdiff1d(np.unique(y_test), train_classes)

    n_test_before_filtering = len(y_test)

    if missing_in_train.size > 0:
        print(
            "Classes présentes dans le test mais absentes du train :",
            missing_in_train,
        )
        keep_test_known = np.isin(y_test, train_classes)
        X_test_all = X_test_all[keep_test_known]
        y_test = y_test[keep_test_known]

    n_removed = n_test_before_filtering - len(y_test)

    if len(y_test) == 0:
        print("Fold ignoré : aucune classe test n'est connue dans le train.")
        fold_results.append({
            "fold": fold,
            "test_group": test_group_id,
            "status": "ignored",
            "n_train_before_sampling": len(y_train_all),
            "n_train_after_sampling": 0,
            "n_test_before_filtering": n_test_before_filtering,
            "n_test_evaluated": 0,
            "n_test_removed_unknown_classes": n_removed,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        })
        continue

    # --------------------------------------------------------
    # Sous-échantillonnage du train uniquement
    # --------------------------------------------------------
    X_train_sampled, y_train_sampled = subsample_by_class(
        X_train_all,
        y_train_all,
        max_per_class=MAX_PER_CLASS,
    )

    print(
        "Classes du train :",
        np.unique(y_train_sampled, return_counts=True),
    )
    print(
        "Classes du test :",
        np.unique(y_test, return_counts=True),
    )

    train_counts = pd.Series(y_train_sampled).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    class_summary = pd.DataFrame({
        "train": train_counts,
        "test": test_counts,
    }).fillna(0).astype(int)

    print("\nEffectifs par classe :")
    print(class_summary)

    # --------------------------------------------------------
    # Sélection des caractéristiques
    # --------------------------------------------------------
    X_train_classifier = X_train_sampled[:, classifier_feature_indices]
    X_test_classifier = X_test_all[:, classifier_feature_indices]

    # --------------------------------------------------------
    # Encodage des classes propre au fold
    # --------------------------------------------------------
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_sampled)
    y_test_encoded = label_encoder.transform(y_test)

    # --------------------------------------------------------
    # Poids des observations
    # --------------------------------------------------------
    classifier_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    # --------------------------------------------------------
    # Nouveau classifieur pour ce fold
    # --------------------------------------------------------
    species_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    species_classifier.fit(
        X_train_classifier,
        y_train_encoded,
        sample_weight=classifier_sample_weights,
    )

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------
    y_pred_encoded = species_classifier.predict(X_test_classifier)
    y_pred_encoded = np.asarray(y_pred_encoded, dtype=np.int64)
    y_species_pred = label_encoder.inverse_transform(y_pred_encoded)

    # --------------------------------------------------------
    # Métriques du fold
    # --------------------------------------------------------
    accuracy = accuracy_score(y_test, y_species_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_species_pred)
    macro_f1 = f1_score(
        y_test,
        y_species_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_test,
        y_species_pred,
        average="weighted",
        zero_division=0,
    )

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    fold_results.append({
        "fold": fold,
        "test_group": test_group_id,
        "status": "evaluated",
        "n_train_before_sampling": len(y_train_all),
        "n_train_after_sampling": len(y_train_sampled),
        "n_test_before_filtering": n_test_before_filtering,
        "n_test_evaluated": len(y_test),
        "n_test_removed_unknown_classes": n_removed,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    # --------------------------------------------------------
    # Rapport de classification
    # --------------------------------------------------------
    report = classification_report(
        y_test,
        y_species_pred,
        labels=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            classification_reports.append({
                "fold": fold,
                "test_group": test_group_id,
                "label": label,
                **metrics,
            })

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------
    fold_labels = label_encoder.classes_
    cm = confusion_matrix(
        y_test,
        y_species_pred,
        labels=fold_labels,
    )

    for i, true_class in enumerate(fold_labels):
        for j, predicted_class in enumerate(fold_labels):
            confusion_results.append({
                "fold": fold,
                "test_group": test_group_id,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "count": int(cm[i, j]),
            })

    # --------------------------------------------------------
    # Importance des variables
    # --------------------------------------------------------
    fold_importance = pd.DataFrame({
        "fold": fold,
        "test_group": test_group_id,
        "feature": classifier_feature_names,
        "importance": species_classifier.feature_importances_,
    })
    importance_results.append(fold_importance)

    print("\nImportance des variables :")
    print(
        fold_importance
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Cette fonction peut produire des graphiques pour chaque fold.
    if RUN_DETAILED_EVALUATION_EACH_FOLD:
        evaluate_classification_model(
            model=species_classifier,
            y_test=y_test,
            y_pred=y_species_pred,
            feature_names=classifier_feature_names,
        )

    all_y_true.append(y_test.copy())
    all_y_pred.append(y_species_pred.copy())
    all_test_groups.append(
        np.full(len(y_test), test_group_id, dtype=np.int32)
    )

    del species_classifier
    gc.collect()


# ============================================================
# SYNTHÈSE DES FOLDS
# ============================================================

fold_results_df = pd.DataFrame(fold_results)
print("\n" + "=" * 80)
print("RÉSULTATS PAR IMAGE")
print("=" * 80)
print(fold_results_df.to_string(index=False))

evaluated_folds_df = fold_results_df[
    fold_results_df["status"] == "evaluated"
].copy()

if evaluated_folds_df.empty:
    raise RuntimeError("Aucun fold n'a pu être évalué.")

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

cv_summary_df = (
    evaluated_folds_df[metric_columns]
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
# MÉTRIQUES GLOBALES SUR LES PRÉDICTIONS HORS-ÉCHANTILLON
# ============================================================

y_true_oof = np.concatenate(all_y_true)
y_pred_oof = np.concatenate(all_y_pred)
groups_oof = np.concatenate(all_test_groups)

global_metrics_df = pd.DataFrame([{
    "n_predictions": len(y_true_oof),
    "accuracy": accuracy_score(y_true_oof, y_pred_oof),
    "balanced_accuracy": balanced_accuracy_score(y_true_oof, y_pred_oof),
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
# IMPORTANCE MOYENNE DES VARIABLES
# ============================================================

importance_by_fold_df = pd.concat(
    importance_results,
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
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("IMPORTANCE MOYENNE DES VARIABLES")
print("=" * 80)
print(importance_summary_df.to_string(index=False))


# ============================================================
# EXPORT DES RÉSULTATS
# ============================================================

classification_reports_df = pd.DataFrame(classification_reports)
confusion_results_df = pd.DataFrame(confusion_results)

oof_predictions_df = pd.DataFrame({
    "group": groups_oof,
    "y_true": y_true_oof,
    "y_pred": y_pred_oof,
    "correct": y_true_oof == y_pred_oof,
})

output_dir = ROOT / "results_leave_one_image_out"
output_dir.mkdir(parents=True, exist_ok=True)

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
classification_reports_df.to_csv(
    output_dir / "classification_reports_by_fold.csv",
    index=False,
)
confusion_results_df.to_csv(
    output_dir / "confusion_matrices_by_fold.csv",
    index=False,
)
oof_predictions_df.to_csv(
    output_dir / "out_of_fold_predictions.csv",
    index=False,
)

print("\nRésultats enregistrés dans :", output_dir)