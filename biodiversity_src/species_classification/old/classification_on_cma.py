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
    config_path / "abbaretz_3.json", #9,
    config_path / "cma_septembre_1.json", #10
    config_path / "cma_septembre_2.json", #11
    config_path / "cma_septembre_3.json", #12
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

    # similarity_cube, similarity_feature_names = concatenate_tifs(
    #     input_dir=Path_res / "SAM_ACE_species_final",
    #     output_path=Path_res / "SAM_ACE_species_final.tif",
    # )

    # similarity_cube, similarity_feature_names, profile = concatenate_ace_maps(
    #     Path_res / "Similarity_maps"
    # )

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

# fusion globale
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)

groups = np.array(groups)

feature_names = (
    indices_feature_names
    + band_depth_feature_names
    + similarity_feature_names
    + topo_feature_names
    + climate_feature_names
)

assert cube.shape[0] == len(feature_names), (
    f"Incohérence : cube a {cube.shape[0]} features mais feature_names en a {len(feature_names)}")
# plot_pca_by_class_and_group(
#     X=X_all,
#     y=y_all,
#     groups=groups,
#     max_samples=50000,
# )


############### Validation croisée GroupKFold #############

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ============================================================
# SPLIT TRAIN / TEST PAR GROUPES
# ============================================================

test_group_ids = [4, 5, 6, 7, 10, 11, 12] #[0, 5,7, 9]# [4, 5, 6, 7]
#test_group_ids = [1, 5, 7, 9]  #[4, 5, 6, 7] 


is_test = np.isin(groups, test_group_ids)
is_train = ~is_test

print("group_train :", np.unique(groups[is_train]))
print("group_test  :", np.unique(groups[is_test]))

X_train_all = X_all[is_train]
y_train_all = y_all[is_train]

X_test_all = X_all[is_test]
y_test_all = y_all[is_test]


# ============================================================
# EXCLUSION DES CLASSES RARES / NON PERTINENTES
# ============================================================

excluded_classes = [10, 11, 7, 5, 9,13, 17, 16, 19, 20, 21] # 10, 11, 7, 5, 9, 13, 17

keep_train = ~np.isin(y_train_all, excluded_classes)
keep_test = ~np.isin(y_test_all, excluded_classes)

X_train_all = X_train_all[keep_train]
y_train_all = y_train_all[keep_train]

X_test_all = X_test_all[keep_test]
y_test = y_test_all[keep_test]

# ============================================================
# SÉCURITÉ : RETIRER DU TEST LES CLASSES ABSENTES DU TRAIN
# ============================================================

missing_in_train = set(np.unique(y_test)) - set(np.unique(y_train_all))

if missing_in_train:
    print("Classes présentes dans le test mais absentes du train :", missing_in_train)

    keep_test_known = np.isin(y_test, np.unique(y_train_all))
    X_test_all = X_test_all[keep_test_known]
    y_test = y_test[keep_test_known]

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    train_isolation_forests_by_class,
    predict_stress_by_class,
    anomaly_summary_by_class,
    combine_xgboost_and_isolation_forest,
)

# ============================================================
# 2. VÉRIFICATION DES EFFECTIFS PAR CLASSE
# ============================================================

train_counts = (
    pd.Series(y_train_all)
    .value_counts()
    .sort_index()
)

test_counts = (
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

class_summary = pd.DataFrame({
    "train": train_counts,
    "test": test_counts,
}).fillna(0).astype(int)

print("\nEffectifs par classe, avant seuillage :")
print(class_summary)


# ============================================================
# 1. SOUS-ÉCHANTILLONNAGE DU JEU D'ENTRAÎNEMENT
# ============================================================

X_train_full_sampled, y_train_sampled = subsample_by_class(
    X_train_all,
    y_train_all,
    max_per_class=30000,
)

print(
    "Classes du train :",
    np.unique(y_train_sampled, return_counts=True),
)

print(
    "Classes du test :",
    np.unique(y_test, return_counts=True),
)


# ============================================================
# 2. VÉRIFICATION DES EFFECTIFS PAR CLASSE
# ============================================================

train_counts = (
    pd.Series(y_train_sampled)
    .value_counts()
    .sort_index()
)

test_counts = (
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

class_summary = pd.DataFrame({
    "train": train_counts,
    "test": test_counts,
}).fillna(0).astype(int)

print("\nEffectifs par classe :")
print(class_summary)


# ============================================================
# 3. FEATURES UTILISÉES PAR LE CLASSIFIEUR XGBOOST
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
    feature_names[i]
    for i in classifier_feature_indices
]

assert computed_classifier_feature_names == classifier_feature_names, (
    "Les indices et les noms des variables du classifieur "
    "ne correspondent pas."
)

X_train_classifier = X_train_full_sampled[
    :,
    classifier_feature_indices,
]

X_test_classifier = X_test_all[
    :,
    classifier_feature_indices,
]

print("\nFeatures du classifieur :")
print(classifier_feature_names)

print(
    "X_train_classifier :",
    X_train_classifier.shape,
)

print(
    "X_test_classifier :",
    X_test_classifier.shape,
)


# ============================================================
# 4. ENCODAGE DES CLASSES
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train_sampled
)

y_test_encoded = label_encoder.transform(
    y_test
)

print(
    "Classes encodées :",
    label_encoder.classes_,
)


# ============================================================
# 5. POIDS DES OBSERVATIONS
# ============================================================

classifier_sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded,
)


# ============================================================
# 6. CLASSIFIEUR XGBOOST
# ============================================================

species_classifier = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    tree_method="hist",
    device="cuda",
    eval_metric="mlogloss",
    random_state=42,
)


# ============================================================
# 7. ENTRAÎNEMENT DU CLASSIFIEUR
# ============================================================

species_classifier.fit(
    X_train_classifier,
    y_train_encoded,
    sample_weight=classifier_sample_weights,
)


# ============================================================
# 8. PRÉDICTION DES ESSENCES
# ============================================================

y_species_pred_encoded = species_classifier.predict(
    X_test_classifier
)

y_species_pred = label_encoder.inverse_transform(
    y_species_pred_encoded
)

classification_accuracy = accuracy_score(
    y_test,
    y_species_pred,
)

print(
    f"\nAccuracy du classifieur : "
    f"{classification_accuracy:.4f}"
)


# ============================================================
# 9. ÉVALUATION DU CLASSIFIEUR
# ============================================================

evaluate_classification_model(
    model=species_classifier,
    y_test=y_test,
    y_pred=y_species_pred,
    feature_names=classifier_feature_names,
)

classifier_importance = pd.DataFrame({
    "feature": classifier_feature_names,
    "importance": species_classifier.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

print("\nImportance des variables XGBoost :")
print(classifier_importance)


# ============================================================
# 10. FEATURES UTILISÉES PAR L'ISOLATION FOREST
# ============================================================

anomaly_feature_indices = [0, 1, 2, 3,4,5]
anomaly_feature_names = ["NDVI", "NDRE", "NDWI", "PRI", "ARI", "EVI"]

computed_anomaly_feature_names = [
    feature_names[i]
    for i in anomaly_feature_indices
]

assert computed_anomaly_feature_names == anomaly_feature_names, (
    "Les indices et les noms des variables de détection "
    "d'anomalies ne correspondent pas."
)

X_train_anomaly = X_train_full_sampled[
    :,
    anomaly_feature_indices,
]

X_test_anomaly = X_test_all[
    :,
    anomaly_feature_indices,
]

print("\nFeatures Isolation Forest :")
print(anomaly_feature_names)

print(
    "X_train_anomaly :",
    X_train_anomaly.shape,
)

print(
    "X_test_anomaly :",
    X_test_anomaly.shape,
)


stats_df = compute_feature_statistics_by_class(
    X=X_train_anomaly,
    y=y_train_sampled,
    feature_names=anomaly_feature_names,
    output_excel=Path_res.parent / "Stats_on_indices/statistiques_indices_par_classe.xlsx",
)


# stats_df = compute_feature_statistics_by_class(
#     X_train_anomaly,
#     y_train_sampled,
#     anomaly_feature_names,
# )

# print(stats_df)

# ============================================================
# 11. ENTRAÎNEMENT DES ISOLATION FORESTS PAR CLASSE
# ============================================================

(
    isolation_models,
    isolation_thresholds,
    isolation_score_references,
    isolation_training_summary,
) = train_isolation_forests_by_class(
    X_train=X_train_anomaly,
    y_train=y_train_sampled,
    feature_names=anomaly_feature_names,
    threshold_quantile=0.95,
    min_samples_per_class=500,
    max_samples_per_class=50000,
    n_estimators=300,
    max_samples=5000,
    random_state=42,
)

print("\nRésumé de l'entraînement Isolation Forest :")
print(isolation_training_summary)


# ============================================================
# 12. ÉVALUATION DES ANOMALIES AVEC LES VRAIES CLASSES
# ============================================================

(
    anomaly_scores_true_labels,
    anomaly_predictions_true_labels,
) = predict_stress_by_class(
    X=X_test_anomaly,
    species_labels=y_test,
    isolation_models=isolation_models,
    thresholds=isolation_thresholds,
)

anomaly_summary_true_labels = anomaly_summary_by_class(
    species_labels=y_test,
    anomaly_scores=anomaly_scores_true_labels,
    stress_prediction=anomaly_predictions_true_labels,
)

print("\nAnomalies évaluées avec les vraies classes :")
print(anomaly_summary_true_labels)


# ============================================================
# 13. ÉVALUATION DES ANOMALIES AVEC LES CLASSES XGBOOST
# ============================================================

(
    anomaly_scores_predicted_labels,
    anomaly_predictions_predicted_labels,
) = predict_stress_by_class(
    X=X_test_anomaly,
    species_labels=y_species_pred,
    isolation_models=isolation_models,
    thresholds=isolation_thresholds,
)

anomaly_summary_predicted_labels = anomaly_summary_by_class(
    species_labels=y_species_pred,
    anomaly_scores=anomaly_scores_predicted_labels,
    stress_prediction=anomaly_predictions_predicted_labels,
)

print("\nAnomalies évaluées avec les classes prédites par XGBoost :")
print(anomaly_summary_predicted_labels)

# ============================================================
# 13bis. COMPARAISON SUR LES MÊMES PIXELS (VRAIS LABELS)
# ============================================================

comparison_rows = []

for species in np.unique(y_test):

    mask = (y_test == species)

    n_pixels = mask.sum()

    if n_pixels == 0:
        continue

    # Anomalies obtenues avec la vraie classe
    anomaly_rate_true = anomaly_predictions_true_labels[mask].mean()

    # Anomalies obtenues avec la classe prédite
    anomaly_rate_pred = anomaly_predictions_predicted_labels[mask].mean()

    # Exactitude de classification
    classification_accuracy = (y_species_pred[mask] == species).mean()

    comparison_rows.append({
        "class": species,
        "n_pixels": n_pixels,
        "classification_accuracy": classification_accuracy,
        "anomaly_rate_true_label": anomaly_rate_true,
        "anomaly_rate_predicted_label": anomaly_rate_pred,
        "difference": anomaly_rate_pred - anomaly_rate_true,
    })

comparison_same_pixels = pd.DataFrame(comparison_rows)

comparison_same_pixels["classification_accuracy"] *= 100
comparison_same_pixels["anomaly_rate_true_label"] *= 100
comparison_same_pixels["anomaly_rate_predicted_label"] *= 100
comparison_same_pixels["difference"] *= 100

print("\nComparaison sur les mêmes pixels :")
print(comparison_same_pixels)

# ============================================================
# 13ter. ANALYSE DES PIXELS BIEN / MAL CLASSÉS
# ============================================================

analysis_rows = []

for species in np.unique(y_test):

    mask_species = (y_test == species)

    mask_correct = mask_species & (y_species_pred == species)
    mask_wrong = mask_species & (y_species_pred != species)

    n_correct = mask_correct.sum()
    n_wrong = mask_wrong.sum()

    anomaly_correct = (
        anomaly_predictions_true_labels[mask_correct].mean()
        if n_correct > 0 else np.nan
    )

    anomaly_wrong = (
        anomaly_predictions_true_labels[mask_wrong].mean()
        if n_wrong > 0 else np.nan
    )

    analysis_rows.append({
        "class": species,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "anomaly_rate_correct": anomaly_correct * 100 if not np.isnan(anomaly_correct) else np.nan,
        "anomaly_rate_wrong": anomaly_wrong * 100 if not np.isnan(anomaly_wrong) else np.nan,
    })

analysis_df = pd.DataFrame(analysis_rows)

print("\nInfluence des erreurs de classification :")
print(analysis_df)

# ============================================================
# 14. COMBINAISON CONFIANCE XGBOOST + SCORE D'ANOMALIE
# ============================================================

combined_anomaly_results = combine_xgboost_and_isolation_forest(
    classifier=species_classifier,
    label_encoder=label_encoder,
    X_classifier=X_test_classifier,
    X_anomaly=X_test_anomaly,
    isolation_models=isolation_models,
    isolation_thresholds=isolation_thresholds,
    isolation_score_references=isolation_score_references,
    min_confidence=0.70,
    high_confidence=0.85,
    combined_threshold=0.90,
    confidence_power=1.0,
)

print("\nRésultats combinés :")
print(combined_anomaly_results.head())


# ============================================================
# 15. RÉSUMÉ DES STATUTS COMBINÉS
# ============================================================

combined_status_summary = (
    combined_anomaly_results["status"]
    .value_counts(dropna=False)
    .rename_axis("status")
    .reset_index(name="n_pixels")
)

combined_status_summary["proportion"] = (
    combined_status_summary["n_pixels"]
    / len(combined_anomaly_results)
)

print("\nRésumé des statuts combinés :")
print(combined_status_summary)
# # ============================================================
# # SAUVEGARDE DES MODÈLES
# # ============================================================

# output_dir = Path_res.parent / "Isolation_Forest_models"
# output_dir.mkdir(parents=True, exist_ok=True)

# model_bundle = {
#     "models": isolation_models,
#     "thresholds": isolation_thresholds,
#     "feature_names": best_features,
#     "feature_indices": best_indices,
#     "training_summary": isolation_summary,
#     "threshold_quantile": 0.95,
# }

# joblib.dump(
#     model_bundle,
#     output_dir / "isolation_forests_by_species.joblib",
# )

# isolation_summary.to_csv(
#     output_dir / "isolation_forest_training_summary.csv",
#     index=False,
# )

# print("Modèles sauvegardés dans :", output_dir)

# # ============================================================
# # CHARGEMENT ULTÉRIEUR DES MODÈLES
# # ============================================================

# model_bundle = joblib.load(
#     output_dir / "isolation_forests_by_species.joblib"
# )

# isolation_models = model_bundle["models"]
# isolation_thresholds = model_bundle["thresholds"]
# best_features = model_bundle["feature_names"]
# best_indices = model_bundle["feature_indices"]