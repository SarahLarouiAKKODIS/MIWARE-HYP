### Le script `main_biodiv_indice_calculation.py` doit être exécuté 
# préalablement afin de générer les indices utilisés pour la classification.

from pathlib import Path
import numpy as np
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import rasterio

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)#import joblib
from utils import evaluate_classification_model, cube_to_dataset, subsample_by_class, concatenate_similarity_best_maps



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

test_group_ids = [4, 5, 6, 7] #[0, 5,7, 9]# [4, 5, 6, 7]

is_test = np.isin(groups, test_group_ids)
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

excluded_classes = [10, 11, 7, 5, 9,13, 17, 16, 19, 20, 21] # 10, 11, 7, 5, 9, 13, 17

keep_train = ~np.isin(y_train_all, excluded_classes)
keep_test = ~np.isin(y_test, excluded_classes)

X_train_all = X_train_all[keep_train]
y_train_all = y_train_all[keep_train]

X_test = X_test[keep_test]
y_test = y_test[keep_test]

# ============================================================
# SÉLECTION DES MEILLEURS CARACTÉRISTIQUES
# ============================================================

best_indices = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21]
best_features = ['NDVI', 'NDRE', 'PRI', 'ARI', 'EVI', 'BD970', 'BD1200', 'BD1450', 'BD1940', 'BD2200', 'ace_max', 'altitude', 'slope', 'aspect', 'twi', 'tri', 'curvature', 'mean_temperature']

X_train_all = X_train_all[:, best_indices]
X_test = X_test[:, best_indices]

selected_feature_names = [
    feature_names[i]
    for i in best_indices
]

print(selected_feature_names)

# ============================================================
# VÉRIFICATION DES EFFECTIFS PAR CLASSE
# ============================================================

train_counts = pd.Series(y_train_all).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

summary = pd.DataFrame({
    "Train": train_counts,
    "Test": test_counts,
}).fillna(0).astype(int)

print(summary)


# ============================================================
# SOUS-ÉCHANTILLONNAGE DU TRAIN
# ============================================================

# X_train, y_train = subsample_by_class(
#     X_train_all,
#     y_train_all,
#     max_per_class=30000,
# )

X_train, y_train = X_train_all, y_train_all

print("y_train :", np.unique(y_train, return_counts=True))
print("y_test  :", np.unique(y_test, return_counts=True))


# ============================================================
# SÉCURITÉ : RETIRER DU TEST LES CLASSES ABSENTES DU TRAIN
# ============================================================

missing_in_train = set(np.unique(y_test)) - set(np.unique(y_train))

if missing_in_train:
    print("Classes présentes dans le test mais absentes du train :", missing_in_train)

    keep_test_known = np.isin(y_test, np.unique(y_train))
    X_test = X_test[keep_test_known]
    y_test = y_test[keep_test_known]


# ============================================================
# ENCODAGE DES LABELS
# ============================================================

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# VÉRIFICATIONS
# ============================================================

feature_names = list(selected_feature_names)

if X_train.shape[1] != len(feature_names):
    raise ValueError(
        f"X_train possède {X_train.shape[1]} colonnes, "
        f"mais feature_names contient {len(feature_names)} noms."
    )

if X_test.shape[1] != len(feature_names):
    raise ValueError(
        f"X_test possède {X_test.shape[1]} colonnes, "
        f"mais feature_names contient {len(feature_names)} noms."
    )


# ============================================================
# ENCODAGE DES LABELS
# ============================================================

le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

n_classes = len(le.classes_)

print("Classes :", le.classes_)
print("Nombre de classes :", n_classes)


# ============================================================
# POIDS D'ÉCHANTILLONS
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_enc,
)


# ============================================================
# DÉFINITION DES MODÈLES
# ============================================================

models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    ),

    "LightGBM": LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=300,
        max_depth=8,
        num_leaves=255,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================

comparison_results = []
feature_importances = {}
trained_models = {}
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"MODÈLE : {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train_enc,
        sample_weight=sample_weights,
    )

    training_time = time.perf_counter() - start_train

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    start_predict = time.perf_counter()

    y_pred_enc = model.predict(X_test)
    y_pred_enc = np.asarray(y_pred_enc, dtype=np.int64)

    prediction_time = time.perf_counter() - start_predict

    y_pred = le.inverse_transform(y_pred_enc)

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    comparison_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
    })

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"Temps entraînement: {training_time:.2f} s")
    print(f"Temps prédiction  : {prediction_time:.2f} s")

    # --------------------------------------------------------
    # Évaluation détaillée
    # --------------------------------------------------------

    evaluate_classification_model(
        model=model,
        y_test=y_test,
        y_pred=y_pred,
        feature_names=feature_names,
    )

    # --------------------------------------------------------
    # Importance des variables
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        feature_importances[model_name] = importance_df

        print("\nImportance des variables :")
        print(importance_df)

    trained_models[model_name] = model
    predictions[model_name] = y_pred


# ============================================================
# TABLEAU COMPARATIF
# ============================================================

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    by="balanced_accuracy",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 70)
print("COMPARAISON FINALE DES MODÈLES")
print("=" * 70)

print(comparison_df.to_string(index=False))


# ============================================================
# MEILLEUR MODÈLE
# ============================================================

best_model_name = comparison_df.loc[0, "model"]
best_balanced_accuracy = comparison_df.loc[0, "balanced_accuracy"]

best_model = trained_models[best_model_name]
best_prediction = predictions[best_model_name]

print("\nMeilleur modèle :", best_model_name)
print(
    f"Meilleure Balanced Accuracy : "
    f"{best_balanced_accuracy:.4f}"
)