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
from feature_extraction import build_species_classification_dataset
from display import class_distribution
from utils import load_config

#classification_config = load_config("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/classification_config.json")

sites_config = load_config(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/sites_config.json"
)

train_configs = [
    entry["config_file"]
    for entry in sorted(
        sites_config["train"],
        key=lambda x: x["image_id"]
    )
]

test_configs = [
    entry["config_file"]
    for entry in sorted(
        sites_config["test"],
        key=lambda x: x["image_id"]
    )
]

X_train_all, y_train_all, groups_train, feature_names_train = build_species_classification_dataset.build_features_dataset(
    config_files=train_configs, Path_config=sites_config["config_dir"]
)

X_test_all, y_test_all, groups_test, feature_names_test = build_species_classification_dataset.build_features_dataset(
    config_files=test_configs, Path_config=sites_config["config_dir"]
)
df_distribution = class_distribution.display_class_distribution(
    y_train=y_train_all,
    y_test=y_test_all,
)

assert list(feature_names_train) == list(feature_names_test), (
    "L'ordre des caractéristiques diffère entre train et test."
)

feature_names = feature_names_train

# ============================================================
# EXCLUSION DES CLASSES RARES / NON PERTINENTES
# ============================================================

#excluded_classes = [10, 11, 7, 5, 9,13, 17, 16, 19, 20, 21] # 10, 11, 7, 5, 9, 13, 17
#EXCLUDED_LABELS = [5, 7, 9, 10, 11, 13, 17, 21]
# EXCLUDED_LABELS = [
#     5, 7, 9, 10, 11, 13, 17, 21,
#     4, 14, 18, 19, 20,
# ]
EXCLUDED_LABELS = [
    5, 7, 9, 10, 11, 13, 17, 19, 20, 21]

# Masques des classes à conserver
keep_train = ~np.isin(y_train_all, EXCLUDED_LABELS)
keep_test = ~np.isin(y_test_all, EXCLUDED_LABELS)

# Filtrage du train
X_train_all = X_train_all[keep_train]
y_train_all = y_train_all[keep_train]
groups_train = groups_train[keep_train]

# Filtrage du test
X_test_all = X_test_all[keep_test]
y_test_all = y_test_all[keep_test]
groups_test = groups_test[keep_test]



# ============================================================
# 1. SOUS-ÉCHANTILLONNAGE DU JEU D'ENTRAÎNEMENT
# ============================================================

X_train_full_sampled, y_train_sampled = subsample_by_class(
    X_train_all,
    y_train_all,
    max_per_class=30000,
)

df_distribution = class_distribution.display_class_distribution(
    y_train=y_train_all,
    y_test=y_test_all,
)

####################################################################################""


from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Validation par image/groupe
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

idx_train, idx_val = next(splitter.split(
    X_train_all, y_train_all, groups_train
))


X_fit, y_fit = subsample_by_class(
    X_train_all[idx_train],
    y_train_all[idx_train],
    max_per_class=30000,
)

X_val = X_train_all[idx_val]
y_val = y_train_all[idx_val]

encoder = LabelEncoder()
y_fit_enc = encoder.fit_transform(y_fit)
y_val_enc = encoder.transform(y_val)

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_fit_enc
)

def create_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=8,
    )

# 2. Modèle initial avec les 31 caractéristiques
model_all = create_model()
model_all.fit(X_fit, y_fit_enc, sample_weight=weights)

# 3. Permutation importance
# On limite le nombre de pixels pour réduire le temps de calcul.
rng = np.random.default_rng(42)
val_idx = rng.choice(
    len(X_val),
    size=min(10000, len(X_val)),
    replace=False
)

X_val_small = X_val[val_idx]
y_val_small = y_val_enc[val_idx]

result = permutation_importance(
    model_all,
    X_val_small,
    y_val_small,
    scoring="balanced_accuracy",
    n_repeats=3,
    random_state=42,
    n_jobs=1,
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": result.importances_mean,
    "std": result.importances_std,
    "index": np.arange(len(feature_names)),
}).sort_values("importance", ascending=False)

print(importance_df[["feature", "importance", "std"]])

# 4. Tester plusieurs nombres de caractéristiques
ranking = importance_df["index"].to_numpy()
results = []

for n_features in [5, 10, 15, 20, 25, 31]:
    selected_idx = ranking[:n_features]

    model = create_model()
    model.fit(
        X_fit[:, selected_idx],
        y_fit_enc,
        sample_weight=weights
    )

    pred = model.predict(X_val[:, selected_idx])
    score = balanced_accuracy_score(y_val_enc, pred)

    results.append({
        "n_features": n_features,
        "balanced_accuracy": score,
    })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(
    results_df["n_features"],
    results_df["balanced_accuracy"],
    marker="o"
)
plt.xlabel("Nombre de caractéristiques")
plt.ylabel("Balanced accuracy")
plt.title("Performance selon le nombre de caractéristiques")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

best_n = int(
    results_df.loc[
        results_df["balanced_accuracy"].idxmax(),
        "n_features"
    ]
)

selected_idx = ranking[:best_n]
selected_features = np.array(feature_names)[selected_idx]

print("Nombre retenu :", best_n)
print("Caractéristiques retenues :", selected_features)

####################################################################################""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================================================
# 4. ENCODAGE DES CLASSES
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train_sampled
)

y_test_encoded = label_encoder.transform(
    y_test_all
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
# SÉLECTION DES MEILLEURES CARACTÉRISTIQUES
# ============================================================

# # Appliquer la sélection au train et au test
X_train_selected = X_train_full_sampled[:, selected_idx]
X_test_selected = X_test_all[:, selected_idx]

# ============================================================
# 6. CLASSIFIEUR XGBOOST
# ============================================================

species_classifier = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    tree_method="hist",
    device="cuda",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=8,
)

# ============================================================
# 7. ENTRAÎNEMENT DU CLASSIFIEUR
# ============================================================

# species_classifier.fit(
#     X_train_selected,
#     y_train_encoded,
#     sample_weight=classifier_sample_weights,
# )

species_classifier.fit(
    X_train_selected,
    y_train_encoded,
)


# # ============================================================
# # 9. ÉVALUATION DU CLASSIFIEUR
# # ============================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Prédictions
y_species_pred_encoded = species_classifier.predict(X_test_selected)

# Retour aux labels d'origine
y_species_pred = label_encoder.inverse_transform(
    y_species_pred_encoded
)

# Métriques
accuracy = accuracy_score(
    y_test_all,
    y_species_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test_all,
    y_species_pred
)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Balanced accuracy : {balanced_accuracy:.4f}")

# Rapport par classe
print(
    classification_report(
        y_test_all,
        y_species_pred,
        labels=label_encoder.classes_,
        zero_division=0,
    )
)

import matplotlib.pyplot as plt

# Matrice de confusion
cm = confusion_matrix(
    y_test_all,
    y_species_pred,
    labels=label_encoder.classes_,
)

# Normalisation par classe réelle
cm_normalized = confusion_matrix(
    y_test_all,
    y_species_pred,
    labels=label_encoder.classes_,
    normalize="true",
)

print(cm)

# Affichage
fig, ax = plt.subplots(figsize=(12, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=label_encoder.classes_,
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    xticks_rotation=45,
    colorbar=True,
)

plt.title("Matrice de confusion normalisée – XGBoost")
plt.tight_layout()
plt.show()

# evaluate_classification_model(
#     model=species_classifier,
#     y_test=y_test,
#     y_pred=y_species_pred,
#     feature_names=classifier_feature_names,
# )

# classifier_importance = pd.DataFrame({
#     "feature": classifier_feature_names,
#     "importance": species_classifier.feature_importances_,
# }).sort_values(
#     "importance",
#     ascending=False,
# ).reset_index(drop=True)

# print("\nImportance des variables XGBoost :")
# print(classifier_importance)

