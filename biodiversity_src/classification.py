### Le script `main_biodiv_indice_calculation.py` doit être exécuté 
# préalablement afin de générer les indices utilisés pour la classification.

from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import train_test_split

from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import joblib
from utils.classification_functions import evaluate_classification_model

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

feature_names = [
    "NDVI",
    "NDRE",
    "NDWI",
    "PRI",
    "ARI",
    "EVI"
    "NBR"
]


X_list = []
y_list = []
groups = []

group_id = 0

for config_path in config_files:

    with open(config_path) as f:
        config = json.load(f)

    data = np.load(Path(config["Path_res"]) / "Indice_values/dataset_indices.npz")

    X = data["X"]
    y = data["y"]

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


############### Validation croisée GroupKFold #############


from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

gkf = GroupKFold(n_splits=4)

scores = []

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X_all, y_all, groups)
):

    print(f"\nFold {fold+1}")

    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # IDs des groupes
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])

    print("group_train", train_groups)
    print("group_test", test_groups)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42
    )

    print('labels dans y_train = ', np.unique(y_train))

    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model.fit(X_train, y_train_enc, sample_weight=sample_weights)

    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)

    scores.append(acc)

    evaluate_classification_model(
    model,
    y_test,
    y_pred,
    feature_names
)

print("\nMean accuracy :", np.mean(scores))














# # Charger les datasets train
# X_train_list = []
# y_train_list = []

# for path in train_paths:

#     data = np.load(path)

#     X_train_list.append(data["X"])
#     y_train_list.append(data["y"])

# # Charger les datasets test
# X_test_list = []
# y_test_list = []

# for path in test_paths:

#     data = np.load(path)

#     X_test_list.append(data["X"])
#     y_test_list.append(data["y"])

# # Fusionner uniquement à l'intérieur de chaque split
# X_train = np.concatenate(X_train_list, axis=0)
# y_train = np.concatenate(y_train_list, axis=0)

# X_test = np.concatenate(X_test_list, axis=0)
# y_test = np.concatenate(y_test_list, axis=0)

# print("Train files:")
# for p in train_paths:
#     print(" -", p)

# print("\nTest files:")
# for p in test_paths:
#     print(" -", p)

# print("\nTrain shape:", X_train.shape, y_train.shape)
# print("Test shape :", X_test.shape, y_test.shape)


# ####### CLASSIFICATION

# # cuML aime les types simples
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)

# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=20,
#     random_state=42,
#     n_streams=8
# )

# # X_train_selected = X_train[:, [0, 1, 3, 4, 5]]
# # X_test_selected = X_test[:, [0, 1, 3, 4, 5]]

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# y_pred = np.asarray(y_pred)


#joblib.dump(rf_gpu, "rf_gpu_cuml.joblib")

#######################################################

# from xgboost import XGBClassifier

# model = XGBClassifier(
#     n_estimators=500,
#     max_depth=8,
#     learning_rate=0.05,
#     tree_method="hist",
#     device="cuda",
#     eval_metric="mlogloss",
#     random_state=42
# )

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)

# # X_train_selected = X_train[:, [1, 2, 4, 5]]
# # X_test_selected = X_test[:, [1, 2, 4, 5]]

# X_train_selected = X_train
# X_test_selected = X_test

# model.fit(X_train_selected, y_train_enc)

# y_pred_enc = model.predict(X_test_selected)
# # Revenir aux labels originaux
# y_pred = le.inverse_transform(y_pred_enc)

