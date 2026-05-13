
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score


# ============================================================
# 5) RECLASSIFICATION DU MASQUE GT EN n CLASSES
# ============================================================

import numpy as np

def reclassify_gt_to_stress_classes(
    gt_mask,
    stress_groups,
    class_map,
    excluded_labels=(0, 10, 11)
):
    """
    Reclassifie le masque GT original en classes de stress.
    stress_groups contient des noms de classes.
    class_map convertit les noms en IDs numériques.
    """

    out = np.zeros_like(gt_mask, dtype=np.uint8)

    for stress_id, class_names in stress_groups.items():
        stress_id = int(stress_id)

        for class_name in class_names:
            original_id = class_map[class_name]
            out[gt_mask == original_id] = stress_id

    for lab in excluded_labels:
        out[gt_mask == lab] = 0

    return out


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

def run_random_forest_classification(X_train, X_test, y_train, y_test, STRESS_CLASS_NAMES):
    

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

