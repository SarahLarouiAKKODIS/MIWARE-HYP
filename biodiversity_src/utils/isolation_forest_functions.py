from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


def train_isolation_forests_by_class(
    X_train,
    y_train,
    feature_names,
    threshold_quantile=0.95,
    min_samples_per_class=500,
    max_samples_per_class=50000,
    n_estimators=300,
    max_samples="auto",
    random_state=42,
):
    """
    Entraîne un modèle Isolation Forest pour chaque classe/essence.

    Parameters
    ----------
    X_train : np.ndarray
        Matrice des caractéristiques de forme :
        (n_pixels, n_features).

    y_train : np.ndarray
        Classe de chaque pixel d'apprentissage.

    feature_names : list[str]
        Noms des variables, dans le même ordre que les colonnes de X_train.

    threshold_quantile : float
        Quantile des scores d'anomalie d'apprentissage utilisé comme seuil.

        Exemple :
        0.95 signifie que les 5 % des pixels d'apprentissage les plus
        atypiques dépassent le seuil.

    min_samples_per_class : int
        Nombre minimal de pixels requis pour entraîner un modèle par classe.

    max_samples_per_class : int | None
        Nombre maximal de pixels utilisés par classe pour limiter le
        temps de calcul. None signifie qu'ils sont tous utilisés.

    n_estimators : int
        Nombre d'arbres dans chaque Isolation Forest.

    max_samples : "auto" | int | float
        Nombre d'échantillons utilisés pour construire chaque arbre.

    random_state : int
        Graine aléatoire.

    Returns
    -------
    isolation_models : dict
        Modèle Isolation Forest associé à chaque classe.

    thresholds : dict
        Seuil d'anomalie associé à chaque classe.

    training_summary : pd.DataFrame
        Tableau récapitulatif de l'entraînement.
    """

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    feature_names = list(feature_names)

    if X_train.ndim != 2:
        raise ValueError(
            f"X_train doit être une matrice 2D, reçu : {X_train.shape}"
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train possède {X_train.shape[1]} colonnes, "
            f"mais feature_names contient {len(feature_names)} noms."
        )

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "X_train et y_train ne contiennent pas "
            "le même nombre d'échantillons."
        )

    if not 0 < threshold_quantile < 1:
        raise ValueError(
            "threshold_quantile doit être strictement compris entre 0 et 1."
        )

    rng = np.random.default_rng(random_state)

    isolation_models = {}
    thresholds = {}
    score_references = {}
    summary_rows = []

    classes = np.unique(y_train)

    for class_label in classes:

        print("\n" + "=" * 70)
        print(f"ENTRAÎNEMENT ISOLATION FOREST — CLASSE {class_label}")
        print("=" * 70)

        class_mask = y_train == class_label
        X_class = X_train[class_mask]

        # Suppression des observations contenant NaN ou infini
        finite_mask = np.all(np.isfinite(X_class), axis=1)
        X_class = X_class[finite_mask]

        n_available = len(X_class)

        if n_available < min_samples_per_class:
            print(
                f"Classe ignorée : seulement {n_available} pixels valides, "
                f"minimum requis = {min_samples_per_class}."
            )

            summary_rows.append({
                "class": class_label,
                "status": "ignored",
                "n_available": n_available,
                "n_used": 0,
                "threshold": np.nan,
                "score_mean": np.nan,
                "score_std": np.nan,
                "score_min": np.nan,
                "score_max": np.nan,
            })

            continue

        # Sous-échantillonnage éventuel
        if (
            max_samples_per_class is not None
            and n_available > max_samples_per_class
        ):
            selected = rng.choice(
                n_available,
                size=max_samples_per_class,
                replace=False,
            )

            X_fit = X_class[selected]
        else:
            X_fit = X_class

        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )

        model.fit(X_fit)

        # score_samples : plus faible = plus anormal
        # On inverse le signe pour obtenir :
        # plus élevé = plus anormal
        training_anomaly_scores = -model.score_samples(X_fit)

        score_references[class_label] = np.sort(
            training_anomaly_scores.astype(np.float32)
        )

        threshold = float(
            np.quantile(
                training_anomaly_scores,
                threshold_quantile,
            )
        )

        isolation_models[class_label] = model
        thresholds[class_label] = threshold

        predicted_training_anomalies = (
            training_anomaly_scores > threshold
        )

        anomaly_rate = predicted_training_anomalies.mean()

        summary_rows.append({
            "class": class_label,
            "status": "trained",
            "n_available": n_available,
            "n_used": len(X_fit),
            "threshold_quantile": threshold_quantile,
            "threshold": threshold,
            "training_anomaly_rate": anomaly_rate,
            "score_mean": float(np.mean(training_anomaly_scores)),
            "score_std": float(np.std(training_anomaly_scores)),
            "score_min": float(np.min(training_anomaly_scores)),
            "score_max": float(np.max(training_anomaly_scores)),
        })

        print("Pixels disponibles :", n_available)
        print("Pixels utilisés    :", len(X_fit))
        print(
            f"Seuil quantile {threshold_quantile:.2f} : "
            f"{threshold:.6f}"
        )
        print(
            f"Taux d'anomalies dans le train : "
            f"{100 * anomaly_rate:.2f} %"
        )

    training_summary = pd.DataFrame(summary_rows)

    return (
        isolation_models,
        thresholds,
        score_references,
        training_summary,
    )


def predict_stress_by_class(
    X,
    species_labels,
    isolation_models,
    thresholds,
):
    """
    Calcule le score d'anomalie avec le modèle propre à chaque essence.

    Parameters
    ----------
    X : np.ndarray
        Matrice de forme (n_pixels, n_features).

    species_labels : np.ndarray
        Essence associée à chaque pixel.
        En application réelle, il s'agit généralement des classes
        prédites par le classifieur d'essences.

    isolation_models : dict
        Modèles entraînés par classe.

    thresholds : dict
        Seuils d'anomalie propres à chaque classe.

    Returns
    -------
    anomaly_scores : np.ndarray
        Score continu. Plus il est élevé, plus le pixel est atypique.

    stress_prediction : np.ndarray
        1 = anomalie potentielle/stress potentiel
        0 = comportement conforme à la référence
        -1 = absence de modèle ou pixel invalide
    """

    X = np.asarray(X, dtype=np.float32)
    species_labels = np.asarray(species_labels)

    if X.shape[0] != len(species_labels):
        raise ValueError(
            "X et species_labels doivent avoir le même nombre de pixels."
        )

    anomaly_scores = np.full(
        X.shape[0],
        np.nan,
        dtype=np.float32,
    )

    stress_prediction = np.full(
        X.shape[0],
        -1,
        dtype=np.int8,
    )

    valid_rows = np.all(np.isfinite(X), axis=1)

    for class_label in np.unique(species_labels):

        if class_label not in isolation_models:
            print(
                f"Aucun Isolation Forest disponible pour "
                f"la classe {class_label}."
            )
            continue

        class_mask = (
            (species_labels == class_label)
            & valid_rows
        )

        if not np.any(class_mask):
            continue

        X_class = X[class_mask]

        model = isolation_models[class_label]
        threshold = thresholds[class_label]

        scores = -model.score_samples(X_class)

        anomaly_scores[class_mask] = scores.astype(np.float32)

        stress_prediction[class_mask] = (
            scores > threshold
        ).astype(np.int8)

    return anomaly_scores, stress_prediction


def anomaly_summary_by_class(
    species_labels,
    anomaly_scores,
    stress_prediction,
):
    rows = []

    for class_label in np.unique(species_labels):

        mask = (
            (species_labels == class_label)
            & np.isfinite(anomaly_scores)
            & (stress_prediction >= 0)
        )

        if not np.any(mask):
            continue

        class_scores = anomaly_scores[mask]
        class_stress = stress_prediction[mask]

        rows.append({
            "class": class_label,
            "n_pixels": int(mask.sum()),
            "n_anomalies": int(np.sum(class_stress == 1)),
            "anomaly_rate": float(np.mean(class_stress == 1)),
            "mean_score": float(np.mean(class_scores)),
            "median_score": float(np.median(class_scores)),
            "score_q95": float(np.quantile(class_scores, 0.95)),
            "score_max": float(np.max(class_scores)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("anomaly_rate", ascending=False)
        .reset_index(drop=True)
    )



import numpy as np
import pandas as pd


def combine_xgboost_and_isolation_forest(
    classifier,
    label_encoder,
    X_classifier,
    X_anomaly,
    isolation_models,
    isolation_thresholds,
    isolation_score_references,
    min_confidence=0.70,
    high_confidence=0.85,
    combined_threshold=0.90,
    confidence_power=1.0,
):
    """
    Combine la confiance XGBoost et le score d'anomalie Isolation Forest.

    Parameters
    ----------
    classifier
        Classifieur entraîné disposant de predict_proba().

    label_encoder : LabelEncoder
        Encodeur utilisé pour les classes du classifieur.

    X_classifier : np.ndarray
        Variables utilisées par XGBoost.

    X_anomaly : np.ndarray
        Variables utilisées par les Isolation Forest.

    isolation_models : dict
        Une Isolation Forest par classe originale.

    isolation_thresholds : dict
        Seuil d'anomalie propre à chaque classe.

    isolation_score_references : dict
        Distribution triée des scores d'apprentissage par classe.

    min_confidence : float
        Sous cette valeur, la classification est considérée comme incertaine.

    high_confidence : float
        Au-dessus de cette valeur, une anomalie est considérée comme
        fortement compatible avec l'essence prédite.

    combined_threshold : float
        Seuil appliqué au score combiné.

    confidence_power : float
        Poids donné à la confiance dans le score combiné.

    Returns
    -------
    result : pd.DataFrame
        Résultats détaillés par pixel.
    """

    X_classifier = np.asarray(X_classifier, dtype=np.float32)
    X_anomaly = np.asarray(X_anomaly, dtype=np.float32)

    if X_classifier.shape[0] != X_anomaly.shape[0]:
        raise ValueError(
            "X_classifier et X_anomaly doivent contenir "
            "le même nombre de pixels."
        )

    n_samples = X_classifier.shape[0]

    # ========================================================
    # CLASSIFICATION XGBOOST
    # ========================================================

    probabilities = classifier.predict_proba(X_classifier)

    predicted_encoded = np.argmax(probabilities, axis=1)
    predicted_labels = label_encoder.inverse_transform(
        predicted_encoded
    )

    classification_confidence = np.max(
        probabilities,
        axis=1,
    ).astype(np.float32)

    # Différence entre les deux meilleures probabilités
    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    probability_margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    ).astype(np.float32)

    # ========================================================
    # TABLEAUX DE SORTIE
    # ========================================================

    anomaly_score = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )

    anomaly_percentile = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )

    anomaly_threshold = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )

    is_anomaly_if = np.full(
        n_samples,
        False,
        dtype=bool,
    )

    valid_rows = np.all(
        np.isfinite(X_anomaly),
        axis=1,
    )

    # ========================================================
    # ISOLATION FOREST PAR CLASSE PRÉDITE
    # ========================================================

    for class_label in np.unique(predicted_labels):

        if class_label not in isolation_models:
            print(
                f"Aucun modèle Isolation Forest pour "
                f"la classe {class_label}."
            )
            continue

        class_mask = (
            (predicted_labels == class_label)
            & valid_rows
        )

        if not np.any(class_mask):
            continue

        model = isolation_models[class_label]
        threshold = isolation_thresholds[class_label]

        scores = -model.score_samples(
            X_anomaly[class_mask]
        )

        reference_scores = np.asarray(
            isolation_score_references[class_label]
        )

        # Rang empirique dans la distribution d'apprentissage
        # Exemple : 0,98 signifie plus atypique que 98 % du train.
        percentiles = np.searchsorted(
            reference_scores,
            scores,
            side="right",
        ) / len(reference_scores)

        anomaly_score[class_mask] = scores
        anomaly_percentile[class_mask] = percentiles
        anomaly_threshold[class_mask] = threshold
        is_anomaly_if[class_mask] = scores > threshold

    # ========================================================
    # COMBINAISON DES DEUX INFORMATIONS
    # ========================================================

    combined_score = (
        anomaly_percentile
        * np.power(
            classification_confidence,
            confidence_power,
        )
    ).astype(np.float32)

    # Anomalie compatible avec l'essence prédite
    reliable_anomaly = (
        is_anomaly_if
        & (classification_confidence >= high_confidence)
        & (combined_score >= combined_threshold)
    )

    # Pixel anormal, mais classe peu fiable
    possible_misclassification = (
        is_anomaly_if
        & (classification_confidence < min_confidence)
    )

    # Cas intermédiaire à examiner
    uncertain_anomaly = (
        is_anomaly_if
        & ~reliable_anomaly
        & ~possible_misclassification
    )

    status = np.full(
        n_samples,
        "normal",
        dtype=object,
    )

    status[uncertain_anomaly] = "anomalie_incertaine"

    status[possible_misclassification] = (
        "classification_incertaine"
    )

    status[reliable_anomaly] = (
        "anomalie_fiable"
    )

    status[~valid_rows] = "pixel_invalide"

    result = pd.DataFrame({
        "predicted_class": predicted_labels,
        "classification_confidence": classification_confidence,
        "probability_margin": probability_margin,
        "anomaly_score": anomaly_score,
        "anomaly_percentile": anomaly_percentile,
        "anomaly_threshold": anomaly_threshold,
        "isolation_forest_anomaly": is_anomaly_if,
        "combined_score": combined_score,
        "reliable_anomaly": reliable_anomaly,
        "possible_misclassification": possible_misclassification,
        "status": status,
    })

    return result