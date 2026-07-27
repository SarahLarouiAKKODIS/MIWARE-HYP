from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from utils import subsample_by_class

def evaluate_feature_subset(
    X_train_all,
    y_train_all,
    X_test,
    y_test,
    selected_indices,
    max_per_class=30000,
    random_state=42,
):
    X_train_subset_all = X_train_all[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    X_train, y_train = subsample_by_class(
        X_train_subset_all,
        y_train_all,
        max_per_class=max_per_class,
    )

    keep_test_known = np.isin(y_test, np.unique(y_train))
    X_test_subset = X_test_subset[keep_test_known]
    y_test_known = y_test[keep_test_known]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test_known)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_enc,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train_enc,
        sample_weight=sample_weights,
    )

    y_pred_enc = model.predict(X_test_subset)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_known, y_pred)
    bal_acc = balanced_accuracy_score(y_test_known, y_pred)

    return acc, bal_acc, model



def sequential_forward_selection_xgb(
    X_train_all,
    y_train_all,
    X_test,
    y_test,
    feature_names,
    max_features=None,
    min_improvement=0.001,
    max_per_class=30000,
):
    n_features = len(feature_names)

    if max_features is None:
        max_features = n_features

    selected_indices = []
    remaining_indices = list(range(n_features))

    best_global_score = -np.inf
    best_global_indices = []
    best_global_features = []

    history = []
    best_per_step = []

    for step in range(max_features):

        print("\n==============================")
        print(f"Étape {step + 1}")
        print("Features sélectionnées :", [feature_names[i] for i in selected_indices])
        print("==============================")

        step_results = []

        for idx in remaining_indices:

            candidate_indices = selected_indices + [idx]
            candidate_features = [feature_names[i] for i in candidate_indices]

            print(f"\nTest ajout : {feature_names[idx]}")

            acc, bal_acc, _ = evaluate_feature_subset(
                X_train_all=X_train_all,
                y_train_all=y_train_all,
                X_test=X_test,
                y_test=y_test,
                selected_indices=candidate_indices,
                max_per_class=max_per_class,
            )

            print(f"Accuracy = {acc:.4f} | Balanced accuracy = {bal_acc:.4f}")

            result = {
                "step": step + 1,
                "n_features": len(candidate_indices),
                "added_feature": feature_names[idx],
                "selected_features": ", ".join(candidate_features),
                "selected_indices": candidate_indices.copy(),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
            }

            step_results.append(result)
            history.append(result)

        step_df = pd.DataFrame(step_results)

        best_row = step_df.sort_values(
            "balanced_accuracy",
            ascending=False,
        ).iloc[0]

        best_added_feature = best_row["added_feature"]
        best_added_idx = feature_names.index(best_added_feature)
        best_step_score = best_row["balanced_accuracy"]
        best_step_accuracy = best_row["accuracy"]

        improvement = best_step_score - best_global_score

        print("\n>>> Meilleure feature ajoutée :", best_added_feature)
        print(f">>> Accuracy : {best_step_accuracy:.4f}")
        print(f">>> Balanced accuracy : {best_step_score:.4f}")
        print(f">>> Amélioration : {improvement:.4f}")

        best_per_step.append({
            "step": step + 1,
            "n_features": len(selected_indices) + 1,
            "added_feature": best_added_feature,
            "selected_features": best_row["selected_features"],
            "selected_indices": best_row["selected_indices"],
            "accuracy": best_step_accuracy,
            "balanced_accuracy": best_step_score,
            "improvement": improvement,
        })

        # if improvement < min_improvement:
        #     print("\nArrêt : amélioration trop faible.")
        #     break

        selected_indices.append(best_added_idx)
        remaining_indices.remove(best_added_idx)

        if best_step_score > best_global_score:
            best_global_score = best_step_score
            best_global_indices = selected_indices.copy()
            best_global_features = [
                feature_names[i]
                for i in best_global_indices
            ]

    history_df = pd.DataFrame(history)
    best_per_step_df = pd.DataFrame(best_per_step)

    summary_df = pd.DataFrame({
        "best_balanced_accuracy": [best_global_score],
        "n_selected_features": [len(best_global_features)],
        "selected_features": [", ".join(best_global_features)],
        "selected_indices": [best_global_indices],
    })

    print("\n==============================")
    print("SÉLECTION TERMINÉE")
    print("==============================")
    print("Meilleure balanced accuracy :", best_global_score)
    print("Meilleures features :", best_global_features)

    return (
        best_global_indices,
        best_global_features,
        best_global_score,
        history_df,
        best_per_step_df,
        summary_df,
    )





import numpy as np
import pandas as pd


def sequential_floating_forward_selection_xgb(
    X_train_all,
    y_train_all,
    X_test,
    y_test,
    feature_names,
    max_features=None,
    min_improvement=1e-6,
    max_per_class=30000,
    random_state=42,
    max_iterations=None,
):
    """
    Sequential Floating Forward Selection (SFFS) avec XGBoost.

    Principe
    --------
    1. Ajouter la meilleure variable restante.
    2. Tester le retrait de chaque variable sélectionnée.
    3. Accepter un retrait si le nouveau sous-ensemble améliore le meilleur
       score déjà obtenu pour cette taille de sous-ensemble.
    4. Répéter les retraits tant qu'ils sont bénéfiques.
    5. Continuer avec une nouvelle inclusion.

    Une mémoire des sous-ensembles visités empêche les cycles tels que :

        + BD1940
        - BD1940
        + BD1940
        - BD1940
        ...

    Parameters
    ----------
    X_train_all : np.ndarray
        Données d'apprentissage, shape (n_samples, n_features).

    y_train_all : np.ndarray
        Labels d'apprentissage.

    X_test : np.ndarray
        Données de test, shape (n_samples, n_features).

    y_test : np.ndarray
        Labels de test.

    feature_names : list[str]
        Noms des variables, dans le même ordre que les colonnes de X.

    max_features : int | None
        Nombre maximal de variables dans le sous-ensemble courant.
        Si None, toutes les variables peuvent être explorées.

    min_improvement : float
        Gain minimal de Balanced Accuracy nécessaire pour considérer
        qu'un sous-ensemble est meilleur.

    max_per_class : int
        Nombre maximal d'échantillons d'apprentissage par classe.

    random_state : int
        Graine aléatoire transmise à evaluate_feature_subset().

    max_iterations : int | None
        Limite de sécurité du nombre d'itérations principales.
        Si None, vaut 20 × le nombre de variables.

    Returns
    -------
    best_indices : list[int]
        Indices du meilleur sous-ensemble rencontré.

    best_features : list[str]
        Noms des variables du meilleur sous-ensemble.

    best_score : float
        Meilleure Balanced Accuracy.

    history_df : pd.DataFrame
        Toutes les inclusions et exclusions évaluées.

    progression_df : pd.DataFrame
        Uniquement les opérations acceptées.

    best_by_size_df : pd.DataFrame
        Meilleur sous-ensemble obtenu pour chaque nombre de variables.

    summary_df : pd.DataFrame
        Résumé du meilleur résultat.
    """

    # ========================================================
    # VÉRIFICATIONS
    # ========================================================

    X_train_all = np.asarray(X_train_all)
    X_test = np.asarray(X_test)
    y_train_all = np.asarray(y_train_all)
    y_test = np.asarray(y_test)

    feature_names = list(feature_names)
    n_features = len(feature_names)

    if X_train_all.ndim != 2:
        raise ValueError(
            f"X_train_all doit être une matrice 2D, reçu {X_train_all.shape}"
        )

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test doit être une matrice 2D, reçu {X_test.shape}"
        )

    if X_train_all.shape[1] != n_features:
        raise ValueError(
            f"X_train_all contient {X_train_all.shape[1]} colonnes, "
            f"mais feature_names contient {n_features} noms."
        )

    if X_test.shape[1] != n_features:
        raise ValueError(
            f"X_test contient {X_test.shape[1]} colonnes, "
            f"mais feature_names contient {n_features} noms."
        )

    if len(y_train_all) != len(X_train_all):
        raise ValueError(
            "X_train_all et y_train_all n'ont pas le même nombre d'échantillons."
        )

    if len(y_test) != len(X_test):
        raise ValueError(
            "X_test et y_test n'ont pas le même nombre d'échantillons."
        )

    if len(set(feature_names)) != len(feature_names):
        duplicates = sorted({
            name
            for name in feature_names
            if feature_names.count(name) > 1
        })

        raise ValueError(
            f"Les noms de variables doivent être uniques. Doublons : {duplicates}"
        )

    if max_features is None:
        max_features = n_features

    max_features = min(int(max_features), n_features)

    if max_features < 1:
        raise ValueError("max_features doit être supérieur ou égal à 1.")

    if max_iterations is None:
        max_iterations = max(20 * n_features, 100)

    # ========================================================
    # STRUCTURES INTERNES
    # ========================================================

    selected_indices = []

    # Cache des scores pour ne pas réentraîner plusieurs fois
    # exactement le même sous-ensemble.
    evaluation_cache = {}

    # Sous-ensembles ayant déjà été acceptés.
    visited_subsets = {tuple()}

    # Meilleur sous-ensemble rencontré pour chaque cardinalité.
    best_by_size = {}

    history = []
    progression = []

    best_global_score = -np.inf
    best_global_accuracy = np.nan
    best_global_indices = []

    operation_id = 0
    iteration = 0

    # ========================================================
    # FONCTIONS INTERNES
    # ========================================================

    def normalize_subset(indices):
        """Normalise un sous-ensemble sous forme de tuple trié."""

        return tuple(sorted(set(indices)))

    def subset_feature_names(indices):
        """Retourne les noms correspondant à un sous-ensemble."""

        return [
            feature_names[idx]
            for idx in normalize_subset(indices)
        ]

    def evaluate_subset(indices):
        """
        Évalue un sous-ensemble de variables ou retourne son score
        depuis le cache.
        """

        subset_key = normalize_subset(indices)

        if len(subset_key) == 0:
            raise ValueError(
                "Un sous-ensemble vide ne peut pas être évalué."
            )

        if subset_key not in evaluation_cache:

            accuracy, balanced_accuracy, _ = evaluate_feature_subset(
                X_train_all=X_train_all,
                y_train_all=y_train_all,
                X_test=X_test,
                y_test=y_test,
                selected_indices=list(subset_key),
                max_per_class=max_per_class,
                random_state=random_state,
            )

            evaluation_cache[subset_key] = {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
            }

        return evaluation_cache[subset_key]

    def update_best_results(
        indices,
        accuracy,
        balanced_accuracy,
    ):
        """
        Met à jour :
        - le meilleur sous-ensemble de cette taille ;
        - le meilleur sous-ensemble global.
        """

        nonlocal best_global_score
        nonlocal best_global_accuracy
        nonlocal best_global_indices

        subset_key = normalize_subset(indices)
        subset_size = len(subset_key)

        current_best_size = best_by_size.get(subset_size)

        if (
            current_best_size is None
            or balanced_accuracy
            > current_best_size["balanced_accuracy"] + min_improvement
        ):
            best_by_size[subset_size] = {
                "indices": list(subset_key),
                "features": subset_feature_names(subset_key),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
            }

        if balanced_accuracy > best_global_score + min_improvement:

            best_global_score = float(balanced_accuracy)
            best_global_accuracy = float(accuracy)
            best_global_indices = list(subset_key)

    def add_history_row(
        *,
        iteration_number,
        operation,
        feature_idx,
        candidate_indices,
        scores,
        accepted=False,
    ):
        """Ajoute une évaluation à l'historique."""

        candidate_key = normalize_subset(candidate_indices)

        row = {
            "iteration": iteration_number,
            "operation": operation,
            "feature": feature_names[feature_idx],
            "n_features": len(candidate_key),
            "selected_features": ", ".join(
                subset_feature_names(candidate_key)
            ),
            "selected_indices": list(candidate_key),
            "accuracy": scores["accuracy"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "accepted": accepted,
        }

        history.append(row)

        return len(history) - 1

    def add_progression_row(
        *,
        iteration_number,
        operation,
        feature_idx,
        current_indices,
        scores,
    ):
        """Ajoute une opération acceptée à la progression."""

        nonlocal operation_id

        operation_id += 1

        current_key = normalize_subset(current_indices)

        progression.append({
            "operation_id": operation_id,
            "iteration": iteration_number,
            "operation": operation,
            "feature": feature_names[feature_idx],
            "n_features": len(current_key),
            "selected_features": ", ".join(
                subset_feature_names(current_key)
            ),
            "selected_indices": list(current_key),
            "accuracy": scores["accuracy"],
            "balanced_accuracy": scores["balanced_accuracy"],
        })

    # ========================================================
    # BOUCLE PRINCIPALE
    # ========================================================

    while iteration < max_iterations:

        iteration += 1

        current_key = normalize_subset(selected_indices)

        print("\n" + "=" * 70)
        print(f"ITÉRATION SFFS {iteration}")
        print(
            "Sous-ensemble courant :",
            subset_feature_names(current_key),
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Critère de fin : taille maximale atteinte
        # ----------------------------------------------------

        if len(selected_indices) >= max_features:
            print(
                f"Arrêt : le nombre maximal de variables "
                f"({max_features}) est atteint."
            )
            break

        # ====================================================
        # 1. PHASE FORWARD
        # ====================================================

        remaining_indices = [
            idx
            for idx in range(n_features)
            if idx not in selected_indices
        ]

        if not remaining_indices:
            print("Arrêt : aucune variable restante.")
            break

        forward_candidates = []

        for idx_to_add in remaining_indices:

            candidate_indices = selected_indices + [idx_to_add]
            candidate_key = normalize_subset(candidate_indices)

            # Empêche de revenir sur un sous-ensemble déjà accepté.
            if candidate_key in visited_subsets:
                continue

            scores = evaluate_subset(candidate_indices)

            history_position = add_history_row(
                iteration_number=iteration,
                operation="test_add",
                feature_idx=idx_to_add,
                candidate_indices=candidate_indices,
                scores=scores,
            )

            forward_candidates.append({
                "feature_idx": idx_to_add,
                "indices": list(candidate_key),
                "scores": scores,
                "history_position": history_position,
            })

            print(
                f"Ajout testé : {feature_names[idx_to_add]:30s} | "
                f"Accuracy = {scores['accuracy']:.4f} | "
                f"Balanced accuracy = "
                f"{scores['balanced_accuracy']:.4f}"
            )

        if not forward_candidates:
            print(
                "Arrêt : aucun nouveau sous-ensemble non visité "
                "n'est disponible pour l'inclusion."
            )
            break

        best_forward = max(
            forward_candidates,
            key=lambda candidate: (
                candidate["scores"]["balanced_accuracy"],
                candidate["scores"]["accuracy"],
            ),
        )

        added_idx = best_forward["feature_idx"]
        selected_indices = best_forward["indices"]

        current_scores = best_forward["scores"]
        current_key = normalize_subset(selected_indices)

        visited_subsets.add(current_key)

        history[
            best_forward["history_position"]
        ]["accepted"] = True

        add_progression_row(
            iteration_number=iteration,
            operation="add",
            feature_idx=added_idx,
            current_indices=selected_indices,
            scores=current_scores,
        )

        update_best_results(
            indices=selected_indices,
            accuracy=current_scores["accuracy"],
            balanced_accuracy=current_scores["balanced_accuracy"],
        )

        print(
            f"\n>>> Variable ajoutée : {feature_names[added_idx]}"
        )
        print(
            f">>> Balanced accuracy : "
            f"{current_scores['balanced_accuracy']:.6f}"
        )

        # ====================================================
        # 2. PHASE FLOATING BACKWARD
        # ====================================================

        while len(selected_indices) > 1:

            backward_candidates = []

            for idx_to_remove in list(selected_indices):

                candidate_indices = [
                    idx
                    for idx in selected_indices
                    if idx != idx_to_remove
                ]

                candidate_key = normalize_subset(candidate_indices)

                # Évite les cycles vers un sous-ensemble déjà accepté.
                if candidate_key in visited_subsets:
                    continue

                scores = evaluate_subset(candidate_indices)

                history_position = add_history_row(
                    iteration_number=iteration,
                    operation="test_remove",
                    feature_idx=idx_to_remove,
                    candidate_indices=candidate_indices,
                    scores=scores,
                )

                backward_candidates.append({
                    "feature_idx": idx_to_remove,
                    "indices": list(candidate_key),
                    "scores": scores,
                    "history_position": history_position,
                })

                print(
                    f"Retrait testé : "
                    f"{feature_names[idx_to_remove]:28s} | "
                    f"Accuracy = {scores['accuracy']:.4f} | "
                    f"Balanced accuracy = "
                    f"{scores['balanced_accuracy']:.4f}"
                )

            if not backward_candidates:
                print(
                    "\nAucun nouveau sous-ensemble non visité "
                    "n'est disponible pour un retrait."
                )
                break

            best_backward = max(
                backward_candidates,
                key=lambda candidate: (
                    candidate["scores"]["balanced_accuracy"],
                    candidate["scores"]["accuracy"],
                ),
            )

            candidate_size = len(best_backward["indices"])
            previous_best_smaller = best_by_size.get(candidate_size)

            # Le retrait est accepté uniquement s'il améliore le
            # meilleur sous-ensemble déjà rencontré pour cette taille.
            if previous_best_smaller is not None:

                threshold = (
                    previous_best_smaller["balanced_accuracy"]
                    + min_improvement
                )

                if (
                    best_backward["scores"]["balanced_accuracy"]
                    <= threshold
                ):
                    print(
                        "\nAucun retrait accepté : le meilleur "
                        f"sous-ensemble testé de {candidate_size} variables "
                        "n'améliore pas le meilleur résultat déjà obtenu "
                        "pour cette cardinalité."
                    )
                    break

            removed_idx = best_backward["feature_idx"]
            selected_indices = best_backward["indices"]

            current_scores = best_backward["scores"]
            current_key = normalize_subset(selected_indices)

            visited_subsets.add(current_key)

            history[
                best_backward["history_position"]
            ]["accepted"] = True

            add_progression_row(
                iteration_number=iteration,
                operation="remove",
                feature_idx=removed_idx,
                current_indices=selected_indices,
                scores=current_scores,
            )

            update_best_results(
                indices=selected_indices,
                accuracy=current_scores["accuracy"],
                balanced_accuracy=current_scores["balanced_accuracy"],
            )

            print(
                f"\n<<< Variable retirée : "
                f"{feature_names[removed_idx]}"
            )
            print(
                f"<<< Balanced accuracy après retrait : "
                f"{current_scores['balanced_accuracy']:.6f}"
            )

            # La boucle continue afin d'autoriser plusieurs
            # retraits successifs.

    # ========================================================
    # SORTIES
    # ========================================================

    if iteration >= max_iterations:
        print(
            "\n[AVERTISSEMENT] Le nombre maximal d'itérations "
            f"a été atteint : {max_iterations}"
        )

    best_features = [
        feature_names[idx]
        for idx in best_global_indices
    ]

    history_df = pd.DataFrame(history)
    progression_df = pd.DataFrame(progression)

    best_by_size_rows = []

    for subset_size in sorted(best_by_size):

        result = best_by_size[subset_size]

        best_by_size_rows.append({
            "n_features": subset_size,
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "selected_indices": result["indices"],
            "selected_features": ", ".join(
                result["features"]
            ),
        })

    best_by_size_df = pd.DataFrame(best_by_size_rows)

    summary_df = pd.DataFrame({
        "best_accuracy": [best_global_accuracy],
        "best_balanced_accuracy": [best_global_score],
        "n_selected_features": [len(best_global_indices)],
        "selected_indices": [best_global_indices],
        "selected_features": [", ".join(best_features)],
        "n_unique_subsets_evaluated": [len(evaluation_cache)],
        "n_accepted_subsets": [len(visited_subsets) - 1],
        "n_iterations": [iteration],
    })

    print("\n" + "=" * 70)
    print("SFFS TERMINÉ")
    print("=" * 70)
    print(f"Meilleure Accuracy : {best_global_accuracy:.6f}")
    print(
        f"Meilleure Balanced Accuracy : "
        f"{best_global_score:.6f}"
    )
    print(
        f"Nombre optimal de variables : "
        f"{len(best_global_indices)}"
    )
    print("Variables sélectionnées :", best_features)
    print(
        "Nombre de sous-ensembles uniques évalués :",
        len(evaluation_cache),
    )

    return (
        best_global_indices,
        best_features,
        best_global_score,
        history_df,
        progression_df,
        best_by_size_df,
        summary_df,
    )