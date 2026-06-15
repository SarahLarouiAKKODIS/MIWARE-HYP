
import numpy as np 
from scipy.stats import kruskal


# ============================================================
# 7) STATS PAR CLASSE
# ============================================================

def compute_mean_indices_per_group(indices_dict, group_mask, STRESS_CLASS_NAMES, valid_pixels_mask=None, ignore_labels=(0,)):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    stats = {}

    for stress_id, stress_name in STRESS_CLASS_NAMES.items():
        if stress_id in ignore_labels:
            continue

        print('stress_id', stress_id)
        print('stress_name', stress_name)

        conditions = {
        "group_mask == stress_id": group_mask == stress_id,
        "valid_pixels_mask": valid_pixels_mask,
        "finite ndvi": np.isfinite(indices_dict["ndvi"]),
        "finite ndre": np.isfinite(indices_dict["ndre"]),
        "finite ndwi": np.isfinite(indices_dict["ndwi"]),
        "finite pri": np.isfinite(indices_dict["pri"]),
        "finite ari": np.isfinite(indices_dict["ari"]),
        "finite evi": np.isfinite(indices_dict["evi"]),
        "finite nbr": np.isfinite(indices_dict["nbr"])
        }

        print("Debug conditions :")
        for name, cond in conditions.items():
            print(f"{name:25s} -> {np.count_nonzero(cond)} pixels True")

        mask = (
            (group_mask == stress_id) &
            valid_pixels_mask &
            np.isfinite(indices_dict["ndvi"]) &
            np.isfinite(indices_dict["ndre"]) &
            np.isfinite(indices_dict["ndwi"]) &
            np.isfinite(indices_dict["pri"]) &
            np.isfinite(indices_dict["ari"]) &
            np.isfinite(indices_dict["evi"]) &
            np.isfinite(indices_dict["nbr"]) 
        )

        n = np.count_nonzero(mask)
        print("n", n)
        if n == 0:
            continue

        stats[stress_id] = {
            "class_name": stress_name,
            "n_pixels": int(n),
            "ndvi": float(np.nanmean(indices_dict["ndvi"][mask])),
            "ndre": float(np.nanmean(indices_dict["ndre"][mask])),
            "ndwi": float(np.nanmean(indices_dict["ndwi"][mask])),
            "pri": float(np.nanmean(indices_dict["pri"][mask])),
            "ari": float(np.nanmean(indices_dict["ari"][mask])),
            "evi": float(np.nanmean(indices_dict["evi"][mask])),
            "nbr": float(np.nanmean(indices_dict["nbr"][mask])),
            "ndvi_std": float(np.nanstd(indices_dict["ndvi"][mask])),
            "ndre_std": float(np.nanstd(indices_dict["ndre"][mask])),
            "ndwi_std": float(np.nanstd(indices_dict["ndwi"][mask])),
            "pri_std": float(np.nanstd(indices_dict["pri"][mask])),
            "ari_std": float(np.nanstd(indices_dict["ari"][mask])),
            "evi_std": float(np.nanstd(indices_dict["evi"][mask])),
            "nbr_std": float(np.nanstd(indices_dict["nbr"][mask]))
        }

    return stats


# ============================================================
# 8) COHÉRENCE ATTENDUE DES INDICES
# ============================================================

def check_stress_coherence(stats):
    """
    Vérifie la cohérence attendue entre les classes disponibles.

    Ordre attendu :
    1 = pas stressée
    2 = peu stressée
    3 = moyennement stressée
    4 = très stressée

    Attendu :
    - NDVI, NDRE, NDWI, PRI, EVI, NBR : décroissants avec le stress
    - ARI : croissant avec le stress
    """

    class_names = {
        1: "Pas stressée",
        2: "Peu stressée",
        3: "Moyennement stressée",
        4: "Très stressée"
    }

    expected_order = [1, 2, 3, 4]
    available_classes = [c for c in expected_order if c in stats]

    if len(available_classes) < 2:
        print("\nImpossible de tester la cohérence : moins de deux classes disponibles.")
        return

    missing_classes = [c for c in expected_order if c not in stats]

    print("\n=== Vérification de cohérence des indices ===")

    if missing_classes:
        print("Classes manquantes :", missing_classes)

    # Sens attendu des indices quand le stress augmente
    index_directions = {
        "ndvi": "decrease",
        "ndre": "decrease",
        "ndwi": "decrease",
        "pri": "decrease",
        "ari": "increase",
        "evi": "decrease",
        "nbr": "decrease"
    }

    for index_name, direction in index_directions.items():

        pairs = [
            (class_names[c], stats[c][index_name])
            for c in available_classes
            if index_name in stats[c]
        ]

        if len(pairs) < 2:
            print(f"{index_name.upper():5s} | Données insuffisantes")
            continue

        labels, values = zip(*pairs)

        if direction == "decrease":
            coherent = all(
                values[i] > values[i + 1]
                for i in range(len(values) - 1)
            )
        elif direction == "increase":
            coherent = all(
                values[i] < values[i + 1]
                for i in range(len(values) - 1)
            )

        values_text = " | ".join(
            f"{label}={value:.4f}"
            for label, value in zip(labels, values)
        )

        expected = "↓" if direction == "decrease" else "↑"

        print(
            f"{index_name.upper():5s} | "
            f"{values_text} | "
            f"Attendu={expected} | "
            f"Cohérent={coherent}"
        )



def check_stress_coherence_from_xy(
    X,
    y,
    feature_names,
    class_names=None,
    expected_order=None,
    index_directions=None,
):
    """
    Vérifie la cohérence attendue des indices à partir d'un dataset X/y.

    Parameters
    ----------
    X : np.ndarray
        Matrice de features, shape = (n_pixels, n_features).
    y : np.ndarray
        Labels de classes, shape = (n_pixels,).
    feature_names : list[str]
        Noms des colonnes de X, ex: ["ndvi", "ndre", "ndwi", ...].
    class_names : dict | None
        Dictionnaire {classe: nom}. Les clés peuvent être int ou str.
    expected_order : list[int] | None
        Ordre des classes selon l'augmentation du stress.
    index_directions : dict | None
        Sens attendu des indices.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X doit être 2D, reçu shape={X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y doit être 1D, reçu shape={y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X et y incompatibles : X={X.shape}, y={y.shape}")

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Nombre de features incohérent : "
            f"X.shape[1]={X.shape[1]}, len(feature_names)={len(feature_names)}"
        )

    feature_names = [f.lower() for f in feature_names]

    if class_names is None:
        class_names = {
            1: "Pas stressée",
            2: "Peu stressée",
            3: "Moyennement stressée",
            4: "Très stressée",
        }
    else:
        class_names = {
            int(k): v
            for k, v in class_names.items()
        }

    if expected_order is None:
        expected_order = sorted(class_names.keys())

    if index_directions is None:
        index_directions = {
            "ndvi": "decrease",
            "ndre": "decrease",
            "ndwi": "decrease",
            "pri": "decrease",
            "ari": "increase",
            "evi": "decrease",
            "nbr": "decrease",
        }

    available_classes = [
        c for c in expected_order
        if np.any(y == c)
    ]

    if len(available_classes) < 2:
        print("\nImpossible de tester la cohérence : moins de deux classes disponibles.")
        return

    missing_classes = [
        c for c in expected_order
        if not np.any(y == c)
    ]

    print("\n=== Vérification de cohérence des indices ===")

    if missing_classes:
        print("Classes manquantes :", missing_classes)

    for index_name, direction in index_directions.items():

        if index_name not in feature_names:
            print(f"{index_name.upper():5s} | Feature absente de X")
            continue

        feature_idx = feature_names.index(index_name)

        pairs = []

        for class_id in available_classes:
            values = X[y == class_id, feature_idx]
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            mean_value = float(np.nanmean(values))
            class_label = class_names.get(class_id, str(class_id))

            pairs.append((class_label, mean_value))

        if len(pairs) < 2:
            print(f"{index_name.upper():5s} | Données insuffisantes")
            continue

        labels, values = zip(*pairs)

        if direction == "decrease":
            coherent = all(
                values[i] > values[i + 1]
                for i in range(len(values) - 1)
            )
            expected = "↓"

        elif direction == "increase":
            coherent = all(
                values[i] < values[i + 1]
                for i in range(len(values) - 1)
            )
            expected = "↑"

        else:
            raise ValueError("direction doit être 'increase' ou 'decrease'.")

        values_text = " | ".join(
            f"{label}={value:.4f}"
            for label, value in zip(labels, values)
        )

        print(
            f"{index_name.upper():5s} | "
            f"{values_text} | "
            f"Attendu={expected} | "
            f"Cohérent={coherent}"
        )
        
# ============================================================
# 9) KRUSKAL-WALLIS
# ============================================================

def kruskal_tests(indices_dict, group_mask, STRESS_CLASS_NAMES, valid_pixels_mask=None, ignore_labels=(0,), min_pixels=30):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(group_mask.shape, dtype=bool)

    results = {}

    for index_name, index_map in indices_dict.items():
        groups = []

        for stress_id in sorted(STRESS_CLASS_NAMES.keys()):
            if stress_id in ignore_labels:
                continue

            mask = (
                (group_mask == stress_id) &
                valid_pixels_mask &
                np.isfinite(index_map)
            )

            values = index_map[mask]
            if values.size >= min_pixels:
                groups.append(values)

        if len(groups) < 2:
            results[index_name] = None
            continue

        stat, pvalue = kruskal(*groups)
        results[index_name] = {
            "statistic": float(stat),
            "pvalue": float(pvalue),
        }

    return results


import numpy as np
import matplotlib.pyplot as plt


def plot_boxplots_from_xy(
    X,
    y,
    feature_names,
    class_names=None,
    ignore_labels=(0,),
    figsize=(8, 5),
):
    """
    Affiche un boxplot par feature regroupé par classe.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Features.
    y : ndarray (n_samples,)
        Labels.
    feature_names : list[str]
        Noms des features.
    class_names : dict, optional
        Dictionnaire {label: nom_classe}.
    ignore_labels : tuple
        Labels à ignorer.
    figsize : tuple
        Taille de la figure.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    classes = sorted(
        c for c in np.unique(y)
        if c not in ignore_labels
    )

    for feature_idx, feature_name in enumerate(feature_names):

        data = []
        labels = []

        for class_id in classes:

            values = X[y == class_id, feature_idx]
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            data.append(values)

            if class_names is not None:
                labels.append(class_names.get(class_id, str(class_id)))
            else:
                labels.append(str(class_id))

        if len(data) < 2:
            print(f"{feature_name}: pas assez de classes pour tracer.")
            continue

        plt.figure(figsize=figsize)

        plt.boxplot(
            data,
            tick_labels=labels,
            showfliers=False
        )

        plt.title(feature_name.upper())
        plt.ylabel(feature_name)
        plt.xlabel("Classe")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()