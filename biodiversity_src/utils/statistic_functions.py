
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




import numpy as np
import pandas as pd


from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path

import numpy as np
import pandas as pd


def compute_feature_statistics_by_class(
    X,
    y,
    feature_names=None,
    output_excel="feature_statistics_by_class.xlsx",
    decimals=6,
):
    """
    Calcule les statistiques descriptives des variables pour chaque classe
    et exporte les résultats dans un fichier Excel.

    Statistiques calculées
    ----------------------
    - nombre d'échantillons ;
    - moyenne ;
    - médiane ;
    - écart-type ;
    - coefficient de variation en pourcentage ;
    - minimum ;
    - premier quartile Q1 ;
    - troisième quartile Q3 ;
    - intervalle interquartile IQR ;
    - maximum.

    Paramètres
    ----------
    X : pandas.DataFrame ou array-like
        Tableau des variables explicatives.

    y : pandas.Series ou array-like
        Classes associées aux observations de X.

    feature_names : list[str], optional
        Noms des variables lorsque X est un tableau NumPy.
        Ce paramètre n'est pas nécessaire lorsque X est un DataFrame.

    output_excel : str, pathlib.Path ou None
        Chemin du fichier Excel à créer.
        Si la valeur est None, aucun fichier Excel n'est créé.

    decimals : int
        Nombre de décimales conservées dans le tableau final.

    Retours
    -------
    stats_df : pandas.DataFrame
        Tableau complet des statistiques par classe.
    """

    # ------------------------------------------------------------------
    # 1. Vérification et conversion de X
    # ------------------------------------------------------------------
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()

        if feature_names is not None:
            if len(feature_names) != X_df.shape[1]:
                raise ValueError(
                    "Le nombre de noms dans feature_names doit correspondre "
                    "au nombre de colonnes de X."
                )

            X_df.columns = feature_names

    else:
        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError(
                "X doit être un tableau à deux dimensions."
            )

        if feature_names is None:
            feature_names = [
                f"feature_{i}"
                for i in range(X_array.shape[1])
            ]

        if len(feature_names) != X_array.shape[1]:
            raise ValueError(
                "Le nombre de noms dans feature_names doit correspondre "
                "au nombre de colonnes de X."
            )

        X_df = pd.DataFrame(
            X_array,
            columns=feature_names,
        )

    # Vérification de l'unicité des noms de variables
    if X_df.columns.duplicated().any():
        duplicated = X_df.columns[
            X_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Les noms de variables doivent être uniques. "
            f"Doublons détectés : {duplicated}"
        )

    # ------------------------------------------------------------------
    # 2. Vérification et conversion de y
    # ------------------------------------------------------------------
    y_array = np.asarray(y).ravel()

    if len(X_df) != len(y_array):
        raise ValueError(
            "X et y doivent contenir le même nombre d'observations. "
            f"X contient {len(X_df)} lignes et y contient {len(y_array)} valeurs."
        )

    y_series = pd.Series(
        y_array,
        name="class",
    )

    # Réinitialisation des index pour garantir l'alignement
    X_df = X_df.reset_index(drop=True)
    y_series = y_series.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Conversion des variables en valeurs numériques
    # ------------------------------------------------------------------
    for feature in X_df.columns:
        X_df[feature] = pd.to_numeric(
            X_df[feature],
            errors="coerce",
        )

    # Création du tableau contenant les variables et la classe
    data = X_df.copy()
    data["class"] = y_series

    # Suppression des lignes dont la classe est manquante
    data = data.dropna(subset=["class"])

    if data.empty:
        raise ValueError(
            "Aucune observation valide n'est disponible après "
            "la suppression des classes manquantes."
        )

    # ------------------------------------------------------------------
    # 4. Calcul des statistiques par classe
    # ------------------------------------------------------------------
    results = []

    for class_value, class_data in data.groupby(
        "class",
        sort=True,
    ):
        class_statistics = {
            "class": class_value,
            "n_samples": len(class_data),
        }

        for feature in X_df.columns:
            values = class_data[feature].dropna()

            # Cas où une variable ne contient aucune valeur valide
            if values.empty:
                class_statistics.update({
                    f"{feature}_mean": np.nan,
                    f"{feature}_median": np.nan,
                    f"{feature}_std": np.nan,
                    f"{feature}_cv": np.nan,
                    f"{feature}_min": np.nan,
                    f"{feature}_q1": np.nan,
                    f"{feature}_q3": np.nan,
                    f"{feature}_iqr": np.nan,
                    f"{feature}_max": np.nan,
                })

                continue

            mean_value = values.mean()
            median_value = values.median()

            # ddof=1 correspond à l'écart-type d'un échantillon
            std_value = values.std(ddof=1)

            q1_value = values.quantile(0.25)
            q3_value = values.quantile(0.75)
            iqr_value = q3_value - q1_value

            # Le CV n'est pas fiable lorsque la moyenne est proche de zéro.
            if (
                pd.isna(std_value)
                or np.isclose(mean_value, 0.0, atol=1e-12)
            ):
                cv_value = np.nan
            else:
                cv_value = (
                    std_value / abs(mean_value)
                ) * 100

            class_statistics.update({
                f"{feature}_mean": mean_value,
                f"{feature}_median": median_value,
                f"{feature}_std": std_value,
                f"{feature}_cv": cv_value,
                f"{feature}_min": values.min(),
                f"{feature}_q1": q1_value,
                f"{feature}_q3": q3_value,
                f"{feature}_iqr": iqr_value,
                f"{feature}_max": values.max(),
            })

        results.append(class_statistics)

    stats_df = pd.DataFrame(results)

    # Arrondissement des colonnes numériques, sauf n_samples
    columns_to_round = [
        column
        for column in stats_df.columns
        if column not in ["class", "n_samples"]
    ]

    stats_df[columns_to_round] = stats_df[
        columns_to_round
    ].round(decimals)

    stats_df["n_samples"] = stats_df[
        "n_samples"
    ].astype(int)

    # ------------------------------------------------------------------
    # 5. Export dans Excel
    # ------------------------------------------------------------------
    if output_excel is not None:
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Alignment, Font
            from openpyxl.utils import get_column_letter

        except ImportError as error:
            raise ImportError(
                "L'export Excel nécessite le module openpyxl. "
                "Installe-le avec la commande : pip install openpyxl"
            ) from error

        output_path = Path(output_excel)

        # Ajout automatique de l'extension xlsx
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")

        # Création du dossier de sortie s'il n'existe pas
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Sélection des colonnes pour les feuilles synthétiques
        identification_columns = [
            "class",
            "n_samples",
        ]

        mean_columns = identification_columns + [
            column
            for column in stats_df.columns
            if column.endswith("_mean")
        ]

        median_columns = identification_columns + [
            column
            for column in stats_df.columns
            if column.endswith("_median")
        ]

        std_columns = identification_columns + [
            column
            for column in stats_df.columns
            if column.endswith("_std")
        ]

        cv_columns = identification_columns + [
            column
            for column in stats_df.columns
            if column.endswith("_cv")
        ]

        iqr_columns = identification_columns + [
            column
            for column in stats_df.columns
            if column.endswith("_iqr")
        ]

        # Écriture initiale du fichier avec pandas
        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
        ) as writer:
            stats_df.to_excel(
                writer,
                sheet_name="Statistiques_completes",
                index=False,
            )

            stats_df[mean_columns].to_excel(
                writer,
                sheet_name="Moyennes",
                index=False,
            )

            stats_df[median_columns].to_excel(
                writer,
                sheet_name="Medianes",
                index=False,
            )

            stats_df[std_columns].to_excel(
                writer,
                sheet_name="Ecarts_types",
                index=False,
            )

            stats_df[cv_columns].to_excel(
                writer,
                sheet_name="CV",
                index=False,
            )

            stats_df[iqr_columns].to_excel(
                writer,
                sheet_name="IQR",
                index=False,
            )

        # ------------------------------------------------------------------
        # 6. Mise en forme avec openpyxl
        # ------------------------------------------------------------------
        workbook = load_workbook(output_path)

        for worksheet in workbook.worksheets:
            # Figer la première ligne et les deux premières colonnes
            worksheet.freeze_panes = "C2"

            # Activer le filtre automatique
            worksheet.auto_filter.ref = worksheet.dimensions

            # Mise en forme des en-têtes
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

            # Alignement et format des cellules
            for row in worksheet.iter_rows(
                min_row=2,
                max_row=worksheet.max_row,
            ):
                for cell in row:
                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center",
                    )

                    # Les deux premières colonnes sont des identifiants
                    if cell.column > 2 and isinstance(
                        cell.value,
                        (int, float),
                    ):
                        cell.number_format = (
                            "0." + ("0" * decimals)
                            if decimals > 0
                            else "0"
                        )

            # Ajustement automatique de la largeur des colonnes
            for column_index in range(
                1,
                worksheet.max_column + 1,
            ):
                column_letter = get_column_letter(
                    column_index
                )

                maximum_length = 0

                for cell in worksheet[column_letter]:
                    if cell.value is not None:
                        cell_length = len(str(cell.value))
                        maximum_length = max(
                            maximum_length,
                            cell_length,
                        )

                # Limitation de la largeur pour éviter des colonnes énormes
                adjusted_width = min(
                    maximum_length + 2,
                    22,
                )

                worksheet.column_dimensions[
                    column_letter
                ].width = adjusted_width

            # Hauteur de la ligne d'en-tête
            worksheet.row_dimensions[1].height = 35

        workbook.save(output_path)

        print(
            "Fichier Excel créé avec succès : "
            f"{output_path.resolve()}"
        )

    return stats_df