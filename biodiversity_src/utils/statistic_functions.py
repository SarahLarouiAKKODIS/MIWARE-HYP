
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


