
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
            "nbr": float(np.nanmean(indices_dict["nbr"][mask])),
            "ndvi_std": float(np.nanstd(indices_dict["ndvi"][mask])),
            "ndre_std": float(np.nanstd(indices_dict["ndre"][mask])),
            "ndwi_std": float(np.nanstd(indices_dict["ndwi"][mask])),
            "pri_std": float(np.nanstd(indices_dict["pri"][mask])),
            "ari_std": float(np.nanstd(indices_dict["ari"][mask])),
            "nbr_std": float(np.nanstd(indices_dict["nbr"][mask]))
        }

    return stats


# ============================================================
# 8) COHÉRENCE ATTENDUE DES INDICES
# ============================================================

def check_stress_coherence(stats):
    """
    Vérifie la cohérence attendue :
    - NDVI : pas stressée > peu stressée > moyenne > très stressée
    - NDRE : pas stressée > peu stressée > moyenne > très stressée
    - GNDVI : pas stressée > peu stressée > moyenne > très stressée
    - NDWI (ta formule NIR-SWIR / NIR+SWIR) :
      on attend généralement pas stressée > peu stressée > moyenne > très stressée
      (plus proche de 0 si végétation moins stressée)
    """
    required_ids = [1, 2, 3, 4]
    if not all(i in stats for i in required_ids):
        print("\nImpossible de tester la cohérence : une ou plusieurs classes manquent.")
        return

    print("\n=== Vérification de cohérence des indices ===")

    for index_name in ["ndvi", "ndre", "ndwi", "pri", "ari", "nbr"]:
        v1 = stats[1][index_name]
        v2 = stats[2][index_name]
        v3 = stats[3][index_name]
        v4 = stats[4][index_name]

        coherent = (v1 > v2 > v3 > v4)

        print(
            f"{index_name.upper():5s} | "
            f"Pas stressée={v1:.4f} | "
            f"Peu stressée={v2:.4f} | "
            f"Moyennement stressée={v3:.4f} | "
            f"Très stressée={v4:.4f} | "
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


