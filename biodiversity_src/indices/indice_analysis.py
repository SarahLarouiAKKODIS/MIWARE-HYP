import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils import plot_boxplots_from_xy
from utils import load_config, check_stress_coherence_from_xy


# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":

    config_site_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/salsigne.json"
    config_site = load_config(config_site_path)
    config_forest = load_config(config_site["config_forest_path"])
    stress_class_names = config_forest["STRESS_CLASS_NAMES"]
    species_class_names = {str(v): k for k, v in config_forest["CLASS_MAP"].items()}

    feature_names = [
        "NDVI",
        "NDRE",
        "NDWI",
        "PRI",
        "ARI",
        "EVI",
        "NBR",
    ]

    stress = True

    if stress == True :
        numpy_file = "dataset_indices_for_stress_classif.npz"
        class_names = stress_class_names

    else:
        numpy_file = "dataset_indices_for_species_classif.npz"
        class_names = species_class_names

    print('class_names', class_names)

    data = np.load(Path(config_site["Path_res"]) / "Indice_values/" / numpy_file)

    X = data["X"]
    y = data["y"]

    if stress == True :
        check_stress_coherence_from_xy(
            X=X,
            y=y,
            feature_names=feature_names,
            class_names=config_forest["STRESS_CLASS_NAMES"],
            expected_order=[1, 2, 3, 4],
        )

    plot_boxplots_from_xy(
        X=X,
        y=y,
        feature_names=feature_names,
        class_names=class_names
    )
