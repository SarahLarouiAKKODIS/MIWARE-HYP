#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from clean_src.preprocessing.spectral_smoothing import savgol_smooth_and_normalize
from utils.commun_functions import load_config, read_mask
from utils.statistic_functions import compute_mean_indices_per_group, check_stress_coherence, kruskal_tests
from utils.classification_functions import reclassify_gt_to_stress_classes, run_random_forest_classification, spatial_block_cv_random_forest, train_test_spatial_split
from utils.display import plot_boxplots, plot_spatial_cv_folds, print_class_distribution_named, print_spatial_cv_results
from utils.hyperspectral_utils import read_hyperspectral_raster, read_mask, read_wavelengths_from_csv, save_mask, build_enmap_valid_mask
from utils.indices_calculation import compute_spectral_indices

# ============================================================
# 6) EXTRACTION DES PIXELS
# ============================================================


def extract_pixels(indices_dict, target_mask, valid_pixels_mask=None, ignore_labels=(0,)):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(target_mask.shape, dtype=bool)

    valid = (
        valid_pixels_mask &
        np.isfinite(indices_dict["ndvi"]) &
        np.isfinite(indices_dict["ndre"]) &
        np.isfinite(indices_dict["ndwi"]) &
        np.isfinite(indices_dict["pri"]) &
        np.isfinite(indices_dict["ari"]) &
        np.isfinite(indices_dict["nbr"]) 
    )

    for lab in ignore_labels:
        valid &= (target_mask != lab)

    # coordonnées pixels
    rows, cols = np.where(valid)

    # features
    X = np.stack([
        indices_dict["ndvi"][valid],
        indices_dict["ndre"][valid],
        indices_dict["ndwi"][valid],
        indices_dict["pri"][valid],
        indices_dict["ari"][valid],
        indices_dict["nbr"][valid],
    ], axis=1).astype(np.float32)

    # labels
    y = target_mask[valid].astype(np.int32)

    return X, y, rows, cols



# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":

    config_site_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/cma_1.json"
    config_site = load_config(config_site_path)

    Path_res = Path(config_site["Path_res"])
    image_hyperspectrale_cleanbands = Path_res / "image_hyperspectrale_cleanbands.tif"
    clean_wavelengths_csv = Path_res / "enmap_clean_bands_full.csv" 

    ## Lissage (optionel mais mieux)
    image_hyperspectrale_cleanbands_smooth = Path_res / "image_hyperspectrale_cleanbands_smooth.tif"

    if not image_hyperspectrale_cleanbands_smooth.exists():
        savgol_smooth_and_normalize(
            img_path=image_hyperspectrale_cleanbands,
            output_path=image_hyperspectrale_cleanbands_smooth,
            normalize=None
        )

    Cloud_mask = read_mask(Path(config_site["Cloud_mask"]))
    Haze_mask = read_mask(Path(config_site["Haze_mask"]))
    Cirrus_mask = read_mask(Path(config_site["Cirrus_mask"]))
    CloudShadow_mask = read_mask(Path(config_site["CloudShadow_mask"]))
    Snow_mask = read_mask(Path(config_site["Snow_mask"]))
    TestFlags_mask = read_mask(Path(config_site["TestFlags_mask"]))
    # masque scène : 0 unclassified, 1 land, 2 water, 3 no data
    Scene_mask = read_mask(Path(config_site["QualityClasses_mask"]))
    

    ## CLASSES DES ESSENCES ET DES ETAT DE STRESSE   
    config_forest = load_config(config_site["config_forest_path"])

    CLASS_MAP = config_forest["CLASS_MAP"]   
    ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}     # id -> nom

    # --- 2) Groupes de stress ---
    def keys_to_int(d):
        return {int(k): v for k, v in d.items()}

    STRESS_GROUPS = keys_to_int(config_forest["STRESS_GROUPS"])
    STRESS_CLASS_NAMES = keys_to_int(config_forest["STRESS_CLASS_NAMES"])

    # --- 3) Conversion des espèces en labels numériques ---
    STRESS_GROUPS_LABELS = {
        group: [CLASS_MAP[species] for species in species_list]
        for group, species_list in STRESS_GROUPS.items()
    }

    EXCLUDED_ORIGINAL_LABELS = (0, CLASS_MAP["NC"], CLASS_MAP["NR"])

    gt_mask_path = Path_res / "Forest/mask_foret_classes.tif"
    output_group_mask_path = Path_res / "Forest/mask_gt_stress_4classes.tif"

    # Lecture
    cube, profile = read_hyperspectral_raster(image_hyperspectrale_cleanbands_smooth)
    wavelengths = read_wavelengths_from_csv(clean_wavelengths_csv)
    gt_mask = read_mask(gt_mask_path)

    print("cube shape:", cube.shape)
    print("min cube:", np.nanmin(cube))
    print("max cube:", np.nanmax(cube))
    print("Nombre wavelengths:", len(wavelengths))

    if len(wavelengths) != cube.shape[0]:
        raise ValueError(
            f"Incohérence bandes : {len(wavelengths)} wavelengths vs {cube.shape[0]} bandes"
        )

    # Masque de validité EnMAP
    enmap_valid_mask = build_enmap_valid_mask(
        cirrus_mask=Cirrus_mask,
        cloud_mask=Cloud_mask,
        haze_mask=Haze_mask,
        cloudshadow_mask=CloudShadow_mask,
        snow_mask=Snow_mask,
        testflags_mask=TestFlags_mask,
        scene_mask=Scene_mask,
    )

    print("Pixels valides :", np.count_nonzero(enmap_valid_mask))
    print("Pixels totaux  :", enmap_valid_mask.size)

    # Indices
    indices = compute_spectral_indices(cube, wavelengths)

    # Reclassification GT -> 4 classes
    stress_mask = reclassify_gt_to_stress_classes(
        gt_mask=gt_mask,
        stress_groups=config_forest["STRESS_GROUPS"],
        class_map=config_forest["CLASS_MAP"],
        excluded_labels=EXCLUDED_ORIGINAL_LABELS
    )

    save_mask(output_group_mask_path, stress_mask, profile)
    print("Masque 4 classes sauvegardé :", output_group_mask_path)

    # Stats par groupe
    stats = compute_mean_indices_per_group(
        indices,
        stress_mask,
        STRESS_CLASS_NAMES,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=(0,)
    )

    print("\n=== Moyennes des indices par classe de stress ===")
    for stress_id in sorted(stats.keys()):
        vals = stats[stress_id]
        print(
            f"{stress_id} - {vals['class_name']:25s} | "
            f"n={vals['n_pixels']:7d} | "
            f"NDVI={vals['ndvi']:.4f} ± {vals['ndvi_std']:.4f} | "
            f"NDRE={vals['ndre']:.4f} ± {vals['ndre_std']:.4f} | "
            f"NDWI={vals['ndwi']:.4f} ± {vals['ndwi_std']:.4f}"
            f"PRI={vals['pri']:.4f} ± {vals['pri_std']:.4f} | "
            f"ARI={vals['ari']:.4f} ± {vals['ari_std']:.4f} | "
            f"NBR={vals['nbr']:.4f} ± {vals['nbr_std']:.4f} | "
        )

    # Vérification de cohérence
    check_stress_coherence(stats)

#     # Kruskal-Wallis
#     kw_results = kruskal_tests(
#         indices,
#         stress_mask,
#         STRESS_CLASS_NAMES,
#         valid_pixels_mask=enmap_valid_mask,
#         ignore_labels=(0,),
#         min_pixels=30
#     )

#     print("\n=== Tests de Kruskal-Wallis ===")
#     for idx_name, res in kw_results.items():
#         if res is None:
#             print(f"{idx_name.upper()} : test impossible")
#             continue
#         print(f"{idx_name.upper()} | H={res['statistic']:.4f} | p={res['pvalue']:.4e}")

#     # Boxplots
#     plot_boxplots(
#         indices,
#         stress_mask,
#         STRESS_CLASS_NAMES,
#         valid_pixels_mask=enmap_valid_mask,
#         ignore_labels=(0,),
#         min_pixels=10
#     )

#     # Random Forest
#     X, y, rows, cols = extract_pixels(
#     indices_dict=indices,
#     target_mask=stress_mask,
#     valid_pixels_mask=enmap_valid_mask,
#     ignore_labels=(0,)
#     )

#     print("\nForme de X :", X.shape)
#     print("Forme de y :", y.shape)

#     Path(Path_res / "Indice_values").mkdir(exist_ok=True)
#     #Save dataset
#     np.savez(
#     Path_res / "Indice_values/dataset_indices.npz",
#     X=X,
#     y=y
# )
    

    ########### Séparation aléatoire stratifée 

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.3,
    #     random_state=42,
    #     stratify=y
    # )

    ########### Spatial split

    # X_train, X_test, y_train, y_test =  train_test_spatial_split(X, y, rows, cols, gt_mask.shape)
    
    # print_class_distribution_named(y_train, STRESS_CLASS_NAMES, "train")
    # print_class_distribution_named(y_test, STRESS_CLASS_NAMES, "test")

    ##################################################################################################

    # rf_results = run_random_forest_classification(X_train, X_test, y_train, y_test, STRESS_CLASS_NAMES)

    # print("\n=== Random Forest sur 3 classes de stress ===")
    # print(f"Accuracy : {rf_results['accuracy']:.4f}")
    # print(f"Kappa    : {rf_results['kappa']:.4f}")

    # print("\nMatrice de confusion :")
    # print(rf_results["confusion_matrix"])

    # print("\nClassification report :")
    # print(rf_results["classification_report"])

    # print("\nImportance des indices :")
    # feature_names = ["NDVI", "NDRE", "GNDVI", "NDWI"]
    # for name, imp in zip(feature_names, rf_results["feature_importances"]):
    #     print(f"{name:6s} : {imp:.4f}")

    ############ Spatial cros-validation par blocs

#     n_splits = 5
#     block_size = 200
#     min_pixels_per_block = 100

#     plot_spatial_cv_folds(
#     rows=rows,
#     cols=cols,
#     image_shape=gt_mask.shape,
#     block_size=block_size,
#     n_splits=n_splits
# )

#     cv_results = spatial_block_cv_random_forest(
#     X=X,
#     y=y,
#     rows=rows,
#     cols=cols,
#     n_splits=5,
#     block_size=200,
#     min_pixels_per_block=100,
#     random_state=42,
#     n_estimators=300,
#     )

#     print_spatial_cv_results(cv_results)

    