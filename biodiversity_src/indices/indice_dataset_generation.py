from pathlib import Path
import numpy as np
from utils import compute_mean_indices_per_group, check_stress_coherence, kruskal_tests
from utils import reclassify_gt_to_stress_classes
from utils import plot_boxplots, save_mask, extract_pixels, plot_gt_and_stress_masks


def build_and_save_datasets(
    gt_mask,
    config_forest: dict,
    indices: dict,
    enmap_valid_mask,
    profile: dict,
    output_group_mask_path: str | Path,
    output_figure_path: str | Path,
    path_res: str | Path,
    excluded_original_labels=(),
    ignore_labels=(0,),
    min_pixels_kw: int = 30,
    min_pixels_boxplot: int = 10,
):
    """
    Reclassifie un masque GT en classes de stress, calcule les statistiques
    par groupe, prépare le dataset X/y, lance les tests statistiques et affiche
    les boxplots.

    Returns
    -------
    dict
        Résultats principaux : stress_mask, stats, X, y, kw_results.
    """

    path_res = Path(path_res)
    output_group_mask_path = Path(output_group_mask_path)

    #stress_class_names = config_forest["STRESS_CLASS_NAMES"]
    stress_class_names = {int(k): v for k, v in config_forest["STRESS_CLASS_NAMES"].items()}

    # 1) Reclassification GT -> classes de stress
    stress_mask = reclassify_gt_to_stress_classes(
        gt_mask=gt_mask,
        stress_groups=config_forest["STRESS_GROUPS"],
        class_map=config_forest["CLASS_MAP"],
        excluded_labels=excluded_original_labels,
    )

     # ============================================================
    # PLOT
    # ============================================================

    plot_gt_and_stress_masks(
        gt_mask=gt_mask,
        stress_mask=stress_mask,
        class_map= config_forest["CLASS_MAP"],
        stress_class_names= stress_class_names,
        output_path=output_figure_path,
    )

    save_mask(output_group_mask_path, stress_mask, profile)
    print("Masque de classes de stress sauvegardé :", output_group_mask_path)

   
    # 6) Extraction dataset X/y for stress classification

    X_st, y_st, rows, cols = extract_pixels(
        indices_dict=indices,
        target_mask=stress_mask,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=ignore_labels,
    )

    print("\nForme de X :", X_st.shape)
    print("Forme de y :", y_st.shape)

    out_dataset_dir = path_res / "Indice_values"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dataset_dir / "dataset_indices_for_stress_classif.npz"

    np.savez(
        dataset_path,
        X=X_st,
        y=y_st,
    )

    print("Dataset indices pour la classification de l'état de stress, sauvegardé :", dataset_path)

     # 6) Extraction dataset X/y for stress classification

    X_sp, y_sp, rows, cols = extract_pixels(
        indices_dict=indices,
        target_mask=gt_mask,
        valid_pixels_mask=enmap_valid_mask,
        ignore_labels=ignore_labels,
    )

    print("\nForme de X :", X_sp.shape)
    print("Forme de y :", y_sp.shape)

    out_dataset_dir = path_res / "Indice_values"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dataset_dir / "dataset_indices_for_species_classif.npz"

    np.savez(
        dataset_path,
        X=X_sp,
        y=y_sp,
    )

    print("Dataset indices pour la classification des espèces végétales sauvegardé :", dataset_path)

    return X_st, y_sp, y_st

