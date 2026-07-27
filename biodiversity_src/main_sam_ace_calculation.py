#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils import load_config, read_mask, read_hyperspectral_raster
from feature_extraction.sam_ace_feature_extraction import compute_final_sam_ace_from_reference_csvs, compute_best_sam_ace_species_from_reference_csvs

# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":

    config_path = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/")

    train_config_files = [
        config_path / "salsigne_3.json", 
        config_path / "salsigne_mars_2.json",
        config_path / "salsigne_mars_3.json", 
        config_path / "salsigne.json", 
        config_path / "abbaretz.json", 
        config_path / "abbaretz_3.json", 
    ]

    test_config_files = [
        config_path / "salau_1.json",
        config_path / "salau_2.json", 
        config_path / "cma_1.json", 
        config_path / "cma.json", 
    ]

    all_config_files = train_config_files + test_config_files

    ## Récupération des spectres de référence par classe

    reference_spectra_paths = []
    for config_site_path in train_config_files:
        config_site = load_config(config_site_path)

        mean_spectra_file = Path(config_site["Path_res"]) / "Feature_extraction/mean_spectra_per_class.csv"
        reference_spectra_paths.append(mean_spectra_file)

    ## Calcul des cartes de similarité

    for config_site_path in all_config_files:
        config_site = load_config(config_site_path)
        Path_res = Path(config_site["Path_res"])

        ## CLASSES DES ESSENCES ET DES ETAT DE STRESSE   
        config_forest = load_config(config_site["config_forest_path"])

        CLASS_MAP = config_forest["CLASS_MAP"]   
        ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}     # id -> nom

        image_hyperspectrale_cleanbands_smooth = Path_res / "image_hyperspectrale_cleanbands_smooth.tif"
        cube, profile = read_hyperspectral_raster(image_hyperspectrale_cleanbands_smooth)
        # Filtrage des réflectances non physiques
        cube_clean = cube.copy()
        cube_clean[cube_clean < 0] = np.nan
        cube_clean[cube_clean > 1.2] = np.nan

        enmap_valid_mask = read_mask(Path_res / "enmap_valid_mask.tif")

        # outputs = compute_final_sam_ace_from_reference_csvs(
        #     img=cube_clean,
        #     profile=profile,
        #     reference_csv_paths=reference_spectra_paths,
        #     out_dir=Path_res / "Similarity_maps",
        #     valid_pixels_mask=enmap_valid_mask,
        # )

        CLASS_TO_ID = {
            v: k
            for k, v in ID_TO_CLASS.items()
        }

        outputs = compute_best_sam_ace_species_from_reference_csvs(
            img=cube_clean,
            profile=profile,
            reference_csv_paths=reference_spectra_paths,
            out_dir=Path_res / "Similarity_best_maps",
            class_id_map=CLASS_TO_ID,
            valid_pixels_mask=enmap_valid_mask,
        )
   

        