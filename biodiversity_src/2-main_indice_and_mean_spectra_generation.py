#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from clean_src.preprocessing.spectral_smoothing import savgol_smooth_and_normalize
from utils import load_config, read_mask
from utils import read_hyperspectral_raster, read_mask, read_wavelengths_from_csv, build_enmap_valid_mask
from utils import compute_spectral_indices

from feature_extraction.indice_dataset_generation import build_and_save_datasets
from feature_extraction.mean_class_spectra import compute_mean_spectrum_per_class
from feature_extraction.absorption_depths import compute_band_depth_cube

from utils import save_valid_mask

# ============================================================
# 12) MAIN
# ============================================================

if __name__ == "__main__":

    #config_site_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/abbaretz_3.json"
    # ============================================================
    # DOSSIER CONTENANT LES CONFIGS
    # ============================================================

    configs_dir = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess")

    config_paths = sorted(configs_dir.glob("*.json"))

    if not config_paths:
        raise FileNotFoundError(
            f"Aucun fichier JSON trouvé dans : {configs_dir}"
        )

    print(f"\n{len(config_paths)} fichier(s) de configuration trouvé(s).")


    # ============================================================
    # TRAITEMENT DE CHAQUE CONFIG
    # ============================================================

    for i, config_path in enumerate(config_paths, start=1):

        print("\n" + "=" * 70)
        print(
            f"[{i}/{len(config_paths)}] "
            f"Traitement : {config_path.name}"
        )
        print("=" * 70)

        config_site = load_config(config_path)

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

        Cloud_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["Cloud_mask"]))
        Haze_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["Haze_mask"]))
        Cirrus_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["Cirrus_mask"]))
        CloudShadow_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["CloudShadow_mask"]))
        Snow_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["Snow_mask"]))
        TestFlags_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["TestFlags_mask"]))
        # masque scène : 0 unclassified, 1 land, 2 water, 3 no data
        Scene_mask = read_mask(Path(config_site["Path_data"]) / Path(config_site["QualityClasses_mask"]))
        

        ## CLASSES DES ESSENCES ET DES ETAT DE STRESSE   
        config_forest = load_config(config_site["config_forest_path"])

        CLASS_MAP = config_forest["CLASS_MAP"]   
        ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}     # id -> nom

        # --- 2) Groupes de stress ---
        def keys_to_int(d):
            return {int(k): v for k, v in d.items()}

        # STRESS_GROUPS = keys_to_int(config_forest["STRESS_GROUPS"])
        # STRESS_CLASS_NAMES = keys_to_int(config_forest["STRESS_CLASS_NAMES"])

        # # --- 3) Conversion des espèces en labels numériques ---
        # STRESS_GROUPS_LABELS = {
        #     group: [CLASS_MAP[species] for species in species_list]
        #     for group, species_list in STRESS_GROUPS.items()
        # }

        EXCLUDED_ORIGINAL_LABELS = (0, CLASS_MAP["NC"], CLASS_MAP["NR"])

        gt_mask_path = Path_res / "Forest/mask_foret_classes.tif"
        output_group_mask_path = Path_res / "Forest/mask_gt_stress_4classes.tif"
        output_figure_path = Path_res / "figure_gt_vs_stress.png"

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

        # Filtrage des réflectances non physiques
        cube_clean = cube.copy()
        cube_clean[cube_clean < 0] = np.nan
        cube_clean[cube_clean > 1.2] = np.nan
        print("MIN :", np.nanmin(cube_clean))
        print("MEAN:", np.nanmean(cube_clean))
        print("MAX :", np.nanmax(cube_clean))

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

        save_valid_mask(
            valid_mask=enmap_valid_mask,
            output_path=Path_res / "enmap_valid_mask.tif",
            profile=profile,
        )

        print("Pixels valides :", np.count_nonzero(enmap_valid_mask))
        print("Pixels totaux  :", enmap_valid_mask.size)

        ######### INDICES CALCULATION #################################################

        indices = compute_spectral_indices(cube, wavelengths)

        ######### DATASETS FEATURES CALCULATION #######################################

        out_dataset_dir = Path_res / "Feature_extraction"
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        ######### SPECIES indices calculation [X, y]

        build_and_save_datasets(
            gt_mask=gt_mask,
            config_forest=config_forest,
            indices=indices,
            enmap_valid_mask=enmap_valid_mask,
            profile=profile,
            output_group_mask_path=output_group_mask_path,
            output_figure_path = output_figure_path,
            out_dataset_dir=out_dataset_dir,
            excluded_original_labels=EXCLUDED_ORIGINAL_LABELS,
        )

        ######### mean spectre per class calculation [X, y]
        
        mean_spectra = compute_mean_spectrum_per_class(
            image_tif=image_hyperspectrale_cleanbands_smooth,
            gt_mask_tif=gt_mask,
            wavelengths=wavelengths,
            class_names=ID_TO_CLASS,
            ignore_labels=EXCLUDED_ORIGINAL_LABELS,
            min_pixels=30,
            valid_pixels_mask=enmap_valid_mask,
            output_csv=out_dataset_dir / "mean_spectra_per_class.csv",
        )

        ######### Band dept calculation

        absorption_wavelengths = [970, 1200, 1450, 1940, 2200]

        custom_shoulders = {
            970.0:  (900, 1070),
            1200.0: (1120, 1280),
            1450.0: (1380, 1520),
            1940.0: (1860, 2020),
            2200.0: (2130, 2280),
        }

        band_depth_cube, band_depth_names, metadata = compute_band_depth_cube(
            image_tif=image_hyperspectrale_cleanbands_smooth,
            wavelengths_csv=clean_wavelengths_csv,
            absorption_wavelengths=absorption_wavelengths,
            custom_shoulders=custom_shoulders,
            window_nm=20,
        )

        np.savez(
            out_dataset_dir / "band_depth_cube.npz",
            X=band_depth_cube    )