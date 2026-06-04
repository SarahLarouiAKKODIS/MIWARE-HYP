"""
Module contenant la classe MineralProcessor pour traiter les images hyperspectrales.
"""
from pathlib import Path
import os
import json

# Importer les fonctions nécessaires pour le pré-traitement et la détection minérale
from .utils.enmap_band_utils import recover_wavelet_band_info
from .utils.enmap_crop_image import crop_hyperspectral_tif
from .utils.analyse_hyperspectral_image import analyze_rescaled_cube_with_wavelengths
from .utils.enmap_rgb_extraction import hyperspectral_to_rgb
from .enmap_water_indices import compute_mndwi_and_water_mask
from .enmap_vegetation_indices import compute_vegetation_indices_wdi_vii
from .enmap_quality_mask import mask_enmap_hyperspectral_cube
from .preprocessing.mask_enmap import apply_water_veg_mask
from .preprocessing.enmap_clean_bands import clean_bands_enmap_from_csv
from .preprocessing.rescaling_image import rescale_enmap_cube_simple
from .preprocessing.spectral_smoothing import savgol_smooth_and_normalize

# Mineral detection imports
from .mineral_detection.olivine_detection import detect_olivine_bd1050_bd2000_clean
from .mineral_detection.pyroxene_detection import detect_pyroxene_bd1um_bd2um_clean
from .mineral_detection.amphiboles_detection import detect_amphiboles_bd2320_clean
from .mineral_detection.carbonates_detection import detect_carbonates_bd2330_bd2500_clean
from .mineral_detection.micas_detection import detect_micas_bd2200_clean
from .mineral_detection.argiles_detection import detect_argiles_bd2200_clean
from .mineral_detection.oxydesFer_detection import detect_iron_oxides_bd900_redness_clean

# Importer les méthodes de comparaison spectrales.
from .spectral_comparison_methodes.sam_mf_detection_spy import run_single_mineral_detection_from_tab_refs
from .spectral_comparison_methodes.sam_ace_detection import run_single_mineral_ace_from_tab_refs

class MineralProcessor:
    """
    Classe pour traiter les images hyperspectrales et extraire des informations sur les minéraux.
    
    Attributes:
        config (dict): Configuration chargée depuis le fichier JSON.
        Path_res (Path): Chemin du répertoire de résultats.
        wavelengths_csv (Path): Chemin du fichier CSV contenant les informations des bandes.
    """

    def __init__(self):
        """
        Initialise le processeur sans configuration.
        """
        self.config = None
        self.Path_res = None
        # Configuration pour la détection spectrale
        self.spectral_library_dir = None

    @staticmethod
    def load_config(path: str) -> dict:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            path (str): Chemin vers le fichier de configuration.
            
        Returns:
            dict: Configuration chargée.
        """
        print(f"Loading configuration from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_config_from_path(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier JSON et initialise Path_res.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration JSON.
        """
        self.config = self.load_config(config_path)
        self.Path_res = Path(self.config["Path_res"])
        self.Path_res.mkdir(parents=True, exist_ok=True)
        # Charger la bibliothèque spectrale depuis la configuration si disponible
        if "spectral_library_dir" in self.config:
            self.spectral_library_dir = self.config["spectral_library_dir"]
        print(f"Configuration loaded from {config_path}")

    # pré-traitement complet
    def pre_process(self) -> None:
        """
        Exécute l'ensemble du traitement de l'image hyperspectrale.
        """
        # 1) Récupération des informations des bandes wavelet
        print("Step 1: Recovering wavelet band information...")
        self.recover_wavelet_band_info()

        # 2) Mise à l'échelle de l'image
        print("Step 2: Rescaling the hyperspectral image...")
        self.rescale_image()

        # 3) Exclusion de certains pixels (nuages, brume, ...)
        print("Step 3: Applying masks...")
        self.apply_masks()

        # 4) Découpage de l'image
        print("Step 4: Cropping the image...")
        self.crop_image()

        # 5) Génération d'une image RGB
        print("Step 5: Generating RGB image...")
        self.generate_rgb_image()

        # 6) Nettoyage des bandes
        print("Step 6: Cleaning bands...")
        self.clean_bands()

        # 7) Contrôle qualité (QC)
        print("Step 7: Performing quality control...")
        self.quality_control()

        # 8) Calcul du masque d'eau
        print("Step 8: Computing water mask...")
        self.compute_water_mask()

        # 9) Calcul du masque de végétation
        print("Step 9: Computing vegetation mask...")
        self.compute_vegetation_mask()

        # 10) Application des masques d'eau et de végétation pour identifier les zones minérales
        print("Step 10: Applying water and vegetation masks...")
        self.apply_water_veg_mask()

        # 11) Lissage spectral (optionnel mais recommandé)
        print("Step 11: Smoothing image for mineral detection...")
        self.smooth_image()

        # 12) Détection des minéraux
        print("Step 12: Detecting minerals...")
        self.detect_minerals()
    
    def recover_wavelet_band_info(self) -> None:
        """
        Récupère les informations des bandes wavelet depuis le fichier XML.
        """
        xml_path = Path(self.config["xml_path"])
        self.wavelengths_csv = self.Path_res / "enmap_bands_full.csv"

        df = recover_wavelet_band_info(xml_path, out_csv=self.wavelengths_csv)
        print("Table complète enregistrée : enmap_bands_full.csv")
        print(df.head(10).to_string(index=False))

    def rescale_image(self) -> None:
        """
        Met à l'échelle l'image hyperspectrale.
        """
        image_hyperspectrale = Path(self.config["image_hyp"])
        self.image_hyperspectrale_reflectance = self.Path_res / "image_hyperspectrale_reflectance.tif"

        rescale_enmap_cube_simple(
            input_tif=image_hyperspectrale,
            output_tif=self.image_hyperspectrale_reflectance,
            scale_factor=10000.0
        )

    def apply_masks(self) -> None:
        """
        Applique les masques pour exclure certains pixels (nuages, brume, etc.).
        """
        mask_files = [
            Path(self.config["Cloud_mask"]),
            Path(self.config["Haze_mask"]),
            Path(self.config["Cirrus_mask"]),
            Path(self.config["CloudShadow_mask"]),
            Path(self.config["Snow_mask"]),
            Path(self.config["TestFlags_mask"])
        ]

        self.image_hyperspectrale_clean = self.Path_res / "image_hyperspectrale_clean.tif"

        mask_enmap_hyperspectral_cube(
            cube_tif=self.image_hyperspectrale_reflectance,
            mask_files=mask_files,
            out_tif=self.image_hyperspectrale_clean
        )

    def crop_image(self) -> None:
        """
        Découpe l'image selon les coordonnées spécifiées dans la configuration.
        """
        if self.config["crop"]:
            Site_coords = self.config["Site_coords"]
            self.image_hyperspectrale_clean_crop = self.Path_res / "image_hyperspectrale_crop.tif"
            crop_hyperspectral_tif(self.image_hyperspectrale_clean, self.image_hyperspectrale_clean_crop, Site_coords)
            print("Crop enregistré :", self.image_hyperspectrale_clean_crop)
        else:
            self.image_hyperspectrale_clean_crop = self.image_hyperspectrale_clean

    def generate_rgb_image(self) -> None:
        """
        Génère une image RGB à partir de l'image hyperspectrale.
        """
        self.image_hyperspectrale_rgb = self.Path_res / "image_hyperspectrale_rgb.tif"

        hyperspectral_to_rgb(
            cube_tif=self.image_hyperspectrale_clean_crop,
            bands_csv=self.wavelengths_csv,
            out_rgb_tif=self.image_hyperspectrale_rgb,
            target_R=650,
            target_G=560,
            target_B=480,
        )

    def clean_bands(self) -> None:
        """
        Nettoie les bandes de l'image hyperspectrale.
        """
        self.image_hyperspectrale_cleanbands = self.Path_res / "image_hyperspectrale_cleanbands.tif"
        self.clean_wavelengths_csv = self.Path_res / "enmap_clean_bands_full.csv"

        summary = clean_bands_enmap_from_csv(
            img_path=self.image_hyperspectrale_clean_crop,
            bands_csv=self.wavelengths_csv,
            output_path=self.image_hyperspectrale_cleanbands,
            output_bands_csv=self.clean_wavelengths_csv,
            band_id_is_one_based=True,
            csv_band_id_is_one_based_out=True,
            drop_edges=(2, 2),
            exclude_ranges_nm=[(1340, 1460), (1800, 1960)],
            use_fwhm_margin=True,
            fwhm_factor=0.5
        )

        print(summary["nbands_in"], "->", summary["nbands_out"])
        print("Removed wavelengths (nm):", summary["removed_wavelengths_nm"])
        print("Bands out:", summary["nbands_out"])

    def quality_control(self) -> None:
        """
        Effectue un contrôle qualité sur les bandes de l'image.
        """
        qc_res = analyze_rescaled_cube_with_wavelengths(
            cube_tif=self.image_hyperspectrale_cleanbands,
            bands_csv=self.clean_wavelengths_csv,
            out_csv=self.Path_res / "band_qc_report.csv",
            min_valid=-0.1,
            max_valid=1.5,
            nan_threshold_pct=50.0,
            outlier_threshold_pct=5.0,
            band_id_is_one_based=True,
        )

    def compute_water_mask(self) -> None:
        """
        Calcule le masque d'eau en utilisant l'indice MNDWI.
        """
        self.Water_results_dir = self.Path_res / "Water_indice_outputs"

        water_res = compute_mndwi_and_water_mask(
            tif_path=self.image_hyperspectrale_cleanbands,
            wavelengths_csv=self.clean_wavelengths_csv,
            outdir=self.Water_results_dir,
            prefix="enmap_salsigne",
            mndwi_th=0.55,
            verbose=True,
        )

        print(water_res["paths"])

    def compute_vegetation_mask(self) -> None:
        """
        Calcule le masque de végétation en utilisant les indices WDI et VII.
        """
        self.Vegetation_results_dir = self.Path_res / "Vegetation_indice_outputs"

        veg_res = compute_vegetation_indices_wdi_vii(
            tif_path=self.image_hyperspectrale_cleanbands,
            wavelengths_csv=self.clean_wavelengths_csv,
            outdir=self.Vegetation_results_dir,
            prefix="enmap_salsigne",
            ndvi_th=0.3,
            verbose=True
        )

        print(veg_res["paths"].keys())

    def apply_water_veg_mask(self) -> None:
        """
        Applique les masques d'eau et de végétation pour identifier les zones minérales.
        """
        self.enmap_mineral_candidates = self.Path_res / "enmap_mineral_candidates.tif"

        apply_water_veg_mask(
            img_path=self.image_hyperspectrale_cleanbands,
            water_mask_path=self.Water_results_dir / "enmap_salsigne_WATER_MASK.tiff",
            veg_mask_path=self.Vegetation_results_dir / "enmap_salsigne_VEG_MASK.tiff",
            output_path=self.enmap_mineral_candidates
        )

    # Détection minérale en utilisant les méthodes de band depth
    def process_banddepth_mineral_detection(self) -> None:
        """
        Effectue la détection minérale en utilisant les méthodes de band depth.
        """
        self.smooth_image()  # Assurez-vous que l'image est lissée avant la détection
        # Cette méthode peut être appelée après le pré-traitement pour effectuer la détection minérale spécifique.
        self.detect_minerals()

    def smooth_image(self) -> None:
        """
        Lisse et normalise l'image pour la détection minérale.
        """
        self.image_hyperspectrale_cleanbands_smooth = self.Path_res / "image_hyperspectrale_cleanbands_smooth.tif"

        savgol_smooth_and_normalize(
            img_path=self.image_hyperspectrale_cleanbands,
            output_path=self.image_hyperspectrale_cleanbands_smooth,
            normalize=None
        )

    def detect_minerals(self) -> None:
        """
        Détecte tous les types de minéraux dans l'image traitée.
        """
        # Répertoires de résultats pour chaque minéral
        self.Olivine_results_dir = self.Path_res / "Mineral_detection" / "olivine"
        self.Pyroxene_results_dir = self.Path_res / "Mineral_detection" / "pyroxene"
        self.Amphiboles_results_dir = self.Path_res / "Mineral_detection" / "amphiboles"
        self.Carbonates_results_dir = self.Path_res / "Mineral_detection" / "carbonates"
        self.Micas_results_dir = self.Path_res / "Mineral_detection" / "micas"
        self.Argiles_results_dir = self.Path_res / "Mineral_detection" / "argiles"
        self.Iron_oxides_results_dir = self.Path_res / "Mineral_detection" / "iron_oxides"

        # Détection de l'olivine
        result = detect_olivine_bd1050_bd2000_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Olivine_results_dir,
            sampling_mode="nearest",
            verbose=True,
        )

        # Détection du pyroxène
        result = detect_pyroxene_bd1um_bd2um_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Pyroxene_results_dir,
            sampling_mode="nearest"
        )

        # Détection des amphiboles
        result = detect_amphiboles_bd2320_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Amphiboles_results_dir,
            sampling_mode="nearest",
        )

        # Détection des carbonates
        result = detect_carbonates_bd2330_bd2500_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Carbonates_results_dir,
            sampling_mode="nearest"
        )

        # Détection des micas
        result = detect_micas_bd2200_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Micas_results_dir,
            sampling_mode="linear",
        )

        # Détection des argiles
        result = detect_argiles_bd2200_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Argiles_results_dir,
            sampling_mode="linear",
        )

        # Détection des oxydes de fer
        result = detect_iron_oxides_bd900_redness_clean(
            tif_path=self.image_hyperspectrale_cleanbands_smooth,
            bands_csv=self.clean_wavelengths_csv,
            outdir=self.Iron_oxides_results_dir,
            sampling_mode="linear",
        )

    # Détection minérale en utilisant les méthodes de comparaison spectrale (SAM/MF/ACE)
    def process_spectral_comparison_mineral_detection(self, mineral: str) -> None:
        """
        Effectue la détection minérale en utilisant les méthodes de comparaison spectrale (SAM/MF/ACE).
        
        Args:
            mineral (str): Nom du minéral à détecter (ex: "arsenopyrite", "chalcopyrite").
        """
        self.smooth_image_for_spectral_comparison()  # Assurez-vous que l'image est lissée et normalisée pour la comparaison spectrale
        self.detect_minerals_spectral_comparison(mineral)

    def smooth_image_for_spectral_comparison(self) -> None:
        """
        Lisse et normalise l'image pour la comparaison spectrale (SAM/MF/ACE).
        Utilise un lissage léger et une normalisation L2.
        """
        self.image_hyperspectrale_cleanbands_smooth_norm = self.Path_res / "image_hyperspectrale_cleanbands_smooth_norm.tif"

        savgol_smooth_and_normalize(
            img_path=self.image_hyperspectrale_cleanbands,
            output_path=self.image_hyperspectrale_cleanbands_smooth_norm,
            window_length=9,  # Léger lissage pour EnMAP
            polyorder=2,
            normalize="l2"  # Recommandé pour SAM/MF/ACE
        )

    def detect_minerals_spectral_comparison(self, mineral: str) -> None:
        """
        Détecte un minéral spécifique en utilisant les méthodes de comparaison spectrale.
        
        Args:
            mineral (str): Nom du minéral à détecter (ex: "arsenopyrite", "chalcopyrite").
        """
        if not self.spectral_library_dir:
            raise ValueError("spectral_library_dir must be set in configuration")
        
        # Vérifier que l'image est prête
        if not hasattr(self, 'image_hyperspectrale_cleanbands_smooth_norm'):
            print("Smoothing image for spectral comparison...")
            self.smooth_image_for_spectral_comparison()
        
        # Créer le répertoire de sortie
        self.Spectral_results_dir = self.Path_res / "Spectral_mineral_detection"
        os.makedirs(self.Spectral_results_dir, exist_ok=True)
        
        print(f"Detecting {mineral} using spectral comparison methods...")
        
        # SAM + MF detection
        sam_out, mf_out, combo_out, used_refs = run_single_mineral_detection_from_tab_refs(
            img_tif_path=self.image_hyperspectrale_cleanbands_smooth_norm,
            bands_csv=self.clean_wavelengths_csv,
            ref_root_dir=self.spectral_library_dir,
            mineral=mineral,
            out_dir=str(self.Spectral_results_dir / "sam_mf" / mineral)
        )
        
        # SAM + ACE detection
        sam_out, ace_out, combo_out, used_refs = run_single_mineral_ace_from_tab_refs(
            img_tif_path=self.image_hyperspectrale_cleanbands_smooth_norm,
            bands_csv=self.clean_wavelengths_csv,
            ref_root_dir=self.spectral_library_dir,
            mineral=mineral,
            out_dir=str(self.Spectral_results_dir / "sam_ace" / mineral)
        )
        
        print(f"Spectral comparison detection completed for {mineral}")
