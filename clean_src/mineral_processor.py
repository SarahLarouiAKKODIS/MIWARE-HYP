"""
Module contenant la classe MineralProcessor pour traiter les images hyperspectrales.
"""

from .utils.enmap_band_utils import recover_wavelet_band_info
from .enmap_quality_mask import mask_enmap_hyperspectral_cube
from .utils.enmap_crop_image import crop_hyperspectral_tif
from .utils.analyse_hyperspectral_image import analyze_rescaled_cube_with_wavelengths
from .enmap_water_indices import compute_mndwi_and_water_mask
from .enmap_vegetation_indices import compute_vegetation_indices_wdi_vii

from .preprocessing.mask_enmap import apply_water_veg_mask
from .preprocessing.enmap_clean_bands import clean_bands_enmap_from_csv
from .preprocessing.rescaling_image import rescale_enmap_cube_simple
import json
from .utils.enmap_rgb_extraction import hyperspectral_to_rgb
from pathlib import Path


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
        print(f"Configuration loaded from {config_path}")

    def process(self) -> None:
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
