from utils.enmap_band_utils import recover_wavelet_band_info
from enmap_quality_mask import mask_enmap_hyperspectral_cube
from utils.enmap_crop_image import crop_hyperspectral_tif
from utils.analyse_hyperspectral_image import analyze_rescaled_cube_with_wavelengths
from enmap_water_indices import compute_mndwi_and_water_mask
from enmap_vegetation_indices import compute_vegetation_indices_wdi_vii

from preprocessing.mask_enmap import apply_water_veg_mask
from preprocessing.enmap_clean_bands import clean_bands_enmap_from_csv
from preprocessing.rescaling_image import rescale_enmap_cube_simple
import json
from utils.enmap_rgb_extraction import hyperspectral_to_rgb
from pathlib import Path

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():

    config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/salsigne_3.json"
    config = load_config(config_path)

    Path_res = Path(config["Path_res"])
    Path_res.mkdir(parents=True, exist_ok=True)
    xml_path = Path(config["xml_path"])
    image_hyperspectrale = Path(config["image_hyp"])
    Cloud_mask = Path(config["Cloud_mask"])
    Haze_mask = Path(config["Haze_mask"])
    Cirrus_mask = Path(config["Cirrus_mask"])
    CloudShadow_mask = Path(config["CloudShadow_mask"])
    Snow_mask = Path(config["Snow_mask"])
    TestFlags_mask = Path(config["TestFlags_mask"] )
    Site_coords = config["Site_coords"]
    Crop = config["crop"]

    # 1) Wavelet band information recovery
    wavelengths_csv = Path_res / "enmap_bands_full.csv"

    df = recover_wavelet_band_info(
        xml_path,
        out_csv=wavelengths_csv
    )

    print("Table complète enregistrée : enmap_bands_full.csv")
    print(df.head(10).to_string(index=False))

    # 2) Rescaling image
    image_hyperspectrale_reflectance = Path_res / "image_hyperspectrale_reflectance.tif"

    rescale_enmap_cube_simple(
        input_tif=image_hyperspectrale,
        output_tif=image_hyperspectrale_reflectance,
        scale_factor= 10000.0
    )

    # 3) Some pixels exclusion (cloud, haze, ...)
    mask_files = [Cloud_mask, Haze_mask, Cirrus_mask, CloudShadow_mask, Snow_mask, TestFlags_mask]

    image_hyperspectrale_clean = Path_res / "image_hyperspectrale_clean.tif"

    mask_enmap_hyperspectral_cube(
        cube_tif=image_hyperspectrale_reflectance,
        mask_files=mask_files,
        out_tif=image_hyperspectrale_clean
    )

    # 4) Crop image
    if Crop :

        image_hyperspectrale_clean_crop = Path_res / "image_hyperspectrale_crop.tif"
        crop_hyperspectral_tif(image_hyperspectrale_clean, image_hyperspectrale_clean_crop, Site_coords)
        print("Crop enregistré :", image_hyperspectrale_clean_crop)
    else:
        image_hyperspectrale_clean_crop = image_hyperspectrale_clean


    # 5) RGB image
    image_hyperspectrale_rgb = Path_res / "image_hyperspectrale_rgb.tif"

    hyperspectral_to_rgb(
        cube_tif=image_hyperspectrale_clean_crop,
        bands_csv=wavelengths_csv,
        out_rgb_tif=image_hyperspectrale_rgb,
        target_R=650,
        target_G=560,
        target_B=480,
    )

        # 6) Clean bands  
    image_hyperspectrale_cleanbands = Path_res / "image_hyperspectrale_cleanbands.tif"
    clean_wavelengths_csv = Path_res / "enmap_clean_bands_full.csv"  

    summary = clean_bands_enmap_from_csv(
        img_path=image_hyperspectrale_clean_crop,
        bands_csv=wavelengths_csv,
        output_path=image_hyperspectrale_cleanbands,
        output_bands_csv=clean_wavelengths_csv,   # <- le CSV corrigé
        band_id_is_one_based=True,           # comme ton CSV actuel
        csv_band_id_is_one_based_out=True,    # sortie en 1..N (recommandé)
        drop_edges=(2, 2),
        exclude_ranges_nm=[(1340,1460),(1800,1960)],
        use_fwhm_margin=True,
        fwhm_factor=0.5
    )

    print(summary["nbands_in"], "->", summary["nbands_out"])
    print("Removed wavelengths (nm):", summary["removed_wavelengths_nm"])
    print("Bands out:", summary["nbands_out"])

    # 7) QC
    qc_res = analyze_rescaled_cube_with_wavelengths(
        cube_tif=image_hyperspectrale_cleanbands,
        bands_csv=clean_wavelengths_csv,
        out_csv=Path_res / "band_qc_report.csv",
        min_valid=-0.1,
        max_valid=1.5,
        nan_threshold_pct=50.0,
        outlier_threshold_pct=5.0,
        band_id_is_one_based=True,
    )

    # 8) Water mask calculation
    Water_results_dir = Path_res / "Water_indice_outputs"

    water_res = compute_mndwi_and_water_mask(
        tif_path=image_hyperspectrale_cleanbands,
        wavelengths_csv=clean_wavelengths_csv,
        outdir=Water_results_dir,
        prefix="enmap_salsigne",
        mndwi_th=0.55,
        verbose=True,
    )

    print(water_res["paths"])

    # 9) Vegetation mask calculation
    Vegetation_results_dir= Path_res / "Vegetation_indice_outputs"

    veg_res = compute_vegetation_indices_wdi_vii(
        tif_path=image_hyperspectrale_cleanbands,
        wavelengths_csv=clean_wavelengths_csv,
        outdir=Vegetation_results_dir,
        prefix="enmap_salsigne",
        ndvi_th=0.3,
        verbose=True
    )

    print(veg_res["paths"].keys())

    # 10) Apply water + vegetation mask => sol nu ?
    enmap_mineral_candidates = Path_res / "enmap_mineral_candidates.tif"

    apply_water_veg_mask(
        img_path=image_hyperspectrale_cleanbands,
        water_mask_path=Water_results_dir / "enmap_salsigne_WATER_MASK.tiff",
        veg_mask_path=Vegetation_results_dir / "enmap_salsigne_VEG_MASK.tiff",
        output_path=enmap_mineral_candidates
    )

    



if __name__ == "__main__":
    main()