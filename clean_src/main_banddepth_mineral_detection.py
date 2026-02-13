from enmap_olivine_detection import detect_olivine_bd1050_bd2000
from enmap_pyroxene_detection import detect_pyroxene_bd1um_bd2um
from enmap_amphiboles_detection import detect_amphiboles_bd2320
from enmap_carbonates_detection import detect_carbonates_bd2330_bd2500
from enmap_micas_detection import detect_micas_bd2200
from enmap_argiles_detection import detect_argiles_bd2200
from enmap_oxydesFer_detection import detect_iron_oxides_bd900_redness

Path_res = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"
image_hyperspectrale_clean_crop = Path_res+ "image_hyperspectrale_crop.tif"
wavelengths_csv = Path_res + "enmap_bands_full.csv"
Vegetation_results_dir= Path_res + "Vegetation_indice_outputs"
Water_results_dir = Path_res + "Water_indice_outputs"

## 4) Mineral detection

Olivine_results_dir= Path_res + "Mineral_detection/olivine/"

res_olivine = detect_olivine_bd1050_bd2000(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Olivine_results_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True )


Pyroxene_results_dir= Path_res + "Mineral_detection/pyroxene/"

res = detect_pyroxene_bd1um_bd2um(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Pyroxene_results_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True 
)

Amphiboles_dir=Path_res + "Mineral_detection/amphiboles/"

res = detect_amphiboles_bd2320(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Amphiboles_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True 
)

Carbonates_dir=Path_res + "Mineral_detection/carbonates/"

res = detect_carbonates_bd2330_bd2500(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Carbonates_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True 
)

Micas_dir=Path_res + "Mineral_detection/micas/"

res = detect_micas_bd2200(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Micas_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True 
)

Argiles_dir=Path_res + "Mineral_detection/argiles/"

res = detect_argiles_bd2200(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Argiles_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True, 
    write_outputs=True 
)


Iron_oxides_dir=Path_res + "Mineral_detection/iron_oxides/"

res = detect_iron_oxides_bd900_redness(
    tif_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    outdir=Iron_oxides_dir,
    veg_mask_path=Vegetation_results_dir +"/enmap_salsigne_VEG_MASK.tiff",
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    apply_land_mask=True, 
)
