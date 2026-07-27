from mineral_detection.olivine_detection import detect_olivine_bd1050_bd2000_clean
from mineral_detection.pyroxene_detection import detect_pyroxene_bd1um_bd2um_clean
from mineral_detection.amphiboles_detection import detect_amphiboles_bd2320_clean
from mineral_detection.carbonates_detection import detect_carbonates_bd2330_bd2500_clean
from mineral_detection.micas_detection import detect_micas_bd2200_clean
from mineral_detection.argiles_detection import detect_argiles_bd2200_clean
from mineral_detection.oxydesFer_detection import detect_iron_oxides_bd900_redness_clean
from preprocessing.spectral_smoothing import savgol_smooth_and_normalize

Path_res = "/home/starfox/FFE_WS/MIWARE-HYP/Results/20260324_Salsigne_2/"
clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"
image_hyperspectrale_cleanbands = Path_res + "image_hyperspectrale_cleanbands.tif"

## Lissage (optionel mais mieux)
image_hyperspectrale_cleanbands_smooth = Path_res + "image_hyperspectrale_cleanbands_smooth.tif"

savgol_smooth_and_normalize(
    img_path=image_hyperspectrale_cleanbands,
    output_path=image_hyperspectrale_cleanbands_smooth,
    normalize=None
)


## 4) Mineral detection

Olivine_results_dir= Path_res + "Mineral_detection/olivine/"

result = detect_olivine_bd1050_bd2000_clean(
        tif_path=image_hyperspectrale_cleanbands_smooth,
        bands_csv=clean_wavelengths_csv,
        outdir=Olivine_results_dir,
        sampling_mode="nearest", # "nearest"
        verbose=True,
    )

Pyroxene_results_dir= Path_res + "Mineral_detection/pyroxene/"

result = detect_pyroxene_bd1um_bd2um_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Pyroxene_results_dir,
    sampling_mode="nearest"
    )

Amphiboles_dir=Path_res + "Mineral_detection/amphiboles/"

result = detect_amphiboles_bd2320_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Amphiboles_dir,
    sampling_mode="nearest", # "nearest"
)

Carbonates_dir=Path_res + "Mineral_detection/carbonates/"

result = detect_carbonates_bd2330_bd2500_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Carbonates_dir,
    sampling_mode="nearest" # "linear"
)

Micas_dir=Path_res + "Mineral_detection/micas/"

result = detect_micas_bd2200_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Micas_dir,
    sampling_mode="linear", # "nearest"
)

Argiles_dir=Path_res + "Mineral_detection/argiles/"

result = detect_argiles_bd2200_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Argiles_dir,
    sampling_mode="linear", # "nearest"
)


Iron_oxides_dir=Path_res + "Mineral_detection/iron_oxides/"

result = detect_iron_oxides_bd900_redness_clean(
    tif_path=image_hyperspectrale_cleanbands_smooth,
    bands_csv=clean_wavelengths_csv,
    outdir=Iron_oxides_dir,
    sampling_mode="linear", # "nearest"
)

