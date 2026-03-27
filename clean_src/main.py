from utils.enmap_band_utils import recover_wavelet_band_info
from enmap_quality_mask import mask_enmap_hyperspectral_cube
from utils.enmap_crop_image import crop_hyperspectral_tif
from enmap_water_indices import compute_mndwi_and_water_mask
from enmap_vegetation_indices import compute_vegetation_indices_wdi_vii

from preprocessing.mask_enmap import apply_water_veg_mask
from preprocessing.enmap_clean_bands import clean_bands_enmap_from_csv
from preprocessing.spectral_smoothing import savgol_smooth_and_normalize

## 1) wavelet band information recovery

Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/"
xml_path = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-METADATA.XML"

Path_res = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"
wavelengths_csv = Path_res + "enmap_bands_full.csv"

df = recover_wavelet_band_info(
    xml_path,
    out_csv=wavelengths_csv
)

print("Table complète enregistrée : enmap_bands_full.csv")
print(df.head(10).to_string(index=False))

## 2) Some pixels exclusion (cloud, haze, ...)

image_hyperspectrale = Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"

mask_files = [
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUD.TIF",
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_HAZE.TIF",
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CIRRUS.TIF",
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLOUDSHADOW.TIF",
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_SNOW.TIF",
    Path_data + "ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_TESTFLAGS.TIF",
]


image_hyperspectrale_clean = Path_res+ "image_hyperspectrale_clean.tif"

mask_enmap_hyperspectral_cube(
    cube_tif=image_hyperspectrale,
    mask_files=mask_files,
    out_tif=image_hyperspectrale_clean
)

## 3) crop image

image_hyperspectrale_clean_crop = Path_res+ "image_hyperspectrale_crop.tif"

crop_hyperspectral_tif(image_hyperspectrale_clean, image_hyperspectrale_clean_crop)
print("Crop enregistré :", image_hyperspectrale_clean_crop)

## 3) Water mask calculation

Water_results_dir = Path_res + "Water_indice_outputs"

res = compute_mndwi_and_water_mask(
    tif_path=image_hyperspectrale_clean_crop,
    wavelengths_csv=wavelengths_csv,
    outdir=Water_results_dir,
    prefix="enmap_salsigne",
    mndwi_th=0.55,
    verbose=True
)

print(res["paths"])

## 3) Vegetation mask calculation and health evaluation

Vegetation_results_dir= Path_res + "Vegetation_indice_outputs"

res = compute_vegetation_indices_wdi_vii(
    tif_path=image_hyperspectrale_clean_crop,
    wavelengths_csv=wavelengths_csv,
    outdir=Vegetation_results_dir,
    prefix="enmap_salsigne",
    ndvi_th=0.3,
    verbose=True
)

print(res["paths"].keys())

####################################################################################

enmap_masked = Path_res + "enmap_masked.tif"
enmap_masked_cleanbands = Path_res + "enmap_masked_cleanbands.tif"
clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"

apply_water_veg_mask(
    img_path=image_hyperspectrale_clean_crop,
    water_mask_path=Water_results_dir + "/enmap_salsigne_WATER_MASK.tiff",
    veg_mask_path=Vegetation_results_dir + "/enmap_salsigne_VEG_MASK.tiff",
    output_path=enmap_masked
)


summary = clean_bands_enmap_from_csv(
    img_path=enmap_masked,
    bands_csv=wavelengths_csv,
    output_path=enmap_masked_cleanbands,
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

enmap_masked_cleanbands_smooth_norm = Path_res + "enmap_masked_cleanbands_smooth_norm.tif"

savgol_smooth_and_normalize(
    img_path=enmap_masked_cleanbands,
    output_path=enmap_masked_cleanbands_smooth_norm,
    window_length=9,   # 7 ou 9 = “léger” pour EnMAP
    polyorder=2,
    normalize="l2"     # "l2" conseillé pour SAM/MF
    
)