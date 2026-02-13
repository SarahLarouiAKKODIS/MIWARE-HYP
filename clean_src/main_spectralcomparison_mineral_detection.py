import os
import numpy as np
from mask_enmap import apply_water_veg_mask
from enmap_clean_bands import clean_bands_enmap_from_csv
from spectral_smoothing import savgol_smooth_and_normalize
from sam_mf_detection_spy import run_single_mineral_detection_from_txt_refs, run_single_mineral_detection_from_tab_refs

Path_res = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"
img_path = Path_res+ "image_hyperspectrale_crop.tif"
water_mask_path = Path_res + "Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"
veg_mask_path = Path_res + "Vegetation_indice_outputs/enmap_salsigne_VEG_MASK.tiff"
wavelengths_csv = Path_res + "enmap_bands_full.csv"

enmap_masked = Path_res + "enmap_masked.tif"
enmap_masked_cleanbands = Path_res + "enmap_masked_cleanbands.tif"
clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"

# spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/USGS/ASCIIdata_splib07a/ChapterM_Minerals/"
spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/RELAB/"

apply_water_veg_mask(
    img_path=img_path,
    water_mask_path=water_mask_path,
    veg_mask_path=veg_mask_path,
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
    exclude_ranges_nm=[(0,420), (1340,1460),(1800,1960),(2420,2445.5)],
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

out_dir = Path_res + "Spectral_mineral_detection/"
os.makedirs(out_dir, exist_ok=True)

mineral = "schwertmannite"

# run_single_mineral_detection_from_txt_refs(
#     img_tif_path=enmap_masked_cleanbands_smooth_norm,
#     bands_csv=clean_wavelengths_csv,
#     ref_dir=spectral_library_dir,
#     mineral=mineral,
#     out_dir=out_dir + mineral,
#     max_refs=10,           # None = tous ; sinon limite
#     seed=0,
#     assume_img_already_normalized=True
# )


sam_out, mf_out, combo_out, used_refs = run_single_mineral_detection_from_tab_refs(
    img_tif_path=enmap_masked_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + mineral
)
