import os
import numpy as np
from spectral_comparison_methodes.sam_mf_detection_spy import run_single_mineral_detection_from_tab_refs
from spectral_comparison_methodes.sam_ace_detection import run_single_mineral_ace_from_tab_refs
from preprocessing.spectral_smoothing import savgol_smooth_and_normalize

Path_res = "/home/starfox/FFE_WS/MIWARE-HYP/Results/20260324_Salsigne_2/"

clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"
image_hyperspectrale_cleanbands = Path_res + "image_hyperspectrale_cleanbands.tif"

## Lissage and normalisation
image_hyperspectrale_cleanbands_smooth_norm = Path_res + "image_hyperspectrale_cleanbands_smooth_norm.tif"

savgol_smooth_and_normalize(
    img_path=image_hyperspectrale_cleanbands,
    output_path=image_hyperspectrale_cleanbands_smooth_norm,
    window_length=9,   # 7 ou 9 = “léger” pour EnMAP
    polyorder=2,
    normalize="l2"     # "l2" conseillé pour SAM/MF
    
)

# spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/USGS/ASCIIdata_splib07a/ChapterM_Minerals/"
spectral_library_dir = "/home/starfox/FFE_WS/MIWARE-HYP/Data/Librairies_spectrales/RELAB/"

out_dir = Path_res + "Spectral_mineral_detection/"
os.makedirs(out_dir, exist_ok=True)

mineral = "arsenopyrite" #arsenopyrite #chalcopyrite

## RELAB

## SAM + MF
sam_out, mf_out, combo_out, used_refs = run_single_mineral_detection_from_tab_refs(
    img_tif_path=image_hyperspectrale_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_mf/" + mineral
)

## SAM + ACE
sam_out, ace_out, combo_out, used_refs = run_single_mineral_ace_from_tab_refs(
    img_tif_path=image_hyperspectrale_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_ace/" + mineral
)

