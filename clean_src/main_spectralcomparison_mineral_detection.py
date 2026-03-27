import os
import numpy as np
from spectral_comparison_methodes.sam_mf_detection_spy import run_single_mineral_detection_from_tab_refs
from spectral_comparison_methodes.sam_ace_detection import run_single_mineral_ace_from_tab_refs

Path_res = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"

clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"

# spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/USGS/ASCIIdata_splib07a/ChapterM_Minerals/"
spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/RELAB/"

#image hyperpectrale masquée et pré-traitée (lissage et normalisation)
enmap_masked_cleanbands_smooth_norm = Path_res + "enmap_masked_cleanbands_smooth_norm.tif"

out_dir = Path_res + "Spectral_mineral_detection/"
os.makedirs(out_dir, exist_ok=True)

mineral = "arsenopyrite"

## RELAB

## SAM + MF
sam_out, mf_out, combo_out, used_refs = run_single_mineral_detection_from_tab_refs(
    img_tif_path=enmap_masked_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_mf/" + mineral
)

## SAM + ACE
sam_out, ace_out, combo_out, used_refs = run_single_mineral_ace_from_tab_refs(
    img_tif_path=enmap_masked_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_ace/" + mineral
)

