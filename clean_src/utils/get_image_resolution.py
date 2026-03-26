import rasterio

with rasterio.open("/home/sarah.laroui/Bureau/MIWARE-HYP/cartes_INRA_RMQS/dataverse_files/mediane/surface/analyse_HF/ni_tot_hf_CE_KED_C1.tif") as src:
    print("Résolution:", src.res)