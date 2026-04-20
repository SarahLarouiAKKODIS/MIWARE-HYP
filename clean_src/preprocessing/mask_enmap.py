import rasterio
import numpy as np

def apply_water_veg_mask(
    img_path,
    water_mask_path,
    veg_mask_path,
    output_path="enmap_masked.tif",
    nodata_value=np.nan
):
    """
    Applique un masque eau + végétation à une image hyperspectrale.

    Pixels exclus si valeur = 255 ou -1 dans l'un des masques:
    eau (255)
    végétation (255)
    nodata (-1)

    Parameters
    ----------
    img_path : str
        Chemin vers l'image EnMAP (L2A).
    water_mask_path : str
        Chemin vers le masque eau.
    veg_mask_path : str
        Chemin vers le masque végétation.
    output_path : str
        Chemin du fichier de sortie.
    nodata_value : float
        Valeur NoData (np.nan recommandé).
    """

    with rasterio.open(img_path) as src:
        img = src.read()
        profile = src.profile
        rows, cols = src.height, src.width

    with rasterio.open(water_mask_path) as wm:
        water_mask = wm.read(1)

    with rasterio.open(veg_mask_path) as vm:
        veg_mask = vm.read(1)

    # --- Correction si axes inversés ---
    if water_mask.shape == (cols, rows):
        water_mask = water_mask.T
    if veg_mask.shape == (cols, rows):
        veg_mask = veg_mask.T

    # --- Vérification stricte ---
    assert water_mask.shape == (rows, cols), \
        f"Water mask {water_mask.shape} != {(rows, cols)}"

    assert veg_mask.shape == (rows, cols), \
        f"Veg mask {veg_mask.shape} != {(rows, cols)}"

    # --- Construction masque combiné ---
    combined_mask = (
        (water_mask == 255) | (water_mask == -1) |
        (veg_mask == 255)   | (veg_mask == -1)
    )

    # --- Conversion float pour NaN ---
    img = img.astype("float32")

    # --- Application masque ---
    img[:, combined_mask] = nodata_value

    # --- Mise à jour profil ---
    profile.update(dtype="float32", nodata=nodata_value)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(img)

    print("Masquage terminé ✔")
