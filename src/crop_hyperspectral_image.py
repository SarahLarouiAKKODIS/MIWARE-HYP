import rasterio
from rasterio.windows import Window

def crop_hyperspectral_tif(
    in_tif: str,
    out_tif: str,
    x_min: int = 756,
    y_min: int = 646,
    x_max: int = 945,
    y_max: int = 780,
):
    """
    Crop par coordonnées pixel (col=x, row=y) avec:
      - coin haut-gauche (x_min, y_min)
      - coin bas-droit (x_max, y_max)  (exclus si tu suis la convention Window)
    """
    # Largeur/hauteur du crop en pixels
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Coordonnées invalides: x_max doit être > x_min et y_max > y_min")

    with rasterio.open(in_tif) as src:
        # Fenêtre de lecture (col_off, row_off, width, height)
        window = Window(col_off=x_min, row_off=y_min, width=width, height=height)

        # Lire toutes les bandes dans la fenêtre
        data = src.read(window=window)  # shape: (bands, height, width)

        # Mettre à jour le profil (taille + transform géographique)
        profile = src.profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=rasterio.windows.transform(window, src.transform),
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data)

if __name__ == "__main__":
    # À MODIFIER
    Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/"
    # input_tif = Path_data + "SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-SPECTRAL_IMAGE.TIF"
    # output_tif = Path_data+ "image_hyperspectrale_crop.tif"

    # input_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean.tif"
    # output_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/image_hyperspectrale_clean_crop.tif"

    input_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/quality_layer_colored.tif"
    output_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/quality_layer_colored_crop.tif"

    crop_hyperspectral_tif(input_tif, output_tif)
    print("✅ Crop enregistré :", output_tif)
