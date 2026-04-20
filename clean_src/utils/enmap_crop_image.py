import rasterio
from rasterio.windows import Window

def crop_hyperspectral_tif(
    in_tif: str,
    out_tif: str,
    coord: list,
):
    """
    Crop par coordonnées pixel (col=x, row=y) avec:
      - coin haut-gauche (x_min, y_min)
      - coin bas-droit (x_max, y_max)  (exclus si tu suis la convention Window)
    """
    # Largeur/hauteur du crop en pixels
    x_min, y_min, x_max, y_max = coord
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Coordonnées invalides: x_max doit être > x_min et y_max > y_min")

    with rasterio.open(in_tif) as src:
        if x_min < 0 or y_min < 0 or x_max > src.width or y_max > src.height:
            raise ValueError(
                f"Fenêtre hors limites : image={src.width}x{src.height}, "
                f"crop=({x_min}, {y_min}, {x_max}, {y_max})"
            )
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