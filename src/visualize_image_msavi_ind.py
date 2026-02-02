import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def save_and_show_msavi_tiff_with_legend(
    msavi: np.ndarray,
    output_path: str = "msavi_msavi_0_045.tiff",
    vmin: float = 0.0,
    vmax: float = 0.5,
    show: bool = True,
    low_rgb: tuple[int, int, int] = (245, 255, 245),  # vert très clair (presque blanc)
    high_rgb: tuple[int, int, int] = (0, 10, 0),      # vert presque noir
    nan_rgb: tuple[int, int, int] = (255, 0, 0),      # rouge
) -> np.ndarray:

    if msavi.ndim != 2:
        raise ValueError("msavi doit être un tableau 2D")

    msavi = msavi.astype(np.float32)
    nan_mask = ~np.isfinite(msavi)

    # Normalisation [0,1]
    norm = (msavi - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # inversion : valeur élevée -> foncé
    t = 1.0 - norm

    low = np.array(low_rgb, dtype=np.float32)
    high = np.array(high_rgb, dtype=np.float32)

    # interpolation couleur -> (H, W, 3) uint8
    rgb = (
        high[None, None, :] * (1.0 - t[..., None])
        + low[None, None, :] * t[..., None]
    ).astype(np.uint8)

    # NaN en rouge
    rgb[nan_mask] = np.array(nan_rgb, dtype=np.uint8)

    # Sécurité : vérifier la forme
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"RGB a une forme inattendue: {rgb.shape} (attendu HxWx3)")

    # Enregistrement TIFF: forcer l'interprétation RGB
    tiff.imwrite(output_path, rgb, photometric="rgb")

    # Affichage
    if show:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(rgb)
        ax.set_title("MSAVI (0 = vert clair, 0.45 = vert foncé, rouge = NaN)")
        ax.axis("off")

        cmap = LinearSegmentedColormap.from_list(
            "msavi_custom",
            [np.array(low_rgb) / 255.0, np.array(high_rgb) / 255.0],
            N=256
        )
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Valeur MSAVI")
        plt.show()

    return rgb




if __name__ == "__main__":

    image_msavi = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_WDI_veg.tiff"
    output_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_WDI_veg_green.tiff"
    
    with rasterio.open(image_msavi) as src:
        msavi = src.read(1)   # bande 1 → tableau 2D (H, W)

        msavi_min = np.nanmin(msavi)
        msavi_max = np.nanmax(msavi)

        print("MSAVI min :", msavi_min)
        print("MSAVI max :", msavi_max)


        #save_and_show_msavi_tiff_with_legend(msavi, output_path)

