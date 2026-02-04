import rasterio
import numpy as np


def compare_pixels(
    tif_a: str,
    tif_b: str,
    value_a: int = 255,
    value_b: int = 2,
) -> dict:
    """
    Compare deux images .tif :
    vérifie que les pixels == value_a dans tif_a
    correspondent aux pixels == value_b dans tif_b.

    Retourne des statistiques et le nombre d'erreurs.
    """

    # Lecture des deux images
    with rasterio.open(tif_a) as src_a, rasterio.open(tif_b) as src_b:
        a = src_a.read(1)
        b = src_b.read(1)

        if a.shape != b.shape:
            raise ValueError("Les deux images n'ont pas la même taille.")

    # Masques logiques
    mask_a = a == value_a
    mask_b = b == value_b

    values_a, counts_a = np.unique(a, return_counts=True)
    values_b, counts_b = np.unique(b, return_counts=True)

    for v, c in zip(values_a, counts_a):
        print(f"A Value {v}: {c} pixels")

    for v, c in zip(values_b, counts_b):
        print(f"B Value {v}: {c} pixels")

    # Pixels attendus mais incorrects
    errors_a_to_b = mask_a & (~mask_b)
    errors_b_to_a = mask_b & (~mask_a)
   

    # Comptes
    total_a = np.sum(mask_a)
    total_b = np.sum(mask_b)
    errors_a = np.sum(errors_a_to_b)
    errors_b = np.sum(errors_b_to_a)

    results = {
        "pixels_value_a": int(total_a),
        "pixels_value_b": int(total_b),
        "errors_value_a_not_value_b": int(errors_a),
        "errors_value_b_not_value_a": int(errors_b),
        "total_errors": int(errors_a + errors_b),
        "perfect_match": (errors_a + errors_b) == 0,
    }

    return results


# Exemple d'utilisation
if __name__ == "__main__":

    # Mask_result = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"
    # Mask_EnMap = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/quality_layer_colored_crop.tif"
    Mask_result ="/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs_all_image/enmap_salsigne_WATER_MASK.tiff"
    Mask_EnMap = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/mask_enmap_allimage_water.tiff"
    res = compare_pixels(
        Mask_result,
        Mask_EnMap,
        value_a=255,
        value_b=255,
    )

    for k, v in res.items():
        print(f"{k}: {v}")
