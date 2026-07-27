from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


def compute_mean_spectrum_per_class(
    image_tif,
    gt_mask_tif,
    wavelengths,
    class_names=None,
    ignore_labels=(0,),
    min_pixels=10,
    valid_pixels_mask=None,
    output_csv=None,
):
    image_tif = Path(image_tif)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)

    if class_names is not None:
        class_names = {int(k): v for k, v in class_names.items()}

    with rasterio.open(image_tif) as src_img:
        cube = src_img.read().astype(np.float32)

    if isinstance(gt_mask_tif, np.ndarray):
        gt_mask = gt_mask_tif
        mask_nodata = None
    else:
        gt_mask_tif = Path(gt_mask_tif)
        with rasterio.open(gt_mask_tif) as src_mask:
            gt_mask = src_mask.read(1)
            mask_nodata = src_mask.nodata

    if cube.shape[1:] != gt_mask.shape:
        raise ValueError(
            f"Dimensions incompatibles : cube={cube.shape[1:]}, masque={gt_mask.shape}"
        )

    if cube.shape[0] != len(wavelengths):
        raise ValueError(
            f"Nombre de longueurs d'onde incompatible : "
            f"{len(wavelengths)} wavelengths pour {cube.shape[0]} bandes"
        )

    valid_pixels = np.isfinite(cube).all(axis=0)

    if valid_pixels_mask is not None:
        valid_pixels_mask = np.asarray(valid_pixels_mask).astype(bool)

        if valid_pixels_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Dimensions incompatibles : valid_pixels_mask={valid_pixels_mask.shape}, "
                f"masque={gt_mask.shape}"
            )

        valid_pixels &= valid_pixels_mask

    if mask_nodata is not None:
        valid_pixels &= (gt_mask != mask_nodata)

    mean_spectra = {}

    for class_id in sorted(np.unique(gt_mask)):
        class_id = int(class_id)

        if class_id in ignore_labels:
            continue

        mask = (gt_mask == class_id) & valid_pixels
        n_pixels = int(np.count_nonzero(mask))

        if n_pixels < min_pixels:
            continue

        spectra = cube[:, mask]

        mean_spectrum = np.nanmean(spectra, axis=1)
        std_spectrum = np.nanstd(spectra, axis=1)

        mean_spectra[class_id] = {
            "class_name": class_names.get(class_id, str(class_id)) if class_names else str(class_id),
            "n_pixels": n_pixels,
            "mean_spectrum": mean_spectrum.astype(np.float32),
            "std_spectrum": std_spectrum.astype(np.float32),
        }

    if output_csv is not None:
        data = {
            "wavelength_nm": wavelengths
        }

        for class_id, values in mean_spectra.items():
            data[values["class_name"]] = values["mean_spectrum"]

        df = pd.DataFrame(data)

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)

        print(f"Spectres moyens sauvegardés : {output_csv}")

    return mean_spectra