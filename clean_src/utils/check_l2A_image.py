#!/usr/bin/env python3

"""
check_hyperspectral_image.py

Contrôle qualité d'une image hyperspectrale EnMAP.

Le script prend en entrée un DOSSIER et recherche automatiquement
le fichier se terminant par :

    -SPECTRAL_IMAGE.TIF

Usage :
    python check_hyperspectral_image.py /chemin/vers/dossier_enmap/
"""

from pathlib import Path
import argparse

import numpy as np
import rasterio


# ============================================================
# PARAMÈTRES
# ============================================================

SPECTRAL_SUFFIX = "-SPECTRAL_IMAGE.TIF"

# Une bande est considérée comme quasi nulle si au moins
# 99 % des pixels valides dans l'emprise sont égaux à zéro.
ZERO_BAND_THRESHOLD = 99.0

# À partir de ce nombre de bandes quasi nulles,
# l'image est considérée comme incomplète / corrompue.
CORRUPTED_ZERO_BANDS_THRESHOLD = 20


# ============================================================
# RECHERCHE DE L'IMAGE HYPERSPECTRALE
# ============================================================

def find_spectral_image(folder):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Le chemin fourni n'est pas un dossier : {folder}"
        )

    # Recherche dans le dossier ET ses sous-dossiers.
    # lower() permet aussi de tolérer .tif / .TIF.
    spectral_images = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.name.upper().endswith(SPECTRAL_SUFFIX)
    )

    if len(spectral_images) == 0:
        raise FileNotFoundError(
            f"Aucune image se terminant par "
            f"'{SPECTRAL_SUFFIX}' trouvée dans :\n"
            f"{folder}"
        )

    if len(spectral_images) > 1:

        print("\nPlusieurs images hyperspectrales trouvées :")

        for image in spectral_images:
            print(f"  - {image}")

        raise RuntimeError(
            f"{len(spectral_images)} images "
            f"'{SPECTRAL_SUFFIX}' trouvées. "
            "Impossible de choisir automatiquement."
        )

    return spectral_images[0]


# ============================================================
# ANALYSE
# ============================================================

def analyse_image(image_path):

    image_path = Path(image_path)

    print("=" * 100)
    print("CONTRÔLE QUALITÉ IMAGE HYPERSPECTRALE")
    print("=" * 100)
    print(f"Image : {image_path}")

    # ========================================================
    # LECTURE
    # ========================================================

    with rasterio.open(image_path) as src:

        cube = src.read(masked=False)

        nodata = src.nodata
        bands = src.count
        rows = src.height
        cols = src.width
        crs = src.crs

    print()
    print(f"Shape  : {cube.shape}")
    print(f"dtype  : {cube.dtype}")
    print(f"nodata : {nodata}")
    print(f"CRS    : {crs}")

    # Passage en float pour pouvoir utiliser NaN
    cube = cube.astype(np.float32)

    # ========================================================
    # NODATA -> NaN
    # ========================================================

    if nodata is not None:

        if not np.isnan(nodata):
            cube[cube == nodata] = np.nan

    cube[~np.isfinite(cube)] = np.nan

    # ========================================================
    # EMPRISE CONTENANT DU SIGNAL
    # ========================================================

    # Un pixel appartient à l'emprise s'il possède au moins
    # une bande finie ET différente de zéro.
    footprint = np.any(
        np.isfinite(cube) & (cube != 0),
        axis=0,
    )

    n_data_pixels = int(footprint.sum())
    total_pixels = rows * cols

    data_percent = (
        100.0 * n_data_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    print()
    print("=" * 100)
    print("EMPRISE")
    print("=" * 100)

    print(f"Pixels totaux              : {total_pixels}")
    print(f"Pixels contenant du signal : {n_data_pixels}")
    print(f"Part du raster             : {data_percent:.4f} %")

    if n_data_pixels == 0:

        print()
        print("❌ CORROMPUE / INCOMPLÈTE")
        print("Aucun pixel contenant de signal.")

        return

    # ========================================================
    # ANALYSE BANDE PAR BANDE
    # ========================================================

    fully_zero_bands = []
    quasi_zero_bands = []
    fully_nodata_bands = []
    bands_with_signal = []

    print()
    print("=" * 100)
    print("ANALYSE BANDE PAR BANDE")
    print("=" * 100)

    for b in range(bands):

        band = cube[b]

        # On regarde uniquement les pixels appartenant
        # à l'emprise de l'image.
        values = band[footprint]

        finite_mask = np.isfinite(values)
        n_valid = int(finite_mask.sum())

        # ----------------------------------------------------
        # Bande entièrement NODATA
        # ----------------------------------------------------

        if n_valid == 0:

            fully_nodata_bands.append(b + 1)

            print(
                f"Bande {b + 1:3d} | "
                f"NODATA"
            )

            continue

        x = values[finite_mask]

        # ----------------------------------------------------
        # Zéros
        # ----------------------------------------------------

        n_zero = int(np.sum(x == 0))

        zero_percent = (
            100.0 * n_zero / len(x)
        )

        # ----------------------------------------------------
        # Bande entièrement à zéro
        # ----------------------------------------------------

        if np.all(x == 0):
            fully_zero_bands.append(b + 1)

        # ----------------------------------------------------
        # Bande quasi nulle
        # ----------------------------------------------------

        if zero_percent >= ZERO_BAND_THRESHOLD:
            quasi_zero_bands.append(b + 1)

        # ----------------------------------------------------
        # Valeurs non nulles
        # ----------------------------------------------------

        nonzero = x[x != 0]

        if len(nonzero) > 0:

            bands_with_signal.append(b + 1)

            min_value = np.min(nonzero)
            median_value = np.median(nonzero)
            max_value = np.max(nonzero)

        else:

            min_value = np.nan
            median_value = np.nan
            max_value = np.nan

        print(
            f"Bande {b + 1:3d} | "
            f"zero={zero_percent:6.2f}% | "
            f"N valide={n_valid:8d} | "
            f"N non-zero={len(nonzero):8d} | "
            f"min={min_value:.4g} | "
            f"median={median_value:.4g} | "
            f"max={max_value:.4g}"
        )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    n_signal = len(bands_with_signal)
    n_zero = len(fully_zero_bands)
    n_quasi_zero = len(quasi_zero_bands)
    n_nodata = len(fully_nodata_bands)

    print()
    print("=" * 100)
    print("RÉSUMÉ")
    print("=" * 100)

    print(f"Bandes totales             : {bands}")
    print(f"Bandes contenant du signal : {n_signal}")
    print(f"Bandes entièrement NODATA  : {n_nodata}")
    print(f"Bandes entièrement à zéro  : {n_zero}")

    print(
        f"Bandes >= {ZERO_BAND_THRESHOLD:.0f}% de zéros     : "
        f"{n_quasi_zero}"
    )

    if fully_nodata_bands:

        print()
        print("Bandes NODATA :")
        print(fully_nodata_bands)

    if fully_zero_bands:

        print()
        print("Bandes entièrement à zéro :")
        print(fully_zero_bands)

    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)

    if n_quasi_zero >= CORRUPTED_ZERO_BANDS_THRESHOLD:

        print("❌ CORROMPUE / INCOMPLÈTE")

        print(
            f"{n_quasi_zero} bandes ont au moins "
            f"{ZERO_BAND_THRESHOLD:.0f}% de valeurs nulles."
        )

    elif n_quasi_zero > 0:

        print("⚠️ SUSPECTE")

        print(
            f"{n_quasi_zero} bande(s) ont au moins "
            f"{ZERO_BAND_THRESHOLD:.0f}% de valeurs nulles."
        )

    else:

        print("✅ OK")

        print(
            "Aucune bande anormalement remplie de zéros "
            "n'a été détectée."
        )

    print()
    print(
        f"Emprise contenant du signal : "
        f"{data_percent:.4f}% du raster "
        f"({n_data_pixels} pixels)."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Recherche et contrôle l'image hyperspectrale "
            "EnMAP d'un dossier."
        )
    )

    parser.add_argument(
        "folder",
        help=(
            "Dossier contenant l'image EnMAP "
            "se terminant par -SPECTRAL_IMAGE.TIF"
        ),
    )

    args = parser.parse_args()

    # Recherche automatique du fichier hyperspectral
    spectral_image = find_spectral_image(
        args.folder
    )

    print()
    print(
        f"Image hyperspectrale trouvée : "
        f"{spectral_image}"
    )
    print()

    # Analyse
    analyse_image(
        spectral_image
    )