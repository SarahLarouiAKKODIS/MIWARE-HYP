#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import rasterio


def check_zero_bands(image_path):

    image_path = Path(image_path)

    print("=" * 80)
    print("RECHERCHE DES BANDES NULLES OU QUASI NULLES")
    print("=" * 80)
    print("Image :", image_path)

    # =========================================================
    # Lecture du cube
    # =========================================================

    with rasterio.open(image_path) as src:
        cube = src.read().astype(np.float32)
        nodata = src.nodata

    print("Shape :", cube.shape)
    print("Nodata :", nodata)

    # =========================================================
    # Gestion des nodata
    # =========================================================

    if nodata is not None and np.isfinite(nodata):
        cube[cube == nodata] = np.nan

    cube[~np.isfinite(cube)] = np.nan

    # =========================================================
    # Détermination de l'emprise réelle
    # =========================================================

    # Un pixel appartient à l'image si au moins une bande
    # contient une valeur finie ET non nulle.
    #
    # Cela évite de considérer le fond du raster comme des
    # pixels réels.

    footprint = np.any(
        np.isfinite(cube) & (cube != 0),
        axis=0,
    )

    n_pixels = int(footprint.sum())

    print("Pixels dans l'emprise réelle :", n_pixels)

    # =========================================================
    # Analyse de chaque bande
    # =========================================================

    print("\n" + "=" * 80)
    print("BANDES SUSPECTES")
    print("=" * 80)

    suspicious = []

    for b in range(cube.shape[0]):

        values = cube[b][footprint]

        finite = values[np.isfinite(values)]

        if finite.size == 0:
            print(
                f"Bande {b + 1:3d} : "
                f"100 % NaN"
            )
            continue

        pct_zero = (
            100 * np.mean(finite == 0)
        )

        minimum = np.min(finite)
        median = np.median(finite)
        mean = np.mean(finite)
        maximum = np.max(finite)

        # On affiche les bandes avec plus de 90 % de zéros
        if pct_zero >= 90:

            suspicious.append(b + 1)

            print(
                f"Bande {b + 1:3d} | "
                f"zéro={pct_zero:6.2f}% | "
                f"min={minimum:.8f} | "
                f"median={median:.8f} | "
                f"mean={mean:.8f} | "
                f"max={maximum:.8f}"
            )

    # =========================================================
    # Résumé
    # =========================================================

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    print(
        "Nombre total de bandes :",
        cube.shape[0],
    )

    print(
        "Nombre de bandes avec >= 90 % de zéros :",
        len(suspicious),
    )

    print(
        "Bandes concernées :",
        suspicious,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help=(
            "Chemin vers "
            "image_hyperspectrale_cleanbands.tif"
        ),
    )

    args = parser.parse_args()

    check_zero_bands(args.image)