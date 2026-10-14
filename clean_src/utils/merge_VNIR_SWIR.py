#!/usr/bin/env python3

"""
merge_vnir_swir.py

1. Recherche automatiquement :
       *VNIR.TIF
       *SWIR.TIF

2. Vérifie les deux images AVANT fusion :
       - pixels NODATA
       - pixels à 0
       - bandes contenant des 0
       - bandes entièrement à 0

3. Si des pixels à 0 sont détectés dans l'emprise valide :
       -> la fusion est arrêtée

4. Sinon :
       VNIR + SWIR -> *SPECTRAL_IMAGE.TIF

Usage :
    python3 merge_vnir_swir.py /chemin/vers/dossier/
"""

from pathlib import Path
import argparse

import numpy as np
import rasterio


# ============================================================
# PARAMÈTRES
# ============================================================

VNIR_SUFFIX = "SPECTRAL_IMAGE_VNIR.TIF"
SWIR_SUFFIX = "SPECTRAL_IMAGE_SWIR.TIF"


# ============================================================
# RECHERCHE DES FICHIERS
# ============================================================

def find_image(folder: Path, suffix: str) -> Path:

    images = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.name.upper().endswith(suffix.upper())
    )

    if len(images) == 0:
        raise FileNotFoundError(
            f"Aucun fichier se terminant par '{suffix}' "
            f"trouvé dans : {folder}"
        )

    if len(images) > 1:

        print(
            f"\nPlusieurs fichiers se terminant par "
            f"'{suffix}' ont été trouvés :"
        )

        for image in images:
            print(f"  - {image}")

        raise RuntimeError(
            f"Impossible de choisir automatiquement {suffix}."
        )

    return images[0]


# ============================================================
# CONTRÔLE DES ZÉROS
# ============================================================

def check_zero_pixels(image_path: Path):

    """
    Vérifie la présence de valeurs 0 dans chaque bande.

    Les pixels NODATA sont exclus de l'analyse.

    Retourne :
        True  -> image OK, aucun zéro détecté
        False -> au moins un zéro détecté
    """

    print()
    print("=" * 100)
    print(f"CONTRÔLE : {image_path.name}")
    print("=" * 100)

    bands_with_zero = []
    fully_zero_bands = []

    total_zero_pixels = 0
    total_valid_values = 0

    with rasterio.open(image_path) as src:

        nodata = src.nodata

        print(f"Nombre de bandes : {src.count}")
        print(f"Dimensions       : {src.width} × {src.height}")
        print(f"dtype            : {src.dtypes[0]}")
        print(f"nodata           : {nodata}")
        print(f"CRS              : {src.crs}")

        # ====================================================
        # ANALYSE BANDE PAR BANDE
        # ====================================================

        for band_index in range(1, src.count + 1):

            band = src.read(band_index)

            # -----------------------------------------------
            # Pixels valides
            # -----------------------------------------------

            if nodata is not None:

                if np.isnan(nodata):
                    valid = np.isfinite(band)

                else:
                    valid = (
                        np.isfinite(band)
                        & (band != nodata)
                    )

            else:
                valid = np.isfinite(band)

            n_valid = int(np.sum(valid))

            # Bande sans donnée
            if n_valid == 0:

                print(
                    f"Bande {band_index:3d} | "
                    f"NODATA"
                )

                continue

            values = band[valid]

            # -----------------------------------------------
            # Zéros
            # -----------------------------------------------

            zero_mask = values == 0

            n_zero = int(np.sum(zero_mask))

            zero_percent = (
                100.0 * n_zero / n_valid
            )

            total_zero_pixels += n_zero
            total_valid_values += n_valid

            # -----------------------------------------------
            # Bande contenant au moins un zéro
            # -----------------------------------------------

            if n_zero > 0:

                bands_with_zero.append(
                    band_index
                )

            # -----------------------------------------------
            # Bande entièrement à zéro
            # -----------------------------------------------

            if n_zero == n_valid:

                fully_zero_bands.append(
                    band_index
                )

            # -----------------------------------------------
            # Affichage
            # -----------------------------------------------

            status = "OK"

            if n_zero == n_valid:
                status = "❌ ENTIÈREMENT À 0"

            elif n_zero > 0:
                status = "⚠️ ZÉROS DÉTECTÉS"

            print(
                f"Bande {band_index:3d} | "
                f"valides={n_valid:8d} | "
                f"zeros={n_zero:8d} | "
                f"{zero_percent:8.4f}% | "
                f"{status}"
            )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    print()
    print("-" * 100)
    print("RÉSUMÉ")
    print("-" * 100)

    print(
        f"Bandes contenant des pixels à 0 : "
        f"{len(bands_with_zero)}"
    )

    print(
        f"Bandes entièrement à 0           : "
        f"{len(fully_zero_bands)}"
    )

    print(
        f"Nombre total de valeurs à 0      : "
        f"{total_zero_pixels}"
    )

    if total_valid_values > 0:

        global_zero_percent = (
            100.0
            * total_zero_pixels
            / total_valid_values
        )

    else:
        global_zero_percent = 0.0

    print(
        f"Pourcentage global de zéros      : "
        f"{global_zero_percent:.6f}%"
    )

    if bands_with_zero:

        print()
        print(
            "Bandes contenant au moins un zéro :"
        )

        print(
            bands_with_zero
        )

    if fully_zero_bands:

        print()
        print(
            "Bandes entièrement à zéro :"
        )

        print(
            fully_zero_bands
        )

    # ========================================================
    # VERDICT
    # ========================================================

    print()

    if len(bands_with_zero) == 0:

        print(
            f"✅ {image_path.name} : "
            f"aucun pixel valide à 0."
        )

        return True

    else:

        print(
            f"❌ {image_path.name} : "
            f"des pixels à 0 ont été détectés."
        )

        return False


# ============================================================
# FUSION VNIR + SWIR
# ============================================================

def merge_vnir_swir(folder: Path):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Le chemin n'est pas un dossier : {folder}"
        )

    # ========================================================
    # RECHERCHE
    # ========================================================

    vnir_path = find_image(
        folder,
        VNIR_SUFFIX,
    )

    swir_path = find_image(
        folder,
        SWIR_SUFFIX,
    )

    print("=" * 100)
    print("FICHIERS TROUVÉS")
    print("=" * 100)

    print(f"VNIR : {vnir_path}")
    print(f"SWIR : {swir_path}")

    # ========================================================
    # CONTRÔLE AVANT FUSION
    # ========================================================

    vnir_ok = check_zero_pixels(
        vnir_path
    )

    swir_ok = check_zero_pixels(
        swir_path
    )

    print()
    print("=" * 100)
    print("CONTRÔLE AVANT FUSION")
    print("=" * 100)

    print(
        f"VNIR : "
        f"{'✅ OK' if vnir_ok else '❌ PROBLÈME'}"
    )

    print(
        f"SWIR : "
        f"{'✅ OK' if swir_ok else '❌ PROBLÈME'}"
    )

    # ========================================================
    # ARRÊT SI ZÉROS
    # ========================================================

    if not vnir_ok or not swir_ok:

        print()
        print("❌ FUSION ANNULÉE")

        print(
            "Au moins une des deux images contient "
            "des pixels valides ayant la valeur 0."
        )

        print(
            "Le SPECTRAL_IMAGE.TIF ne sera pas créé."
        )

        return None

    # ========================================================
    # OUVERTURE POUR FUSION
    # ========================================================

    with rasterio.open(vnir_path) as vnir_src, \
         rasterio.open(swir_path) as swir_src:

        # ====================================================
        # VÉRIFICATIONS GÉOMÉTRIQUES
        # ====================================================

        if (
            vnir_src.width != swir_src.width
            or vnir_src.height != swir_src.height
        ):

            raise ValueError(
                "VNIR et SWIR n'ont pas les mêmes dimensions.\n"
                f"VNIR : "
                f"{vnir_src.width} × {vnir_src.height}\n"
                f"SWIR : "
                f"{swir_src.width} × {swir_src.height}"
            )

        if vnir_src.crs != swir_src.crs:

            raise ValueError(
                "VNIR et SWIR n'ont pas le même CRS.\n"
                f"VNIR : {vnir_src.crs}\n"
                f"SWIR : {swir_src.crs}"
            )

        if not vnir_src.transform.almost_equals(
            swir_src.transform
        ):

            raise ValueError(
                "VNIR et SWIR n'ont pas exactement "
                "la même grille géographique."
            )

        # ====================================================
        # TYPE
        # ====================================================

        vnir_dtype = np.dtype(
            vnir_src.dtypes[0]
        )

        swir_dtype = np.dtype(
            swir_src.dtypes[0]
        )

        output_dtype = np.result_type(
            vnir_dtype,
            swir_dtype,
        )

        # ====================================================
        # NODATA
        # ====================================================

        if vnir_src.nodata != swir_src.nodata:

            raise ValueError(
                "VNIR et SWIR n'ont pas la même "
                "valeur NODATA.\n"
                f"VNIR : {vnir_src.nodata}\n"
                f"SWIR : {swir_src.nodata}"
            )

        output_nodata = vnir_src.nodata

        # ====================================================
        # BANDES
        # ====================================================

        n_vnir = vnir_src.count
        n_swir = swir_src.count

        total_bands = (
            n_vnir + n_swir
        )

        print()
        print("=" * 100)
        print("FUSION")
        print("=" * 100)

        print(
            f"Bandes VNIR : {n_vnir}"
        )

        print(
            f"Bandes SWIR : {n_swir}"
        )

        print(
            f"Total       : {total_bands}"
        )

        # ====================================================
        # NOM DE SORTIE
        # ====================================================

        name = vnir_path.name

        if name.upper().endswith(
            VNIR_SUFFIX
        ):

            prefix = name[
                :-len(VNIR_SUFFIX)
            ]

        else:

            prefix = ""

        output_path = folder / (
            f"{prefix}SPECTRAL_IMAGE.TIF"
        )

        # ====================================================
        # PROFIL
        # ====================================================

        profile = vnir_src.profile.copy()

        profile.update(
            count=total_bands,
            dtype=output_dtype.name,
            nodata=output_nodata,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        # ====================================================
        # ÉCRITURE
        # ====================================================

        print()
        print(
            f"Création : {output_path}"
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            # -----------------------------------------------
            # VNIR
            # -----------------------------------------------

            print("Écriture VNIR...")

            for band_index in range(
                1,
                n_vnir + 1,
            ):

                data = vnir_src.read(
                    band_index
                )

                dst.write(
                    data.astype(
                        output_dtype,
                        copy=False,
                    ),
                    band_index,
                )

                description = (
                    vnir_src.descriptions[
                        band_index - 1
                    ]
                )

                if description:

                    dst.set_band_description(
                        band_index,
                        description,
                    )

            # -----------------------------------------------
            # SWIR
            # -----------------------------------------------

            print("Écriture SWIR...")

            for band_index in range(
                1,
                n_swir + 1,
            ):

                data = swir_src.read(
                    band_index
                )

                output_band = (
                    n_vnir
                    + band_index
                )

                dst.write(
                    data.astype(
                        output_dtype,
                        copy=False,
                    ),
                    output_band,
                )

                description = (
                    swir_src.descriptions[
                        band_index - 1
                    ]
                )

                if description:

                    dst.set_band_description(
                        output_band,
                        description,
                    )

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 100)
    print("IMAGE HYPERSPECTRALE CRÉÉE")
    print("=" * 100)

    print(
        f"Fichier : {output_path}"
    )

    print(
        f"Bandes : "
        f"{n_vnir} VNIR + "
        f"{n_swir} SWIR = "
        f"{total_bands}"
    )

    print()
    print("✅ Fusion terminée.")

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Contrôle puis fusion des images "
            "VNIR et SWIR."
        )
    )

    parser.add_argument(
        "folder",
        help=(
            "Dossier contenant "
            "*VNIR.TIF et *SWIR.TIF"
        ),
    )

    args = parser.parse_args()

    merge_vnir_swir(
        Path(args.folder)
    )