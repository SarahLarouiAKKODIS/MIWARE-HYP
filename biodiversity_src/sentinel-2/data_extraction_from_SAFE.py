#!/usr/bin/env python3

from pathlib import Path
import argparse
import zipfile
import shutil
import sys
import time


# ============================================================
# Bandes à conserver
# ============================================================

FILES_TO_KEEP = {
    "10m": ["B02", "B03", "B04", "B08"],
    "20m": ["B05", "B06", "B07", "B8A", "B11", "B12", "SCL"],
    "60m": ["B01", "B09"],
}

EXPECTED_COUNT = sum(len(v) for v in FILES_TO_KEEP.values())  # 13


# ============================================================
# Construction des motifs recherchés
# ============================================================

def expected_patterns():

    patterns = []

    for resolution, bands in FILES_TO_KEEP.items():

        for band in bands:

            if band == "SCL":
                suffix = f"_SCL_{resolution}.jp2"
            else:
                suffix = f"_{band}_{resolution}.jp2"

            patterns.append(
                (resolution, band, suffix)
            )

    return patterns


# ============================================================
# Recherche des fichiers dans un .SAFE.zip
# ============================================================

def find_files_in_zip(zip_file):

    all_names = zip_file.namelist()

    selected = []
    missing = []

    for resolution, band, suffix in expected_patterns():

        matches = [
            name
            for name in all_names
            if name.endswith(suffix)
            and f"/R{resolution}/" in name
        ]

        if len(matches) == 0:

            missing.append(
                (band, resolution)
            )

        else:

            if len(matches) > 1:
                print(
                    f"    [ATTENTION] Plusieurs fichiers "
                    f"pour {band} {resolution}. "
                    "Le premier sera utilisé."
                )

            selected.append(
                (
                    resolution,
                    band,
                    matches[0]
                )
            )

    return selected, missing


# ============================================================
# Création du ZIP allégé
# ============================================================

def create_reduced_zip(
    input_zip,
    output_zip,
    selected
):

    output_zip.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(
        input_zip,
        mode="r"
    ) as zin:

        with zipfile.ZipFile(
            output_zip,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as zout:

            for resolution, band, internal_path in selected:

                filename = Path(internal_path).name

                # Organisation du ZIP final :
                #
                # R10m/
                # R20m/
                # R60m/

                archive_path = (
                    Path(f"R{resolution}")
                    / filename
                )

                print(
                    f"        {band:4s} -> {archive_path}"
                )

                # Copie en streaming :
                # aucun .SAFE complet n'est extrait sur disque

                with zin.open(
                    internal_path,
                    mode="r"
                ) as source:

                    with zout.open(
                        str(archive_path),
                        mode="w"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target,
                            length=1024 * 1024
                        )


# ============================================================
# Traitement d'un fichier
# ============================================================

def process_one_file(
    input_zip,
    output_dir,
    overwrite=False
):

    print("\n" + "=" * 70)
    print(f"Traitement : {input_zip.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Vérification ZIP
    # --------------------------------------------------------

    if not zipfile.is_zipfile(input_zip):

        print("    [ERREUR] Archive ZIP invalide.")

        return False

    # --------------------------------------------------------
    # Nom du fichier de sortie
    # --------------------------------------------------------

    name = input_zip.name

    if name.lower().endswith(".safe.zip"):
        name = name[:-9]

    elif name.lower().endswith(".zip"):
        name = name[:-4]

    output_zip = (
        output_dir
        / f"{name}_selected_bands.zip"
    )

    # --------------------------------------------------------
    # Si le fichier existe déjà
    # --------------------------------------------------------

    if output_zip.exists() and not overwrite:

        print(
            f"    [SKIP] Le fichier existe déjà : "
            f"{output_zip.name}"
        )

        return True

    # --------------------------------------------------------
    # Recherche des bandes
    # --------------------------------------------------------

    try:

        with zipfile.ZipFile(
            input_zip,
            mode="r"
        ) as zf:

            selected, missing = find_files_in_zip(
                zf
            )

    except Exception as e:

        print(
            f"    [ERREUR] Impossible de lire l'archive : {e}"
        )

        return False

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        f"\n    Fichiers trouvés : "
        f"{len(selected)}/{EXPECTED_COUNT}"
    )

    for resolution, band, _ in selected:

        print(
            f"        [OK] {band:4s} ({resolution})"
        )

    # --------------------------------------------------------
    # Vérification des fichiers manquants
    # --------------------------------------------------------

    if missing:

        print("\n    [ERREUR] Fichiers manquants :")

        for band, resolution in missing:

            print(
                f"        {band} ({resolution})"
            )

        print(
            "\n    Archive allégée NON créée "
            "pour éviter un produit incomplet."
        )

        return False

    # --------------------------------------------------------
    # Création du ZIP
    # --------------------------------------------------------

    print(
        f"\n    Création : {output_zip.name}\n"
    )

    try:

        create_reduced_zip(
            input_zip,
            output_zip,
            selected
        )

    except Exception as e:

        print(
            f"\n    [ERREUR] Création impossible : {e}"
        )

        # Suppression d'une archive partielle
        if output_zip.exists():
            output_zip.unlink()

        return False

    # --------------------------------------------------------
    # Taille
    # --------------------------------------------------------

    original_size = (
        input_zip.stat().st_size
        / (1024 ** 3)
    )

    reduced_size = (
        output_zip.stat().st_size
        / (1024 ** 3)
    )

    reduction = (
        100
        * (1 - reduced_size / original_size)
        if original_size > 0
        else 0
    )

    print("\n    [OK] Archive créée.")

    print(
        f"    Taille originale : {original_size:.2f} Go"
    )

    print(
        f"    Taille réduite   : {reduced_size:.2f} Go"
    )

    print(
        f"    Réduction        : {reduction:.1f} %"
    )

    return True


# ============================================================
# Programme principal
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extraction en batch des bandes utiles "
            "depuis plusieurs Sentinel-2 L2A .SAFE.zip."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Dossier contenant les fichiers "
            "Sentinel-2 .SAFE.zip"
        )
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier dans lequel créer "
            "les ZIP allégés"
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Écrase les ZIP de sortie "
            "s'ils existent déjà"
        )
    )

    args = parser.parse_args()

    input_dir = args.input_dir

    # --------------------------------------------------------
    # Vérification du dossier
    # --------------------------------------------------------

    if not input_dir.exists():

        print(
            f"Erreur : dossier introuvable : "
            f"{input_dir}"
        )

        sys.exit(1)

    if not input_dir.is_dir():

        print(
            f"Erreur : ce chemin n'est pas "
            f"un dossier : {input_dir}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Dossier de sortie
    # --------------------------------------------------------

    if args.output_dir is None:

        output_dir = (
            input_dir.parent
            / f"{input_dir.name}_reduced"
        )

    else:

        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Recherche des .SAFE.zip
    # --------------------------------------------------------

    safe_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file()
            and p.name.lower().endswith(".safe.zip")
        ]
    )

    if not safe_files:

        print(
            "\nAucun fichier .SAFE.zip trouvé dans :"
        )

        print(input_dir)

        sys.exit(1)

    print("\n========================================")
    print("Sentinel-2 SAFE extraction")
    print("========================================")

    print(
        f"\nDossier source : {input_dir}"
    )

    print(
        f"Dossier sortie : {output_dir}"
    )

    print(
        f"Nombre d'archives : {len(safe_files)}"
    )

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    start_time = time.time()

    success = 0
    failed = 0

    for i, input_zip in enumerate(
        safe_files,
        start=1
    ):

        print(
            f"\n\n######## IMAGE {i}/{len(safe_files)} ########"
        )

        result = process_one_file(
            input_zip,
            output_dir,
            overwrite=args.overwrite
        )

        if result:
            success += 1
        else:
            failed += 1

    # --------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)

    print(
        f"Archives détectées : {len(safe_files)}"
    )

    print(
        f"Succès              : {success}"
    )

    print(
        f"Échecs              : {failed}"
    )

    print(
        f"Temps total         : {elapsed / 60:.1f} min"
    )

    print(
        f"\nRésultats : {output_dir}"
    )


if __name__ == "__main__":
    main()