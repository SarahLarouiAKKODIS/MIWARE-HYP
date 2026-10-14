#!/usr/bin/env python3

"""
check_all_hyperspectral_images.py

Analyse automatiquement toutes les images hyperspectrales référencées
dans un ensemble de fichiers de configuration JSON.

Chaque JSON doit contenir :
    "Path_data": "/chemin/vers/donnees"
    "image_hyp": "nom_image.tif"

Le chemin de l'image est construit avec :
    Path_data / image_hyp

Usage
-----
python check_all_hyperspectral_images.py /path/to/configs/

Le dossier peut contenir directement les JSON ou des JSON dans
des sous-dossiers.

Sorties
-------
hyperspectral_quality_report.csv

Classification :
    OK
    SUSPECTE
    CORROMPUE
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

# Une bande est considérée comme quasi nulle si au moins
# 99 % des pixels de l'emprise sont à zéro.
ZERO_BAND_THRESHOLD = 99.0

# Nombre de bandes entièrement/quasi entièrement nulles
# au-delà duquel l'image est considérée comme corrompue.
CORRUPTED_ZERO_BANDS_THRESHOLD = 20

# Entre 1 et ce seuil -> image suspecte.
SUSPICIOUS_ZERO_BANDS_THRESHOLD = 1

# Nombre minimum de pixels contenant réellement des données.
# Sert à signaler les images ayant une emprise anormalement petite.
MIN_DATA_PIXELS_WARNING = 10000


# ============================================================
# LECTURE CONFIG
# ============================================================

def read_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    if "Path_data" not in config:
        raise KeyError(
            f"{config_path.name} : clé 'Path_data' absente."
        )

    if "image_hyp" not in config:
        raise KeyError(
            f"{config_path.name} : clé 'image_hyp' absente."
        )

    path_data = Path(
        config["Path_data"]
    )

    image_hyp = Path(
        config["image_hyp"]
    )

    # Si image_hyp est déjà un chemin absolu
    if image_hyp.is_absolute():
        image_path = image_hyp

    else:
        image_path = path_data / image_hyp

    return image_path


# ============================================================
# ANALYSE D'UNE IMAGE
# ============================================================

def analyse_image(
    image_path,
    config_path,
):

    print("\n")
    print("=" * 100)
    print("CONFIG :", config_path.name)
    print("=" * 100)

    print("Image :", image_path)

    # --------------------------------------------------------
    # Image absente
    # --------------------------------------------------------

    if not image_path.exists():

        print("[ERREUR] Image introuvable.")

        return {
            "config": config_path.name,
            "image": image_path.name,
            "image_path": str(image_path),
            "status": "CORROMPUE",
            "reason": "Image introuvable",
        }

    # --------------------------------------------------------
    # Lecture raster
    # --------------------------------------------------------

    try:

        with rasterio.open(image_path) as src:

            cube = src.read(
                masked=False
            )

            nodata = src.nodata

            bands = src.count
            rows = src.height
            cols = src.width

            crs = str(src.crs)

    except Exception as exc:

        print(
            "[ERREUR] Impossible de lire l'image :",
            exc,
        )

        return {
            "config": config_path.name,
            "image": image_path.name,
            "image_path": str(image_path),
            "status": "CORROMPUE",
            "reason": f"Lecture impossible : {exc}",
        }

    print(
        f"Shape : "
        f"{bands} bandes × {rows} × {cols}"
    )

    print(
        "dtype :",
        cube.dtype,
    )

    print(
        "nodata :",
        nodata,
    )

    # --------------------------------------------------------
    # Float
    # --------------------------------------------------------

    cube = cube.astype(
        np.float32
    )

    # --------------------------------------------------------
    # NODATA -> NaN
    # --------------------------------------------------------

    if nodata is not None:

        if np.isnan(nodata):
            pass

        else:
            cube[
                cube == nodata
            ] = np.nan

    cube[
        ~np.isfinite(cube)
    ] = np.nan

    # ========================================================
    # EMPRISE RÉELLE
    # ========================================================

    # Un pixel appartient à l'emprise s'il possède au moins
    # une valeur finie ET différente de zéro.

    footprint = np.any(
        np.isfinite(cube)
        & (cube != 0),
        axis=0,
    )

    n_data_pixels = int(
        footprint.sum()
    )

    total_pixels = (
        rows * cols
    )

    footprint_percent = (
        100
        * n_data_pixels
        / total_pixels
    )

    print(
        "Pixels contenant des données :",
        n_data_pixels,
        f"({footprint_percent:.2f} %)",
    )

    # --------------------------------------------------------
    # Aucun signal
    # --------------------------------------------------------

    if n_data_pixels == 0:

        return {
            "config": config_path.name,
            "image": image_path.name,
            "image_path": str(image_path),
            "status": "CORROMPUE",
            "reason": "Aucun pixel contenant du signal",
            "bands": bands,
            "rows": rows,
            "cols": cols,
            "data_pixels": 0,
            "data_percent": 0,
        }

    # ========================================================
    # ANALYSE DES BANDES
    # ========================================================

    fully_zero_bands = []

    quasi_zero_bands = []

    fully_nodata_bands = []

    bands_with_signal = []

    band_records = []

    for b in range(bands):

        values = cube[b][footprint]

        finite_mask = np.isfinite(
            values
        )

        n_valid = int(
            finite_mask.sum()
        )

        # ----------------------------------------------------
        # Bande entièrement nodata
        # ----------------------------------------------------

        if n_valid == 0:

            fully_nodata_bands.append(
                b + 1
            )

            band_records.append(
                {
                    "band": b + 1,
                    "valid_pixels": 0,
                    "zero_percent": np.nan,
                    "status": "NODATA",
                }
            )

            continue

        x = values[
            finite_mask
        ]

        # ----------------------------------------------------
        # Pourcentage de zéro
        # ----------------------------------------------------

        zero_percent = float(
            100
            * np.mean(x == 0)
        )

        # ----------------------------------------------------
        # Bande entièrement à zéro
        # ----------------------------------------------------

        if np.all(x == 0):

            fully_zero_bands.append(
                b + 1
            )

        # ----------------------------------------------------
        # Bande quasi nulle
        # ----------------------------------------------------

        if (
            zero_percent
            >= ZERO_BAND_THRESHOLD
        ):

            quasi_zero_bands.append(
                b + 1
            )

        # ----------------------------------------------------
        # Signal réel
        # ----------------------------------------------------

        if np.any(x != 0):

            bands_with_signal.append(
                b + 1
            )

        band_records.append(
            {
                "band": b + 1,
                "valid_pixels": n_valid,
                "zero_percent": zero_percent,
                "min": float(
                    np.min(x)
                ),
                "median": float(
                    np.median(x)
                ),
                "max": float(
                    np.max(x)
                ),
                "status": (
                    "ZERO"
                    if np.all(x == 0)
                    else "OK"
                ),
            }
        )

    # ========================================================
    # NOMBRE DE BANDES PROBLÉMATIQUES
    # ========================================================

    n_fully_zero = len(
        fully_zero_bands
    )

    n_quasi_zero = len(
        quasi_zero_bands
    )

    n_fully_nodata = len(
        fully_nodata_bands
    )

    n_signal = len(
        bands_with_signal
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    reasons = []

    # --------------------------------------------------------
    # Cas clairement corrompu / incomplet
    # --------------------------------------------------------

    if (
        n_quasi_zero
        >= CORRUPTED_ZERO_BANDS_THRESHOLD
    ):

        status = "CORROMPUE"

        reasons.append(
            f"{n_quasi_zero} bandes "
            f">= {ZERO_BAND_THRESHOLD:.0f}% de zéros"
        )

    # --------------------------------------------------------
    # Quelques bandes suspectes
    # --------------------------------------------------------

    elif (
        n_quasi_zero
        >= SUSPICIOUS_ZERO_BANDS_THRESHOLD
    ):

        status = "SUSPECTE"

        reasons.append(
            f"{n_quasi_zero} bande(s) "
            f">= {ZERO_BAND_THRESHOLD:.0f}% de zéros"
        )

    else:

        status = "OK"

    # --------------------------------------------------------
    # Emprise très petite
    #
    # ATTENTION :
    # ce critère seul ne suffit pas à déclarer une image
    # corrompue.
    # --------------------------------------------------------

    if (
        n_data_pixels
        < MIN_DATA_PIXELS_WARNING
    ):

        reasons.append(
            f"emprise très faible "
            f"({n_data_pixels} pixels)"
        )

        if status == "OK":
            status = "SUSPECTE"

    # --------------------------------------------------------
    # Aucune bande avec signal
    # --------------------------------------------------------

    if n_signal == 0:

        status = "CORROMPUE"

        reasons.append(
            "aucune bande contenant du signal"
        )

    # --------------------------------------------------------
    # Texte raison
    # --------------------------------------------------------

    if not reasons:

        reason = "Aucune anomalie majeure détectée"

    else:

        reason = " ; ".join(
            reasons
        )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    print("\nRésultat :")

    print(
        "  Statut :",
        status,
    )

    print(
        "  Bandes avec signal :",
        f"{n_signal}/{bands}",
    )

    print(
        "  Bandes totalement nodata :",
        n_fully_nodata,
    )

    print(
        "  Bandes totalement à zéro :",
        n_fully_zero,
    )

    print(
        f"  Bandes >= "
        f"{ZERO_BAND_THRESHOLD:.0f}% zéro :",
        n_quasi_zero,
    )

    if fully_zero_bands:

        print(
            "  Bandes totalement à zéro :",
            fully_zero_bands,
        )

    if fully_nodata_bands:

        print(
            "  Bandes totalement nodata :",
            fully_nodata_bands,
        )

    print(
        "  Raison :",
        reason,
    )

    # ========================================================
    # RÉSULTAT
    # ========================================================

    return {
        "config": config_path.name,
        "image": image_path.name,
        "image_path": str(
            image_path
        ),
        "status": status,
        "reason": reason,
        "bands": bands,
        "rows": rows,
        "cols": cols,
        "data_pixels": n_data_pixels,
        "data_percent": footprint_percent,
        "bands_with_signal": n_signal,
        "fully_nodata_bands": n_fully_nodata,
        "fully_zero_bands": n_fully_zero,
        "quasi_zero_bands": n_quasi_zero,
        "fully_nodata_band_ids": ",".join(
            map(
                str,
                fully_nodata_bands,
            )
        ),
        "fully_zero_band_ids": ",".join(
            map(
                str,
                fully_zero_bands,
            )
        ),
        "crs": crs,
    }


# ============================================================
# ANALYSE DES CONFIGS
# ============================================================

def analyse_configs(
    configs_path,
):

    configs_path = Path(
        configs_path
    )

    if not configs_path.exists():

        raise FileNotFoundError(
            f"Chemin introuvable : "
            f"{configs_path}"
        )

    # ========================================================
    # RECHERCHE JSON
    # ========================================================

    if configs_path.is_file():

        if (
            configs_path.suffix.lower()
            != ".json"
        ):

            raise ValueError(
                "Le fichier fourni n'est pas un JSON."
            )

        config_files = [
            configs_path
        ]

        output_dir = (
            configs_path.parent
        )

    else:

        config_files = sorted(
            configs_path.rglob(
                "*.json"
            )
        )

        output_dir = (
            configs_path
        )

    if not config_files:

        raise FileNotFoundError(
            "Aucun fichier JSON trouvé."
        )

    print("=" * 100)
    print("CONTRÔLE QUALITÉ DES IMAGES HYPERSPECTRALES")
    print("=" * 100)

    print(
        "Nombre de configs trouvées :",
        len(config_files),
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    results = []

    for i, config_path in enumerate(
        config_files,
        start=1,
    ):

        print(
            f"\n[{i}/{len(config_files)}]"
        )

        try:

            image_path = read_config(
                config_path
            )

            result = analyse_image(
                image_path=image_path,
                config_path=config_path,
            )

        except Exception as exc:

            print(
                "[ERREUR CONFIG]",
                exc,
            )

            result = {
                "config": config_path.name,
                "image": "",
                "image_path": "",
                "status": "CORROMPUE",
                "reason": (
                    f"Erreur config : {exc}"
                ),
            }

        results.append(
            result
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Ordre : corrompues en premier
    # --------------------------------------------------------

    status_order = {
        "CORROMPUE": 0,
        "SUSPECTE": 1,
        "OK": 2,
    }

    df["_order"] = (
        df["status"]
        .map(status_order)
        .fillna(99)
    )

    df = (
        df
        .sort_values(
            [
                "_order",
                "config",
            ]
        )
        .drop(
            columns="_order"
        )
    )

    # ========================================================
    # CSV
    # ========================================================

    output_csv = (
        output_dir
        / "hyperspectral_quality_report.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ========================================================
    # RÉSUMÉ FINAL
    # ========================================================

    n_total = len(df)

    n_ok = int(
        (df["status"] == "OK").sum()
    )

    n_suspicious = int(
        (df["status"] == "SUSPECTE").sum()
    )

    n_corrupted = int(
        (df["status"] == "CORROMPUE").sum()
    )

    print("\n")
    print("=" * 100)
    print("RÉSUMÉ FINAL")
    print("=" * 100)

    print(
        f"Images analysées : {n_total}"
    )

    print(
        f"OK          : {n_ok}"
    )

    print(
        f"SUSPECTES   : {n_suspicious}"
    )

    print(
        f"CORROMPUES  : {n_corrupted}"
    )

    # ========================================================
    # LISTE CORROMPUES
    # ========================================================

    corrupted = df[
        df["status"]
        == "CORROMPUE"
    ]

    if not corrupted.empty:

        print("\n" + "=" * 100)
        print("IMAGES CORROMPUES / INCOMPLÈTES")
        print("=" * 100)

        for _, row in corrupted.iterrows():

            print(
                f"\n{row['image']}"
            )

            print(
                "  Config :",
                row["config"],
            )

            print(
                "  Raison :",
                row["reason"],
            )

    # ========================================================
    # LISTE SUSPECTES
    # ========================================================

    suspicious = df[
        df["status"]
        == "SUSPECTE"
    ]

    if not suspicious.empty:

        print("\n" + "=" * 100)
        print("IMAGES SUSPECTES")
        print("=" * 100)

        for _, row in suspicious.iterrows():

            print(
                f"\n{row['image']}"
            )

            print(
                "  Config :",
                row["config"],
            )

            print(
                "  Raison :",
                row["reason"],
            )

    # ========================================================
    # SORTIE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("RAPPORT")
    print("=" * 100)

    print(
        output_csv
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analyse toutes les images hyperspectrales "
            "référencées dans des fichiers JSON."
        )
    )

    parser.add_argument(
        "configs",
        help=(
            "Chemin vers un fichier JSON ou un dossier "
            "contenant les configs JSON."
        ),
    )

    args = parser.parse_args()

    analyse_configs(
        args.configs
    )