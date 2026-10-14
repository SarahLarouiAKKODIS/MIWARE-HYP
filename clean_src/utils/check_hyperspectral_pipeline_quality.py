#!/usr/bin/env python3

"""
check_hyperspectral_quality.py

Diagnostic des différentes étapes du prétraitement hyperspectral.

Usage
-----
python check_hyperspectral_quality.py \
    /path/image_L2A.tif \
    /path/dossier_intermediaires/

Images intermédiaires recherchées automatiquement :
    image_hyperspectrale_reflectance.tif
    image_hyperspectrale_clean.tif
    image_hyperspectrale_cleanbands.tif
    image_hyperspectrale_cleanbands_smooth.tif

Le script recherche :
    - NaN / nodata
    - bandes entièrement ou presque entièrement nulles
    - valeurs négatives
    - valeurs > 1.2
    - bandes constantes
    - apparition/disparition de signal entre les étapes
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "reflectance": "image_hyperspectrale_reflectance.tif",
    "clean": "image_hyperspectrale_clean.tif",
    "cleanbands": "image_hyperspectrale_cleanbands.tif",
    "smooth": "image_hyperspectrale_cleanbands_smooth.tif",
}

OUTPUT_DIR_NAME = "quality_comparison"

# Une bande est considérée comme quasi nulle si au moins
# ce pourcentage de ses pixels sont exactement égaux à zéro.
ZERO_WARNING_PERCENT = 90.0


# ============================================================
# RECHERCHE DES FICHIERS
# ============================================================

def find_intermediate_files(folder):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {folder}"
        )

    found = {}

    for stage, filename in FILES.items():

        matches = list(folder.rglob(filename))

        if len(matches) == 1:

            found[stage] = matches[0]

        elif len(matches) > 1:

            print(
                f"[WARNING] Plusieurs fichiers trouvés pour "
                f"{filename} :"
            )

            for p in matches:
                print("   ", p)

            found[stage] = matches[0]

            print(
                "   -> premier fichier utilisé :",
                matches[0],
            )

        else:

            print(
                f"[WARNING] Fichier non trouvé : {filename}"
            )

    return found


# ============================================================
# LECTURE
# ============================================================

def read_cube(path):

    with rasterio.open(path) as src:

        cube = src.read(masked=False).astype(np.float32)

        nodata = src.nodata
        crs = src.crs
        transform = src.transform

    # --------------------------------------------------------
    # Nodata -> NaN
    # --------------------------------------------------------

    if nodata is not None:

        if np.isnan(nodata):
            pass
        else:
            cube[cube == nodata] = np.nan

    cube[~np.isfinite(cube)] = np.nan

    return cube, nodata, crs, transform


# ============================================================
# DÉTERMINATION DE L'EMPRISE
# ============================================================

def get_footprint(cube):

    """
    Un pixel appartient à l'emprise réelle s'il existe au moins
    une bande finie ET différente de zéro.

    Cela évite que des zones de fond codées 0 soient considérées
    comme des données valides.
    """

    return np.any(
        np.isfinite(cube) & (cube != 0),
        axis=0,
    )


# ============================================================
# ANALYSE D'UNE IMAGE
# ============================================================

def analyse_image(stage, path):

    print("\n")
    print("=" * 100)
    print(f"ÉTAPE : {stage}")
    print("=" * 100)

    print("Fichier :", path)

    cube, nodata, crs, transform = read_cube(path)

    bands, rows, cols = cube.shape

    print(
        f"Shape : {cube.shape} "
        f"({bands} bandes, {rows} x {cols})"
    )

    print("dtype analysé : float32")
    print("nodata :", nodata)
    print("CRS :", crs)

    # ========================================================
    # EMPRISE RÉELLE
    # ========================================================

    footprint = get_footprint(cube)

    n_footprint = int(
        footprint.sum()
    )

    print(
        "Pixels dans l'emprise réelle :",
        n_footprint,
    )

    if n_footprint == 0:

        print(
            "[ERREUR] Aucun pixel contenant du signal."
        )

        return None, None

    # ========================================================
    # COMPLÉTUDE SPECTRALE
    # ========================================================

    valid_count = np.isfinite(cube).sum(axis=0)

    counts = valid_count[footprint]

    min_valid = int(counts.min())
    max_valid = int(counts.max())

    median_valid = float(
        np.median(counts)
    )

    mean_valid = float(
        np.mean(counts)
    )

    complete_pixels = int(
        np.sum(counts == bands)
    )

    complete_percent = (
        100
        * complete_pixels
        / n_footprint
    )

    print("\nComplétude spectrale")

    print(
        f"  Minimum : {min_valid}/{bands}"
    )

    print(
        f"  Médiane : {median_valid:.1f}/{bands}"
    )

    print(
        f"  Moyenne : {mean_valid:.2f}/{bands}"
    )

    print(
        f"  Maximum : {max_valid}/{bands}"
    )

    print(
        f"  Pixels complets : "
        f"{complete_pixels}/{n_footprint} "
        f"({complete_percent:.2f} %)"
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print("\nDistribution :")

    unique, numbers = np.unique(
        counts,
        return_counts=True,
    )

    for n_bands, n_pixels in zip(
        unique,
        numbers,
    ):

        percentage = (
            100
            * n_pixels
            / n_footprint
        )

        print(
            f"  {int(n_bands):3d}/{bands} bandes : "
            f"{n_pixels:8d} pixels "
            f"({percentage:6.2f} %)"
        )

    # ========================================================
    # ANALYSE BANDE PAR BANDE
    # ========================================================

    records = []

    for b in range(bands):

        values = cube[b][footprint]

        finite_mask = np.isfinite(values)

        n_valid = int(
            finite_mask.sum()
        )

        n_nan = (
            n_footprint
            - n_valid
        )

        nan_percent = (
            100
            * n_nan
            / n_footprint
        )

        if n_valid > 0:

            x = values[finite_mask]

            minimum = float(np.min(x))
            maximum = float(np.max(x))
            mean = float(np.mean(x))
            median = float(np.median(x))
            std = float(np.std(x))

            p01 = float(np.percentile(x, 1))
            p05 = float(np.percentile(x, 5))
            p95 = float(np.percentile(x, 95))
            p99 = float(np.percentile(x, 99))

            zero_percent = float(
                100 * np.mean(x == 0)
            )

            negative_percent = float(
                100 * np.mean(x < 0)
            )

            above_1_percent = float(
                100 * np.mean(x > 1.0)
            )

            above_1_2_percent = float(
                100 * np.mean(x > 1.2)
            )

            all_zero = bool(
                np.all(x == 0)
            )

            constant = bool(
                np.all(x == x[0])
            )

            has_signal = bool(
                np.any(x != 0)
            )

        else:

            minimum = np.nan
            maximum = np.nan
            mean = np.nan
            median = np.nan
            std = np.nan

            p01 = np.nan
            p05 = np.nan
            p95 = np.nan
            p99 = np.nan

            zero_percent = np.nan
            negative_percent = np.nan
            above_1_percent = np.nan
            above_1_2_percent = np.nan

            all_zero = False
            constant = False
            has_signal = False

        records.append(
            {
                "stage": stage,
                "band_id": b + 1,
                "band_index": b,
                "valid_pixels": n_valid,
                "nan_pixels": n_nan,
                "nan_percent": nan_percent,
                "zero_percent": zero_percent,
                "negative_percent": negative_percent,
                "above_1_percent": above_1_percent,
                "above_1.2_percent": above_1_2_percent,
                "min": minimum,
                "p01": p01,
                "p05": p05,
                "median": median,
                "mean": mean,
                "p95": p95,
                "p99": p99,
                "max": maximum,
                "std": std,
                "all_zero": all_zero,
                "constant": constant,
                "has_signal": has_signal,
            }
        )

    band_df = pd.DataFrame(records)

    # ========================================================
    # NaN
    # ========================================================

    problematic_nan = band_df[
        band_df["nan_percent"] > 0
    ]

    print("\n" + "-" * 100)
    print("BANDES CONTENANT DES NaN / NODATA")
    print("-" * 100)

    if problematic_nan.empty:

        print("Aucune")

    else:

        for _, row in problematic_nan.iterrows():

            print(
                f"Bande {int(row['band_id']):3d} | "
                f"NaN={row['nan_percent']:6.2f}%"
            )

    # ========================================================
    # BANDES NULLES
    # ========================================================

    zero_bands = band_df[
        band_df["zero_percent"]
        >= ZERO_WARNING_PERCENT
    ]

    print("\n" + "-" * 100)
    print(
        f"BANDES AVEC >= "
        f"{ZERO_WARNING_PERCENT:.0f}% DE ZÉROS"
    )
    print("-" * 100)

    if zero_bands.empty:

        print("Aucune")

    else:

        for _, row in zero_bands.iterrows():

            print(
                f"Bande {int(row['band_id']):3d} | "
                f"zéro={row['zero_percent']:6.2f}% | "
                f"min={row['min']:.8f} | "
                f"median={row['median']:.8f} | "
                f"max={row['max']:.8f}"
            )

    # ========================================================
    # BANDES ENTIÈREMENT À ZÉRO
    # ========================================================

    fully_zero = band_df[
        band_df["all_zero"]
    ]

    print("\n" + "-" * 100)
    print("BANDES ENTIÈREMENT À ZÉRO")
    print("-" * 100)

    if fully_zero.empty:

        print("Aucune")

    else:

        print(
            fully_zero["band_id"]
            .astype(int)
            .tolist()
        )

    # ========================================================
    # VALEURS NÉGATIVES
    # ========================================================

    negative_bands = band_df[
        band_df["negative_percent"] > 0
    ]

    print("\n" + "-" * 100)
    print("BANDES CONTENANT DES VALEURS NÉGATIVES")
    print("-" * 100)

    if negative_bands.empty:

        print("Aucune")

    else:

        for _, row in negative_bands.iterrows():

            print(
                f"Bande {int(row['band_id']):3d} | "
                f"<0={row['negative_percent']:6.2f}% | "
                f"min={row['min']:.8f}"
            )

    # ========================================================
    # VALEURS > 1.2
    # ========================================================

    high_bands = band_df[
        band_df["above_1.2_percent"] > 0
    ]

    print("\n" + "-" * 100)
    print("BANDES CONTENANT DES VALEURS > 1.2")
    print("-" * 100)

    if high_bands.empty:

        print("Aucune")

    else:

        for _, row in high_bands.iterrows():

            print(
                f"Bande {int(row['band_id']):3d} | "
                f">1.2={row['above_1.2_percent']:6.2f}% | "
                f"max={row['max']:.8f}"
            )

    # ========================================================
    # BANDES CONTENANT RÉELLEMENT DU SIGNAL
    # ========================================================

    signal_bands = band_df[
        band_df["has_signal"]
    ]

    print("\n" + "-" * 100)
    print("ÉTENDUE DU SIGNAL")
    print("-" * 100)

    if signal_bands.empty:

        first_signal = None
        last_signal = None

        print("Aucune bande contenant du signal.")

    else:

        first_signal = int(
            signal_bands["band_id"].min()
        )

        last_signal = int(
            signal_bands["band_id"].max()
        )

        print(
            "Première bande contenant du signal :",
            first_signal,
        )

        print(
            "Dernière bande contenant du signal :",
            last_signal,
        )

        print(
            "Nombre de bandes contenant du signal :",
            len(signal_bands),
        )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    summary = {
        "stage": stage,
        "filename": path.name,
        "bands": bands,
        "rows": rows,
        "cols": cols,
        "footprint_pixels": n_footprint,
        "min_valid_bands": min_valid,
        "median_valid_bands": median_valid,
        "mean_valid_bands": mean_valid,
        "max_valid_bands": max_valid,
        "complete_pixels": complete_pixels,
        "complete_pixels_percent": complete_percent,

        "bands_with_nan": int(
            np.sum(
                band_df["nan_percent"] > 0
            )
        ),

        "fully_invalid_bands": int(
            np.sum(
                band_df["nan_percent"] == 100
            )
        ),

        "bands_over_90pct_zero": int(
            np.sum(
                band_df["zero_percent"]
                >= ZERO_WARNING_PERCENT
            )
        ),

        "fully_zero_bands": int(
            band_df["all_zero"].sum()
        ),

        "bands_with_negative": int(
            np.sum(
                band_df["negative_percent"] > 0
            )
        ),

        "bands_with_values_above_1.2": int(
            np.sum(
                band_df["above_1.2_percent"] > 0
            )
        ),

        "first_signal_band": first_signal,
        "last_signal_band": last_signal,
        "signal_band_count": int(
            len(signal_bands)
        ),
    }

    return summary, band_df


# ============================================================
# COMPARAISON ENTRE DEUX ÉTAPES
# ============================================================

def compare_two_stages(
    previous_stage,
    previous_df,
    current_stage,
    current_df,
):

    print("\n")
    print("=" * 100)
    print(
        f"COMPARAISON BANDE PAR BANDE : "
        f"{previous_stage} → {current_stage}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Si le nombre de bandes change, on ne peut pas supposer
    # que les indices correspondent exactement.
    # --------------------------------------------------------

    if len(previous_df) != len(current_df):

        print(
            f"Nombre de bandes différent : "
            f"{len(previous_df)} → {len(current_df)}"
        )

        print(
            "Comparaison directe bande par bande non fiable "
            "sans table de correspondance des longueurs d'onde."
        )

        return

    # --------------------------------------------------------
    # Bandes qui avaient du signal et deviennent nulles
    # --------------------------------------------------------

    lost_signal = []

    appeared_signal = []

    for i in range(len(previous_df)):

        before = previous_df.iloc[i]
        after = current_df.iloc[i]

        if (
            bool(before["has_signal"])
            and bool(after["all_zero"])
        ):

            lost_signal.append(i + 1)

        if (
            bool(before["all_zero"])
            and bool(after["has_signal"])
        ):

            appeared_signal.append(i + 1)

    if lost_signal:

        print(
            "\n[ALERTE] Bandes contenant du signal "
            "avant mais entièrement nulles après :"
        )

        print(lost_signal)

    else:

        print(
            "\nAucune bande avec disparition totale "
            "du signal."
        )

    if appeared_signal:

        print(
            "\n[ALERTE] Bandes nulles avant mais "
            "contenant du signal après :"
        )

        print(appeared_signal)

        print(
            "Cela peut notamment révéler un artefact "
            "introduit par un traitement spectral."
        )


# ============================================================
# PIPELINE COMPLET
# ============================================================

def compare_pipeline(
    l2a_path,
    intermediate_folder,
):

    l2a_path = Path(l2a_path)
    intermediate_folder = Path(intermediate_folder)

    if not l2a_path.exists():

        raise FileNotFoundError(
            f"Image L2A introuvable : {l2a_path}"
        )

    if not intermediate_folder.exists():

        raise FileNotFoundError(
            f"Dossier intermédiaire introuvable : "
            f"{intermediate_folder}"
        )

    output_dir = (
        intermediate_folder
        / OUTPUT_DIR_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("DIAGNOSTIC DU PIPELINE HYPERSPECTRAL")
    print("=" * 100)

    print(
        "\nImage originale L2A :",
        l2a_path,
    )

    print(
        "Dossier intermédiaire :",
        intermediate_folder,
    )

    # ========================================================
    # FICHIERS
    # ========================================================

    files = {
        "L2A_original": l2a_path,
    }

    files.update(
        find_intermediate_files(
            intermediate_folder
        )
    )

    desired_order = [
        "L2A_original",
        "reflectance",
        "clean",
        "cleanbands",
        "smooth",
    ]

    summaries = []
    all_band_stats = []

    stage_dataframes = {}

    # ========================================================
    # ANALYSE DE CHAQUE ÉTAPE
    # ========================================================

    for stage in desired_order:

        if stage not in files:
            continue

        summary, band_df = analyse_image(
            stage,
            files[stage],
        )

        if summary is None:
            continue

        summaries.append(summary)

        all_band_stats.append(
            band_df
        )

        stage_dataframes[stage] = band_df

        band_df.to_csv(
            output_dir
            / f"quality_{stage}.csv",
            index=False,
        )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        output_dir
        / "pipeline_quality_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if all_band_stats:

        all_bands_df = pd.concat(
            all_band_stats,
            ignore_index=True,
        )

        all_bands_df.to_csv(
            output_dir
            / "pipeline_quality_all_bands.csv",
            index=False,
        )

    # ========================================================
    # TABLEAU FINAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print("COMPARAISON GLOBALE DU PIPELINE")
    print("=" * 100)

    display_columns = [
        "stage",
        "bands",
        "footprint_pixels",
        "median_valid_bands",
        "complete_pixels_percent",
        "bands_with_nan",
        "fully_zero_bands",
        "bands_with_negative",
        "first_signal_band",
        "last_signal_band",
        "signal_band_count",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # COMPARAISONS SUCCESSIVES
    # ========================================================

    available_stages = [
        stage
        for stage in desired_order
        if stage in stage_dataframes
    ]

    for previous_stage, current_stage in zip(
        available_stages[:-1],
        available_stages[1:],
    ):

        compare_two_stages(
            previous_stage,
            stage_dataframes[previous_stage],
            current_stage,
            stage_dataframes[current_stage],
        )

    # ========================================================
    # DIAGNOSTIC AUTOMATIQUE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("DIAGNOSTIC AUTOMATIQUE")
    print("=" * 100)

    for summary in summaries:

        stage = summary["stage"]

        if summary["fully_zero_bands"] > 0:

            print(
                f"[ALERTE] {stage} : "
                f"{summary['fully_zero_bands']} "
                f"bande(s) entièrement à zéro."
            )

        if summary["bands_with_negative"] > 0:

            print(
                f"[INFO] {stage} : "
                f"{summary['bands_with_negative']} "
                f"bande(s) contiennent des valeurs négatives."
            )

        if summary["bands_with_nan"] > 0:

            print(
                f"[INFO] {stage} : "
                f"{summary['bands_with_nan']} "
                f"bande(s) contiennent des NaN."
            )

    # ========================================================
    # SORTIES
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FICHIERS DE SORTIE")
    print("=" * 100)

    print(
        "Résumé :",
        summary_path,
    )

    print(
        "Détails :",
        output_dir
        / "pipeline_quality_all_bands.csv",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic complet du pipeline "
            "de prétraitement hyperspectral."
        )
    )

    parser.add_argument(
        "l2a",
        help=(
            "Chemin vers l'image hyperspectrale "
            "L2A originale."
        ),
    )

    parser.add_argument(
        "folder",
        help=(
            "Dossier contenant les images "
            "hyperspectrales intermédiaires."
        ),
    )

    args = parser.parse_args()

    compare_pipeline(
        l2a_path=args.l2a,
        intermediate_folder=args.folder,
    )