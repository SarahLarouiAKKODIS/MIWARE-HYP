from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SPATIAL_GROUPS_CSV = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/spatial_groups/spatial_groups.csv"
)

SPECIES_BY_SCENE_CSV = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/phenology_inventory/species_by_scene.csv"
)

OUTPUT_DIR = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/train_test_split"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Fraction souhaitée pour le TEST
TEST_RATIO = 0.25

# Nombre de répartitions aléatoires testées
N_ITERATIONS = 50_000

RANDOM_SEED = 42


# ============================================================
# OPTIONNEL : CLASSES À PRENDRE EN COMPTE POUR OPTIMISER LE SPLIT
# ============================================================

# None = toutes les classes présentes dans species_by_scene.csv
#
# Sinon par exemple :
#
# TARGET_SPECIES = [
#     "Sapin, épicéa",
#     "Châtaignier",
#     "Chênes décidus",
#     "Chênes sempervirents",
#     "Douglas",
#     "Hêtre",
#     "Peuplier",
#     "Pin d'Alep",
#     "Pin laricio/noir",
#     "Pin maritime",
#     "Pin sylvestre",
# ]

TARGET_SPECIES = None


# ============================================================
# POIDS DU SCORE D'OPTIMISATION
# ============================================================

# Ratio global Train/Test
WEIGHT_GLOBAL_RATIO = 10.0

# Équilibre du nombre de pixels par essence
WEIGHT_SPECIES_PIXELS = 3.0

# Représentation des essences en nombre d'images
WEIGHT_SPECIES_IMAGES = 2.0

# Couverture essence × mois
WEIGHT_MONTHS = 3.0

# Forte pénalité si une essence pouvant être dans les deux ensembles
# disparaît complètement du Train ou du Test.
MISSING_SPECIES_PENALTY = 10.0

# Pénalité si une combinaison essence × mois suffisamment disponible
# disparaît du Train ou du Test.
MISSING_MONTH_PENALTY = 5.0


# ============================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================

def load_data():
    spatial = pd.read_csv(SPATIAL_GROUPS_CSV)
    species = pd.read_csv(SPECIES_BY_SCENE_CSV)

    required_spatial = {
        "group_id",
        "image_name",
        "config_path",
    }

    required_species = {
        "config",
        "month",
        "species",
        "pixel_count",
    }

    missing = required_spatial - set(spatial.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans spatial_groups.csv : {missing}"
        )

    missing = required_species - set(species.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans species_by_scene.csv : {missing}"
        )

    # --------------------------------------------------------
    # Harmonisation du nom de config
    # --------------------------------------------------------

    spatial["config"] = spatial["image_name"].astype(str)

    species["config"] = species["config"].astype(str)

    # --------------------------------------------------------
    # Vérification correspondance entre fichiers
    # --------------------------------------------------------

    spatial_configs = set(spatial["config"])
    species_configs = set(species["config"])

    absent_species = spatial_configs - species_configs

    if absent_species:
        print(
            "\n[ATTENTION] Certaines configs de spatial_groups.csv "
            "n'ont aucune ligne dans species_by_scene.csv :"
        )

        for name in sorted(absent_species):
            print("  -", name)

    # On garde toutes les images spatiales.
    # Une image sans essence aura simplement 0 contribution.

    # --------------------------------------------------------
    # Classes utilisées pour l'optimisation
    # --------------------------------------------------------

    if TARGET_SPECIES is not None:
        species = species[
            species["species"].isin(TARGET_SPECIES)
        ].copy()

    return spatial, species


# ============================================================
# AGRÉGATION PAR GROUPE
# ============================================================

def prepare_group_statistics(
    spatial: pd.DataFrame,
    species: pd.DataFrame,
):
    """
    Associe chaque scène à son group_id et prépare les matrices
    nécessaires à l'optimisation.
    """

    mapping = spatial[
        ["config", "group_id"]
    ].drop_duplicates()

    species = species.merge(
        mapping,
        on="config",
        how="inner",
    )

    groups = sorted(
        spatial["group_id"].unique()
    )

    all_species = sorted(
        species["species"].unique()
    )

    months = sorted(
        species["month"].unique()
    )

    # --------------------------------------------------------
    # Pixels par groupe × essence
    # --------------------------------------------------------

    pixels = (
        species
        .groupby(["group_id", "species"])["pixel_count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            index=groups,
            columns=all_species,
            fill_value=0,
        )
    )

    # --------------------------------------------------------
    # Nombre d'images par groupe × essence
    # --------------------------------------------------------

    images = (
        species
        .groupby(["group_id", "species"])["config"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            index=groups,
            columns=all_species,
            fill_value=0,
        )
    )

    # --------------------------------------------------------
    # Présence groupe × essence × mois
    # --------------------------------------------------------

    species_month = (
        species
        .groupby(
            ["group_id", "species", "month"]
        )["config"]
        .nunique()
    )

    return (
        groups,
        all_species,
        months,
        pixels,
        images,
        species_month,
        species,
    )


# ============================================================
# SCORE D'UN SPLIT
# ============================================================

def evaluate_split(
    test_groups: set,
    groups,
    all_species,
    pixels,
    images,
    species_month,
):
    """
    Plus le score est faible, meilleur est le split.
    """

    train_groups = set(groups) - test_groups

    if not train_groups or not test_groups:
        return np.inf

    # ========================================================
    # 1. RATIO GLOBAL DE PIXELS
    # ========================================================

    total_pixels = pixels.values.sum()

    test_pixels_total = (
        pixels.loc[list(test_groups)].values.sum()
    )

    if total_pixels == 0:
        return np.inf

    actual_test_ratio = (
        test_pixels_total / total_pixels
    )

    score = (
        WEIGHT_GLOBAL_RATIO
        * abs(actual_test_ratio - TEST_RATIO)
    )


    # ========================================================
    # 2. RATIO PAR ESSENCE
    # ========================================================

    total_species_pixels = pixels.sum(axis=0)

    train_species_pixels = pixels.loc[
        list(train_groups)
    ].sum(axis=0)

    test_species_pixels = pixels.loc[
        list(test_groups)
    ].sum(axis=0)

    pixel_errors = []

    for species_name in all_species:

        total = total_species_pixels[species_name]

        if total <= 0:
            continue

        test_ratio_species = (
            test_species_pixels[species_name]
            / total
        )

        pixel_errors.append(
            abs(test_ratio_species - TEST_RATIO)
        )

        # Essence totalement absente d'un des ensembles
        if (
            train_species_pixels[species_name] == 0
            or test_species_pixels[species_name] == 0
        ):
            score += MISSING_SPECIES_PENALTY

    if pixel_errors:
        score += (
            WEIGHT_SPECIES_PIXELS
            * np.mean(pixel_errors)
        )


    # ========================================================
    # 3. RATIO DU NOMBRE D'IMAGES PAR ESSENCE
    # ========================================================

    total_species_images = images.sum(axis=0)

    test_species_images = images.loc[
        list(test_groups)
    ].sum(axis=0)

    image_errors = []

    for species_name in all_species:

        total = total_species_images[species_name]

        if total <= 1:
            # Impossible d'avoir cette essence dans les deux
            # ensembles si elle n'existe que sur une image.
            continue

        test_ratio_species = (
            test_species_images[species_name]
            / total
        )

        image_errors.append(
            abs(test_ratio_species - TEST_RATIO)
        )

    if image_errors:
        score += (
            WEIGHT_SPECIES_IMAGES
            * np.mean(image_errors)
        )


    # ========================================================
    # 4. COUVERTURE ESSENCE × MOIS
    # ========================================================

    monthly_errors = []

    # On travaille sur chaque combinaison réellement existante.
    for (
        species_name,
        month
    ), month_data in species_month.groupby(
        level=["species", "month"]
    ):

        # Groupes contenant cette essence pendant ce mois
        available_groups = set(
            month_data.index.get_level_values(
                "group_id"
            )
        )

        n_available = len(available_groups)

        # S'il n'existe qu'un seul groupe spatial,
        # impossible de partager entre Train et Test.
        if n_available < 2:
            continue

        n_test = len(
            available_groups & test_groups
        )

        n_train = len(
            available_groups & train_groups
        )

        test_ratio_month = n_test / n_available

        monthly_errors.append(
            abs(test_ratio_month - TEST_RATIO)
        )

        # Si plusieurs groupes existent mais tous partent
        # du même côté, pénalité.
        if n_test == 0 or n_train == 0:
            score += MISSING_MONTH_PENALTY

    if monthly_errors:
        score += (
            WEIGHT_MONTHS
            * np.mean(monthly_errors)
        )

    return score


# ============================================================
# RECHERCHE DU MEILLEUR SPLIT
# ============================================================

from itertools import combinations


def find_best_split(
    groups,
    all_species,
    pixels,
    images,
    species_month,
    target_test_ratio=0.25,
    min_test_ratio=0.20,
    max_test_ratio=0.30,
):
    """
    Recherche exhaustive du meilleur split spatial.

    Le Test doit représenter entre min_test_ratio et
    max_test_ratio des pixels totaux.

    Aucun groupe spatial ne peut être partagé entre Train et Test.
    """

    groups = list(groups)

    n_groups = len(groups)

    if n_groups < 2:
        raise ValueError(
            "Il faut au moins deux groupes spatiaux."
        )

    total_pixels = pixels.values.sum()

    if total_pixels <= 0:
        raise ValueError(
            "Aucun pixel disponible pour construire le split."
        )

    print("\nRecherche exhaustive du meilleur split...")
    print(f"Groupes spatiaux : {n_groups}")
    print(
        f"Ratio Test autorisé : "
        f"{100 * min_test_ratio:.0f}–"
        f"{100 * max_test_ratio:.0f} %"
    )

    best_score = np.inf
    best_test_groups = None
    best_ratio = None

    n_candidates = 0
    n_valid = 0

    # ========================================================
    # Tester toutes les combinaisons
    # ========================================================

    for n_test_groups in range(1, n_groups):

        for combination in combinations(
            groups,
            n_test_groups,
        ):

            n_candidates += 1

            test_groups = set(combination)

            # ------------------------------------------------
            # Calcul du ratio réel en pixels
            # ------------------------------------------------

            test_pixels = (
                pixels
                .loc[list(test_groups)]
                .values
                .sum()
            )

            actual_ratio = (
                test_pixels / total_pixels
            )

            # ------------------------------------------------
            # CONTRAINTE FORTE SUR LE RATIO
            # ------------------------------------------------

            if not (
                min_test_ratio
                <= actual_ratio
                <= max_test_ratio
            ):
                continue

            n_valid += 1

            # ------------------------------------------------
            # Score essence / images / mois
            # ------------------------------------------------

            score = evaluate_split(
                test_groups,
                groups,
                all_species,
                pixels,
                images,
                species_month,
            )

            # ------------------------------------------------
            # Donner davantage d'importance à 25 %
            # ------------------------------------------------

            score += (
                10
                * abs(
                    actual_ratio
                    - target_test_ratio
                )
            )

            if score < best_score:

                best_score = score
                best_test_groups = (
                    test_groups.copy()
                )
                best_ratio = actual_ratio

                print(
                    f"Nouveau meilleur split : "
                    f"score={score:.4f} | "
                    f"Test={100 * actual_ratio:.1f}% | "
                    f"{len(test_groups)} groupe(s)"
                )

    # ========================================================
    # Vérification
    # ========================================================

    if best_test_groups is None:

        raise RuntimeError(
            "\nImpossible de trouver un split respectant "
            f"{100 * min_test_ratio:.0f}–"
            f"{100 * max_test_ratio:.0f} % de pixels Test.\n"
            "Les groupes spatiaux sont probablement trop "
            "déséquilibrés."
        )

    train_groups = (
        set(groups)
        - best_test_groups
    )

    print("\nMeilleur split trouvé :")
    print(
        f"Test = {100 * best_ratio:.2f} % "
        f"des pixels"
    )
    print(
        f"Train = {100 * (1 - best_ratio):.2f} %"
    )
    print(
        f"Groupes Train : {len(train_groups)}"
    )
    print(
        f"Groupes Test : {len(best_test_groups)}"
    )
    print(
        f"Combinaisons testées : {n_candidates}"
    )
    print(
        f"Combinaisons respectant le ratio : {n_valid}"
    )

    return (
        train_groups,
        best_test_groups,
        best_score,
    )

# ============================================================
# GÉNÉRATION DES TABLEAUX
# ============================================================

def create_outputs(
    spatial,
    species,
    train_groups,
    test_groups,
):
    # --------------------------------------------------------
    # Ajouter le split
    # --------------------------------------------------------

    group_split = {}

    for g in train_groups:
        group_split[g] = "Train"

    for g in test_groups:
        group_split[g] = "Test"

    spatial = spatial.copy()

    spatial["split"] = spatial["group_id"].map(
        group_split
    )

    # ========================================================
    # train_groups.csv / test_groups.csv
    # ========================================================

    train_df = (
        spatial[
            spatial["split"] == "Train"
        ]
        .sort_values(
            ["group_id", "image_name"]
        )
        .copy()
    )

    test_df = (
        spatial[
            spatial["split"] == "Test"
        ]
        .sort_values(
            ["group_id", "image_name"]
        )
        .copy()
    )

    train_df.to_csv(
        OUTPUT_DIR / "train_groups.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_groups.csv",
        index=False,
    )


    # ========================================================
    # train_configs.txt / test_configs.txt
    # ========================================================

    train_configs = (
        train_df["config_path"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    test_configs = (
        test_df["config_path"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    (
        OUTPUT_DIR / "train_configs.txt"
    ).write_text(
        "\n".join(train_configs) + "\n",
        encoding="utf-8",
    )

    (
        OUTPUT_DIR / "test_configs.txt"
    ).write_text(
        "\n".join(test_configs) + "\n",
        encoding="utf-8",
    )


    # ========================================================
    # Associer chaque config du tableau species à son split
    # ========================================================

    config_split = (
        spatial[
            ["config", "split"]
        ]
        .drop_duplicates()
    )

    species_split = species.merge(
        config_split,
        on="config",
        how="inner",
    )


    # ========================================================
    # split_species_pixels.csv
    # ========================================================

    pixels = (
        species_split
        .groupby(
            ["species", "split"]
        )["pixel_count"]
        .sum()
        .unstack(fill_value=0)
    )

    for col in ["Train", "Test"]:
        if col not in pixels.columns:
            pixels[col] = 0

    pixels["Total"] = (
        pixels["Train"]
        + pixels["Test"]
    )

    pixels["Test_ratio"] = np.where(
        pixels["Total"] > 0,
        pixels["Test"] / pixels["Total"],
        np.nan,
    )

    pixels = pixels[
        [
            "Train",
            "Test",
            "Total",
            "Test_ratio",
        ]
    ]

    pixels.to_csv(
        OUTPUT_DIR
        / "split_species_pixels.csv"
    )


    # ========================================================
    # split_species_images.csv
    # ========================================================

    images = (
        species_split
        .groupby(
            ["species", "split"]
        )["config"]
        .nunique()
        .unstack(fill_value=0)
    )

    for col in ["Train", "Test"]:
        if col not in images.columns:
            images[col] = 0

    images["Total"] = (
        images["Train"]
        + images["Test"]
    )

    images["Test_ratio"] = np.where(
        images["Total"] > 0,
        images["Test"] / images["Total"],
        np.nan,
    )

    images = images[
        [
            "Train",
            "Test",
            "Total",
            "Test_ratio",
        ]
    ]

    images.to_csv(
        OUTPUT_DIR
        / "split_species_images.csv"
    )


    # ========================================================
    # split_species_months.csv
    # ========================================================

    monthly = (
        species_split
        .groupby(
            [
                "species",
                "month",
                "split",
            ]
        )["config"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["Train", "Test"]:
        if col not in monthly.columns:
            monthly[col] = 0

    monthly["Total_images"] = (
        monthly["Train"]
        + monthly["Test"]
    )

    monthly["Train_present"] = (
        monthly["Train"] > 0
    )

    monthly["Test_present"] = (
        monthly["Test"] > 0
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "split_species_months.csv",
        index=False,
    )


    return (
        train_df,
        test_df,
        pixels,
        images,
        monthly,
    )


# ============================================================
# AFFICHAGE DU BILAN
# ============================================================

def print_summary(
    train_df,
    test_df,
    pixels,
    images,
    monthly,
    score,
):
    print("\n" + "=" * 80)
    print("SPLIT FINAL")
    print("=" * 80)

    print(
        f"\nScore optimisation : {score:.6f}"
    )

    print(
        f"Groupes Train : "
        f"{train_df['group_id'].nunique()}"
    )

    print(
        f"Groupes Test  : "
        f"{test_df['group_id'].nunique()}"
    )

    print(
        f"Images Train  : "
        f"{len(train_df)}"
    )

    print(
        f"Images Test   : "
        f"{len(test_df)}"
    )


    # --------------------------------------------------------
    # Ratio global pixels
    # --------------------------------------------------------

    train_pixels = pixels["Train"].sum()
    test_pixels = pixels["Test"].sum()

    total = train_pixels + test_pixels

    if total > 0:

        print(
            "\nPixels :"
        )

        print(
            f"Train : {train_pixels:,} "
            f"({100 * train_pixels / total:.1f} %)"
        )

        print(
            f"Test  : {test_pixels:,} "
            f"({100 * test_pixels / total:.1f} %)"
        )


    # --------------------------------------------------------
    # Classes problématiques
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLASSES SANS TRAIN OU TEST")
    print("=" * 80)

    problematic = pixels[
        (pixels["Train"] == 0)
        | (pixels["Test"] == 0)
    ]

    if problematic.empty:

        print(
            "Toutes les essences sont présentes "
            "dans Train et Test."
        )

    else:

        print(problematic)


    # --------------------------------------------------------
    # Phénologie problématique
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ESSENCE × MOIS ABSENT DU TRAIN OU DU TEST")
    print("=" * 80)

    problematic_months = monthly[
        (
            ~monthly["Train_present"]
            | ~monthly["Test_present"]
        )
        & (monthly["Total_images"] >= 2)
    ]

    if problematic_months.empty:

        print(
            "Toutes les combinaisons essence × mois "
            "suffisamment disponibles sont représentées "
            "dans les deux ensembles."
        )

    else:

        print(
            problematic_months.to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Charger
    # --------------------------------------------------------

    spatial, species = load_data()

    # --------------------------------------------------------
    # Préparer statistiques
    # --------------------------------------------------------

    (
        groups,
        all_species,
        months,
        pixels,
        images,
        species_month,
        species_with_groups,
    ) = prepare_group_statistics(
        spatial,
        species,
    )


    # ============================================================
    # DIAGNOSTIC : NOMBRE DE GROUPES SPATIAUX PAR ESSENCE
    # ============================================================

    species_group_coverage = (
        species_with_groups
        .groupby("species")["group_id"]
        .nunique()
        .sort_values()
    )

    print("\n" + "=" * 80)
    print("NOMBRE DE GROUPES SPATIAUX PAR ESSENCE")
    print("=" * 80)

    print(species_group_coverage)

    print("\n" + "=" * 80)
    print("DONNÉES")
    print("=" * 80)

    print(
        f"Images       : {spatial['config'].nunique()}"
    )

    print(
        f"Groupes      : {len(groups)}"
    )

    print(
        f"Essences     : {len(all_species)}"
    )

    print(
        f"Mois         : {months}"
    )

    # --------------------------------------------------------
    # Optimiser split
    # --------------------------------------------------------

    (
        train_groups,
        test_groups,
        score,
    ) = find_best_split(
        groups,
        all_species,
        pixels,
        images,
        species_month,
    )

    # --------------------------------------------------------
    # Exports
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        pixels_result,
        images_result,
        monthly_result,
    ) = create_outputs(
        spatial,
        species,
        train_groups,
        test_groups,
    )

    # --------------------------------------------------------
    # Bilan
    # --------------------------------------------------------

    print_summary(
        train_df,
        test_df,
        pixels_result,
        images_result,
        monthly_result,
        score,
    )

    print(
        f"\nRésultats enregistrés dans : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()