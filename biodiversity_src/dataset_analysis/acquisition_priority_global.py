from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import rasterio

from utils import load_config



# ============================================================
# CONFIGURATION
# ============================================================

CONFIGS_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess")

OUTPUT_DIR = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/acquisition_priority"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Mois pour lesquels on souhaite rechercher de nouvelles
# acquisitions EnMAP.
#
# Exemple :
# juillet, août, septembre, octobre
REQUEST_MONTHS = [7, 8, 9, 10]


MONTH_NAMES = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}


# ============================================================
# CLASSES FORESTIÈRES D'INTÉRÊT
# ============================================================

# Les classes réellement intéressantes pour une classification
# d'essences.

TARGET_SPECIES = [
    "Sapin, épicéa",
    "Châtaignier",
    "Chênes décidus",
    "Chênes sempervirents",
    "Douglas",
    "Hêtre",
    "Mélèze",
    "Peuplier",
    "Pin d'Alep",
    "Pin laricio, pin noir",
    "Pin maritime",
    "Pin sylvestre",
    "Pin à crochets, pin cembro",
    "Robinier",
]


# Classes présentes dans la BD Forêt mais non utilisées
# pour mesurer la richesse spécifique.

GENERIC_CLASSES = {
    "Conifères",
    "Feuillus",
    "Mixte",
    "NC",
    "NR",
    "Pin autre",
    "Pins mélangés",
}


# ============================================================
# ÉTAT ACTUEL DU DATASET
# ============================================================
#
# Nombre de pixels actuellement disponibles par essence
# et par mois.
#
# Cela permet de donner plus d'importance aux essences
# sous-représentées lors de la sélection des nouvelles scènes.
# ============================================================

CURRENT_MONTHLY_PIXELS = {

    "Châtaignier": {
        3: 258608,
        5: 226962,
        6: 144905,
        11: 103674,
    },

    "Chênes décidus": {
        3: 869046,
        5: 2642657,
        6: 2132971,
        11: 429728,
    },

    "Chênes sempervirents": {
        3: 187186,
        5: 0,
        6: 143,
        11: 182485,
    },

    "Douglas": {
        3: 236730,
        5: 237804,
        6: 271193,
        11: 168062,
    },

    "Hêtre": {
        3: 631998,
        5: 313892,
        6: 188766,
        11: 178415,
    },

    "Mélèze": {
        3: 6929,
        5: 7288,
        6: 10352,
        11: 1362,
    },

    "Peuplier": {
        3: 9711,
        5: 20435,
        6: 53780,
        11: 16142,
    },

    "Pin d'Alep": {
        3: 58763,
        5: 0,
        6: 0,
        11: 56897,
    },

    "Pin laricio, pin noir": {
        3: 149627,
        5: 35656,
        6: 35417,
        11: 142158,
    },

    "Pin maritime": {
        3: 10261,
        5: 5189,
        6: 223581,
        11: 46451,
    },

    "Pin sylvestre": {
        3: 18682,
        5: 117779,
        6: 113511,
        11: 13394,
    },

    "Pin à crochets, pin cembro": {
        3: 2856,
        5: 1208,
        6: 0,
        11: 543,
    },

    "Robinier": {
        3: 20948,
        5: 9681,
        6: 15546,
        11: 0,
    },

    "Sapin, épicéa": {
        3: 370433,
        5: 136610,
        6: 145752,
        11: 269927,
    },
}


# ============================================================
# SCORE
# ============================================================
#
# Le score final combine :
#
# 40 % : intérêt des essences présentes
# 35 % : quantité de forêt exploitable
# 25 % : diversité d'essences
#
# ============================================================

WEIGHT_PRIORITY = 0.40
WEIGHT_FOREST = 0.35
WEIGHT_DIVERSITY = 0.25


BACKGROUND_VALUE = 0
NODATA_VALUE = 255


# ============================================================
# OUTILS
# ============================================================

def normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalization entre 0 et 1.
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# POIDS GLOBAL D'UNE ESSENCE
# ============================================================

def compute_global_species_weights() -> dict[str, float]:
    """
    Plus une essence possède peu de pixels dans le dataset
    actuel, plus son poids est élevé.

    Utilisation d'une racine carrée pour éviter des poids
    extrêmement différents entre classes.
    """

    totals = {}

    for species in TARGET_SPECIES:

        monthly = CURRENT_MONTHLY_PIXELS.get(
            species,
            {}
        )

        totals[species] = sum(
            monthly.values()
        )

    raw_weights = {
        species: 1.0 / np.sqrt(total + 1)
        for species, total in totals.items()
    }

    max_weight = max(
        raw_weights.values()
    )

    return {
        species: weight / max_weight
        for species, weight in raw_weights.items()
    }


# ============================================================
# POIDS POUR UN MOIS DONNÉ
# ============================================================

def compute_month_species_weights(
    month: int,
) -> dict[str, float]:
    """
    Poids d'une essence pour une nouvelle acquisition
    d'un mois donné.

    Une essence absente ou très peu représentée pendant
    ce mois reçoit une priorité élevée.

    Pour les mois totalement absents du dataset actuel,
    la rareté globale sert de critère secondaire.
    """

    global_weights = compute_global_species_weights()

    current_counts = {
        species: CURRENT_MONTHLY_PIXELS
        .get(species, {})
        .get(month, 0)
        for species in TARGET_SPECIES
    }

    max_current = max(
        current_counts.values(),
        default=0,
    )

    weights = {}

    for species in TARGET_SPECIES:

        count = current_counts[species]

        # ---------------------------------------------
        # Si aucun échantillon n'existe pour ce mois
        # ---------------------------------------------

        if max_current == 0:

            weights[species] = (
                0.5
                + 0.5 * global_weights[species]
            )

        else:

            # Rareté au sein du mois
            month_need = (
                1.0
                - np.sqrt(
                    count / max_current
                )
            )

            # Mélange besoin mensuel + rareté globale
            weights[species] = (
                0.75 * month_need
                + 0.25 * global_weights[species]
            )

    return weights


# ============================================================
# ANALYSE D'UN MASQUE FORESTIER
# ============================================================

def analyse_forest_mask(
    mask_path: str | Path,
    id_to_species: dict[int, str],
) -> dict:

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Masque introuvable : {mask_path}"
        )

    class_counts = {}

    total_pixels = 0

    background_pixels = 0
    nodata_pixels = 0

    # Lecture bloc par bloc
    with rasterio.open(mask_path) as src:

        total_pixels = (
            src.width
            * src.height
        )

        for _, window in src.block_windows(1):

            data = src.read(
                1,
                window=window,
            )

            values, counts = np.unique(
                data,
                return_counts=True,
            )

            for value, count in zip(
                values,
                counts,
            ):

                value = int(value)
                count = int(count)

                if value == BACKGROUND_VALUE:
                    background_pixels += count
                    continue

                if value == NODATA_VALUE:
                    nodata_pixels += count
                    continue

                species = id_to_species.get(
                    value,
                    f"Classe_{value}",
                )

                class_counts[species] = (
                    class_counts.get(species, 0)
                    + count
                )

    # ========================================================
    # PIXELS FORESTIERS
    # ========================================================

    forest_pixels = sum(
        class_counts.values()
    )

    forest_fraction = (
        forest_pixels
        / total_pixels
        if total_pixels > 0
        else 0
    )


    # ========================================================
    # CLASSES D'ESSENCES SPÉCIFIQUES
    # ========================================================

    target_counts = {
        species: class_counts.get(
            species,
            0,
        )
        for species in TARGET_SPECIES
    }

    species_present = [
        species
        for species, count
        in target_counts.items()
        if count > 0
    ]

    n_species = len(
        species_present
    )

    target_pixels = sum(
        target_counts.values()
    )


    # ========================================================
    # PROPORTION DE PIXELS RÉELLEMENT UTILISABLES
    # POUR LES CLASSES D'ESSENCES
    # ========================================================

    target_fraction = (
        target_pixels
        / forest_pixels
        if forest_pixels > 0
        else 0
    )


    return {
        "total_pixels": total_pixels,
        "forest_pixels": forest_pixels,
        "forest_fraction": forest_fraction,
        "target_pixels": target_pixels,
        "target_fraction": target_fraction,
        "n_species": n_species,
        "species_present": species_present,
        "class_counts": class_counts,
        "target_counts": target_counts,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    config_paths = sorted(
        CONFIGS_DIR.glob("*.json")
    )

    if not config_paths:

        raise FileNotFoundError(
            f"Aucune config dans : {CONFIGS_DIR}"
        )


    print(
        f"{len(config_paths)} configs trouvées."
    )


    # ========================================================
    # POIDS DES ESSENCES
    # ========================================================

    global_species_weights = (
        compute_global_species_weights()
    )

    month_species_weights = {
        month: compute_month_species_weights(
            month
        )
        for month in REQUEST_MONTHS
    }


    rows = []
    species_rows = []
    errors = []


    # ========================================================
    # BOUCLE CONFIG
    # ========================================================

    for i, config_path in enumerate(
        config_paths,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"[{i}/{len(config_paths)}] "
            f"{config_path.name}"
        )
        print("=" * 70)

        try:

            config = load_config(
                config_path
            )

            config_forest = load_config(
                config["config_forest_path"]
            )


            # =================================================
            # CLASS MAP
            # =================================================

            class_map = (
                config_forest["CLASS_MAP"]
            )

            id_to_species = {
                int(class_id): species
                for species, class_id
                in class_map.items()
            }


            # =================================================
            # MASQUE
            # =================================================

            path_res = Path(
                config["Path_res"]
            )

            mask_path = (
                path_res
                / "Forest"
                / "mask_foret_classes.tif"
            )


            result = analyse_forest_mask(
                mask_path,
                id_to_species,
            )


            # =================================================
            # SCORE GLOBAL DES ESSENCES
            # =================================================

            global_priority_raw = 0.0

            for species, count in (
                result["target_counts"].items()
            ):

                if count <= 0:
                    continue

                global_priority_raw += (
                    global_species_weights[
                        species
                    ]
                    * np.log1p(count)
                )


            row = {
                "config": config_path.stem,
                "config_path": str(
                    config_path
                ),
                "mask_path": str(
                    mask_path
                ),

                "total_pixels": (
                    result["total_pixels"]
                ),

                "forest_pixels": (
                    result["forest_pixels"]
                ),

                "forest_fraction": (
                    result["forest_fraction"]
                ),

                "target_pixels": (
                    result["target_pixels"]
                ),

                "target_fraction": (
                    result["target_fraction"]
                ),

                "n_species": (
                    result["n_species"]
                ),

                "species_present": "; ".join(
                    result[
                        "species_present"
                    ]
                ),

                "global_priority_raw": (
                    global_priority_raw
                ),
            }


            # =================================================
            # SCORE POUR CHAQUE MOIS DEMANDÉ
            # =================================================

            for month in REQUEST_MONTHS:

                priority = 0.0

                for species, count in (
                    result[
                        "target_counts"
                    ].items()
                ):

                    if count <= 0:
                        continue

                    priority += (
                        month_species_weights[
                            month
                        ][species]
                        * np.log1p(count)
                    )

                row[
                    f"priority_{month:02d}"
                ] = priority


            rows.append(row)


            # =================================================
            # DÉTAIL ESSENCES PAR IMAGE
            # =================================================

            for species in TARGET_SPECIES:

                count = result[
                    "target_counts"
                ][species]

                species_rows.append(
                    {
                        "config": (
                            config_path.stem
                        ),

                        "species": species,

                        "pixel_count": count,

                        "present": (
                            count > 0
                        ),

                        "global_need_weight": (
                            global_species_weights[
                                species
                            ]
                        ),
                    }
                )


            print(
                f"Pixels forêt : "
                f"{result['forest_pixels']:,}"
            )

            print(
                f"Essences spécifiques : "
                f"{result['n_species']}"
            )

            print(
                "Présentes :",
                result["species_present"],
            )


        except Exception as exc:

            print(
                f"[ERREUR] {exc}"
            )

            errors.append(
                {
                    "config": (
                        config_path.name
                    ),
                    "error": str(exc),
                }
            )


    # ============================================================
    # DATAFRAME
    # ============================================================

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Aucune image valide analysée."
        )


    # ============================================================
    # NORMALISATION DES CRITÈRES
    # ============================================================

    # Quantité de forêt :
    # log pour éviter qu'une très grosse scène domine tout.
    df["forest_raw"] = np.log1p(
        df["target_pixels"]
    )

    df["forest_score"] = normalize(
        df["forest_raw"]
    )


    # Diversité d'essences
    df["diversity_score"] = (
        df["n_species"]
        / len(TARGET_SPECIES)
    )


    # Priorité globale
    df["priority_score"] = normalize(
        df["global_priority_raw"]
    )


    # ============================================================
    # SCORE GLOBAL
    # ============================================================

    df["global_score"] = (

        WEIGHT_FOREST
        * df["forest_score"]

        + WEIGHT_DIVERSITY
        * df["diversity_score"]

        + WEIGHT_PRIORITY
        * df["priority_score"]
    )


    # ============================================================
    # SCORE PAR MOIS
    # ============================================================

    for month in REQUEST_MONTHS:

        raw_col = (
            f"priority_{month:02d}"
        )

        normalized_col = (
            f"priority_score_{month:02d}"
        )

        final_col = (
            f"score_{month:02d}"
        )

        rank_col = (
            f"rank_{month:02d}"
        )

        df[normalized_col] = normalize(
            df[raw_col]
        )

        df[final_col] = (

            WEIGHT_FOREST
            * df["forest_score"]

            + WEIGHT_DIVERSITY
            * df["diversity_score"]

            + WEIGHT_PRIORITY
            * df[normalized_col]
        )

        df[rank_col] = (
            df[final_col]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )


    # ============================================================
    # CLASSEMENT GLOBAL
    # ============================================================

    df["global_rank"] = (
        df["global_score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )


    df = df.sort_values(
        "global_rank"
    )


    # ============================================================
    # EXPORT GLOBAL
    # ============================================================

    df.to_csv(
        OUTPUT_DIR
        / "acquisition_priority_global.csv",
        index=False,
    )


    # ============================================================
    # EXPORT PAR MOIS
    # ============================================================

    for month in REQUEST_MONTHS:

        month_name = (
            MONTH_NAMES[month]
            .lower()
            .replace("û", "u")
            .replace("é", "e")
            .replace("è", "e")
        )

        month_df = df.sort_values(
            f"rank_{month:02d}"
        )

        month_df.to_csv(
            OUTPUT_DIR
            / f"acquisition_priority_{month:02d}_{month_name}.csv",
            index=False,
        )


    # ============================================================
    # DÉTAIL ESSENCE × IMAGE
    # ============================================================

    species_df = pd.DataFrame(
        species_rows
    )

    species_df.to_csv(
        OUTPUT_DIR
        / "image_species_details.csv",
        index=False,
    )


    # ============================================================
    # MATRICE IMAGE × ESSENCE
    # ============================================================

    species_matrix = (
        species_df
        .pivot(
            index="config",
            columns="species",
            values="pixel_count",
        )
        .fillna(0)
        .astype(int)
    )

    species_matrix.to_csv(
        OUTPUT_DIR
        / "image_species_pixel_matrix.csv"
    )


    # ============================================================
    # ERREURS
    # ============================================================

    if errors:

        pd.DataFrame(
            errors
        ).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )


    # ============================================================
    # AFFICHAGE
    # ============================================================

    print("\n" + "=" * 90)
    print("CLASSEMENT GLOBAL DES EMPRISES")
    print("=" * 90)

    display_columns = [
        "global_rank",
        "config",
        "forest_pixels",
        "target_pixels",
        "n_species",
        "global_score",
        "species_present",
    ]

    print(
        df[display_columns]
        .to_string(
            index=False
        )
    )


    for month in REQUEST_MONTHS:

        print("\n" + "=" * 90)
        print(
            f"PRIORITÉ POUR UNE ACQUISITION EN "
            f"{MONTH_NAMES[month].upper()}"
        )
        print("=" * 90)

        cols = [
            f"rank_{month:02d}",
            "config",
            "target_pixels",
            "n_species",
            f"score_{month:02d}",
            "species_present",
        ]

        print(
            df.sort_values(
                f"rank_{month:02d}"
            )[cols]
            .head(15)
            .to_string(
                index=False
            )
        )


    print(
        f"\nRésultats enregistrés dans : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()