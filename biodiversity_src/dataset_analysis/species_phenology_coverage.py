from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import re
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio

from utils import load_config


# ============================================================
# CONFIGURATION
# ============================================================

CONFIGS_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess")

OUTPUT_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/phenology_inventory")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_VALUE = 0
NODATA_VALUE = 255


# ============================================================
# SAISONS
# ============================================================

def get_season(month: int) -> str:
    """
    Retourne la saison météorologique correspondant au mois.
    """
    if month in (12, 1, 2):
        return "Hiver"

    if month in (3, 4, 5):
        return "Printemps"

    if month in (6, 7, 8):
        return "Été"

    if month in (9, 10, 11):
        return "Automne"

    raise ValueError(f"Mois invalide : {month}")


# ============================================================
# EXTRACTION DATE XML
# ============================================================

def extract_acquisition_date(xml_path: str | Path) -> pd.Timestamp:
    """
    Extrait une date d'acquisition depuis le XML EnMAP.

    La fonction cherche d'abord dans les champs dont le nom
    ressemble à une date d'acquisition, puis utilise une recherche
    ISO dans le XML comme fallback.
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML introuvable : {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Noms de champs susceptibles de contenir la date
    candidate_keywords = (
        "acquisition",
        "sensing",
        "observation",
        "starttime",
        "start_time",
        "datetime",
        "date",
    )

    candidates = []

    for elem in root.iter():

        if elem.text is None:
            continue

        text = elem.text.strip()

        if not text:
            continue

        # Supprime éventuellement le namespace
        tag = elem.tag.split("}")[-1].lower()

        if any(keyword in tag for keyword in candidate_keywords):
            candidates.append((tag, text))

    # --------------------------------------------------------
    # Essayer les champs XML candidats
    # --------------------------------------------------------

    for tag, value in candidates:

        try:
            date = pd.to_datetime(value)

            if not pd.isna(date):
                return date

        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback : chercher une date ISO dans tout le XML
    # --------------------------------------------------------

    xml_text = xml_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    patterns = [
        # 2024-07-12T10:32:45
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?",
        # 2024-07-12
        r"\d{4}-\d{2}-\d{2}",
    ]

    for pattern in patterns:

        matches = re.findall(pattern, xml_text)

        for value in matches:

            try:
                date = pd.to_datetime(value)

                if not pd.isna(date):
                    return date

            except Exception:
                pass

    raise ValueError(
        f"Impossible d'extraire une date d'acquisition depuis : {xml_path}"
    )


# ============================================================
# LECTURE DU MASQUE FORESTIER
# ============================================================

def count_forest_classes(
    raster_path: str | Path,
    *,
    background_value: int = 0,
    nodata_value: int = 255,
) -> Counter:
    """
    Compte les pixels de chaque classe forestière.

    Lecture par blocs pour éviter de charger tout le raster
    en mémoire.
    """

    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Masque forestier introuvable : {raster_path}"
        )

    counts = Counter()

    with rasterio.open(raster_path) as src:

        if src.count != 1:
            raise ValueError(
                f"Le masque doit avoir une seule bande : {raster_path}"
            )

        for _, window in src.block_windows(1):

            data = src.read(1, window=window)

            values, n_pixels = np.unique(
                data,
                return_counts=True,
            )

            for value, count in zip(values, n_pixels):

                value = int(value)

                if value in (
                    background_value,
                    nodata_value,
                ):
                    continue

                counts[value] += int(count)

    return counts


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    config_paths = sorted(
        CONFIGS_DIR.glob("*.json")
    )

    if not config_paths:
        raise FileNotFoundError(
            f"Aucune config trouvée dans : {CONFIGS_DIR}"
        )

    print(
        f"{len(config_paths)} configurations trouvées."
    )

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    scene_rows = []
    species_rows = []
    errors = []

    # --------------------------------------------------------
    # Parcours des configs
    # --------------------------------------------------------

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

            # =================================================
            # CONFIG
            # =================================================

            config = load_config(config_path)

            config_forest = load_config(
                config["config_forest_path"]
            )

            Path_data = Path(config["Path_data"])
            Path_res = Path(config["Path_res"])

            # =================================================
            # XML
            # =================================================

            xml_path = (
                Path_data
                / Path(config["xml_path"])
            )

            acquisition_date = (
                extract_acquisition_date(xml_path)
            )

            month = acquisition_date.month
            season = get_season(month)

            print(
                "Date acquisition :",
                acquisition_date.date(),
            )
            print("Saison :", season)

            # =================================================
            # CLASS MAP
            # =================================================

            class_map = config_forest["CLASS_MAP"]

            # CLASS_MAP :
            # {
            #     "Sapin, épicéa": 1,
            #     "Châtaignier": 2,
            #     ...
            # }

            id_to_species = {
                int(class_id): species
                for species, class_id
                in class_map.items()
            }

            # =================================================
            # MASQUE FORET
            # =================================================

            forest_mask = (
                Path_res
                / "Forest"
                / "mask_foret_classes.tif"
            )

            counts = count_forest_classes(
                forest_mask,
                background_value=BACKGROUND_VALUE,
                nodata_value=NODATA_VALUE,
            )

            # =================================================
            # ESSENCES PRÉSENTES
            # =================================================

            species_present = []

            for class_id, pixel_count in sorted(
                counts.items()
            ):

                species_name = id_to_species.get(
                    class_id,
                    f"Classe_{class_id}",
                )

                species_present.append(
                    species_name
                )

                species_rows.append(
                    {
                        "config": config_path.stem,
                        "date": acquisition_date.date(),
                        "year": acquisition_date.year,
                        "month": month,
                        "season": season,
                        "class_id": class_id,
                        "species": species_name,
                        "pixel_count": pixel_count,
                    }
                )

            print(
                "Essences présentes :",
                species_present,
            )

            # =================================================
            # RÉSUMÉ IMAGE
            # =================================================

            scene_rows.append(
                {
                    "config": config_path.stem,
                    "xml": str(xml_path),
                    "forest_mask": str(forest_mask),
                    "date": acquisition_date.date(),
                    "year": acquisition_date.year,
                    "month": month,
                    "season": season,
                    "n_species": len(species_present),
                    "species_present": "; ".join(
                        species_present
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[ERREUR] {config_path.name} : {exc}"
            )

            errors.append(
                {
                    "config": config_path.name,
                    "error": str(exc),
                }
            )


    # ============================================================
    # DATAFRAMES
    # ============================================================

    scenes_df = pd.DataFrame(scene_rows)
    species_df = pd.DataFrame(species_rows)
    errors_df = pd.DataFrame(errors)

    # ============================================================
    # SYNTHÈSE ESSENCE × MOIS
    # ============================================================

    month_order = [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12
    ]

    month_names = {
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


    if not species_df.empty:

        # ========================================================
        # 1. NOMBRE D'IMAGES PAR ESSENCE ET PAR MOIS
        # ========================================================

        monthly_scene_count = (
            species_df
            .groupby(["species", "month"])["config"]
            .nunique()
            .unstack(fill_value=0)
        )

        # Garantit les 12 mois, même s'ils sont absents
        monthly_scene_count = monthly_scene_count.reindex(
            columns=month_order,
            fill_value=0,
        )

        # Remplace 1,2,3... par Janvier, Février...
        monthly_scene_count.columns = [
            month_names[m]
            for m in monthly_scene_count.columns
        ]


        # ========================================================
        # 2. NOMBRE DE PIXELS PAR ESSENCE ET PAR MOIS
        # ========================================================

        monthly_pixel_count = (
            species_df
            .groupby(["species", "month"])["pixel_count"]
            .sum()
            .unstack(fill_value=0)
        )

        monthly_pixel_count = monthly_pixel_count.reindex(
            columns=month_order,
            fill_value=0,
        )

        monthly_pixel_count.columns = [
            month_names[m]
            for m in monthly_pixel_count.columns
        ]


        # ========================================================
        # 3. NIVEAU DE REPRÉSENTATION
        # ========================================================

        def representation_level(pixel_count: int) -> str:
            """
            Classe la quantité de pixels disponibles.

            Seuils indicatifs à adapter selon ton étude.
            """

            if pixel_count == 0:
                return "Absent"

            elif pixel_count < 1_000:
                return "Très faible"

            elif pixel_count < 10_000:
                return "Faible"

            elif pixel_count < 50_000:
                return "Moyen"

            else:
                return "Bon"


        monthly_representation = monthly_pixel_count.map(
            representation_level
        )


        # ========================================================
        # 4. NOMBRE DE MOIS REPRÉSENTÉS
        # ========================================================

        monthly_coverage = pd.DataFrame(
            index=monthly_pixel_count.index
        )

        monthly_coverage["n_months_present"] = (
            monthly_pixel_count > 0
        ).sum(axis=1)

        monthly_coverage["n_months_good"] = (
            monthly_pixel_count >= 50_000
        ).sum(axis=1)


        # ========================================================
        # 5. MOIS ABSENTS
        # ========================================================

        monthly_coverage["missing_months"] = (
            monthly_pixel_count.apply(
                lambda row: ", ".join(
                    month
                    for month in monthly_pixel_count.columns
                    if row[month] == 0
                ),
                axis=1,
            )
        )


        # ========================================================
        # 6. MOIS FAIBLEMENT REPRÉSENTÉS
        # ========================================================

        monthly_coverage["weak_months"] = (
            monthly_pixel_count.apply(
                lambda row: ", ".join(
                    month
                    for month in monthly_pixel_count.columns
                    if 0 < row[month] < 10_000
                ),
                axis=1,
            )
        )


        # ========================================================
        # EXPORT
        # ========================================================

        monthly_scene_count.to_csv(
            OUTPUT_DIR / "species_month_scene_count.csv"
        )

        monthly_pixel_count.to_csv(
            OUTPUT_DIR / "species_month_pixel_count.csv"
        )

        monthly_representation.to_csv(
            OUTPUT_DIR / "species_month_representation.csv"
        )

        monthly_coverage.to_csv(
            OUTPUT_DIR / "species_month_coverage.csv"
        )


        # ========================================================
        # AFFICHAGE
        # ========================================================

        print("\n" + "=" * 80)
        print("NOMBRE DE PIXELS PAR ESSENCE ET PAR MOIS")
        print("=" * 80)

        print(monthly_pixel_count)


        print("\n" + "=" * 80)
        print("NOMBRE D'IMAGES PAR ESSENCE ET PAR MOIS")
        print("=" * 80)

        print(monthly_scene_count)


        print("\n" + "=" * 80)
        print("COUVERTURE MENSUELLE")
        print("=" * 80)

        print(monthly_coverage)
        
        # ============================================================
        # SYNTHÈSE ESSENCE × SAISON
        # ============================================================

        season_order = [
            "Hiver",
            "Printemps",
            "Été",
            "Automne",
        ]

        all_species = sorted(
            {
                species
                for row in scene_rows
                for species in (
                    row["species_present"].split("; ")
                    if row["species_present"]
                    else []
                )
            }
        )

        if not species_df.empty:

            # --------------------------------------------------------
            # Nombre d'images contenant l'essence
            # --------------------------------------------------------

            presence_matrix = (
                species_df
                .groupby(
                    ["species", "season"]
                )["config"]
                .nunique()
                .unstack(fill_value=0)
            )

            presence_matrix = presence_matrix.reindex(
                columns=season_order,
                fill_value=0,
            )

            # --------------------------------------------------------
            # Nombre total de pixels
            # --------------------------------------------------------

            pixel_matrix = (
                species_df
                .groupby(
                    ["species", "season"]
                )["pixel_count"]
                .sum()
                .unstack(fill_value=0)
            )

            pixel_matrix = pixel_matrix.reindex(
                columns=season_order,
                fill_value=0,
            )

            # --------------------------------------------------------
            # Présence / absence
            # --------------------------------------------------------

            binary_matrix = (
                presence_matrix > 0
            ).astype(int)

            # --------------------------------------------------------
            # Nombre de saisons représentées
            # --------------------------------------------------------

            coverage = pd.DataFrame(
                index=presence_matrix.index
            )

            coverage[
                "n_seasons_represented"
            ] = binary_matrix.sum(axis=1)

            coverage[
                "missing_seasons"
            ] = binary_matrix.apply(
                lambda row: ", ".join(
                    season
                    for season in season_order
                    if row.get(season, 0) == 0
                ),
                axis=1,
            )

            coverage[
                "complete_phenology"
            ] = (
                coverage[
                    "n_seasons_represented"
                ] == 4
            )

        else:

            presence_matrix = pd.DataFrame()
            pixel_matrix = pd.DataFrame()
            binary_matrix = pd.DataFrame()
            coverage = pd.DataFrame()


        # ============================================================
        # EXPORTS
        # ============================================================

        scenes_df.to_csv(
            OUTPUT_DIR / "scenes_inventory.csv",
            index=False,
        )

        species_df.to_csv(
            OUTPUT_DIR / "species_by_scene.csv",
            index=False,
        )

        presence_matrix.to_csv(
            OUTPUT_DIR / "species_season_scene_count.csv"
        )

        pixel_matrix.to_csv(
            OUTPUT_DIR / "species_season_pixel_count.csv"
        )

        binary_matrix.to_csv(
            OUTPUT_DIR / "species_season_presence.csv"
        )

        coverage.to_csv(
            OUTPUT_DIR / "species_phenology_coverage.csv"
        )

        if not errors_df.empty:
            errors_df.to_csv(
                OUTPUT_DIR / "errors.csv",
                index=False,
            )


        # ============================================================
        # AFFICHAGE
        # ============================================================

        print("\n" + "=" * 70)
        print("COUVERTURE PHÉNOLOGIQUE")
        print("=" * 70)

        print(presence_matrix)

        print("\n" + "=" * 70)
        print("SAISONS MANQUANTES PAR ESSENCE")
        print("=" * 70)

        print(coverage)

        print(
            f"\nRésultats enregistrés dans : {OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()