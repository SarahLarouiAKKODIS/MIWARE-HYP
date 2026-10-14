
#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import date

CONFIG_DIR = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess"
)

RESULTS_DIR = Path(
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results"
)

FOREST_SPECIES_MASK = (
    "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/"
    "BD_ForetV2_France/BDForetV2_all_departments/"
    "BDFORET_V2_FRANCE.gpkg"
)

FOREST_CLASS_PATH = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/class_map.json"

# Correspondance entre les clés JSON et les suffixes des fichiers
FILE_SUFFIXES = {
    "image_hyp": "-SPECTRAL_IMAGE.TIF",
    "xml_path": "-METADATA.XML",
    "Cloud_mask": "-QL_QUALITY_CLOUD.TIF",
    "Haze_mask": "-QL_QUALITY_HAZE.TIF",
    "Cirrus_mask": "-QL_QUALITY_CIRRUS.TIF",
    "CloudShadow_mask": "-QL_QUALITY_CLOUDSHADOW.TIF",
    "Snow_mask": "-QL_QUALITY_SNOW.TIF",
    "TestFlags_mask": "-QL_QUALITY_TESTFLAGS.TIF",
    "QualityClasses_mask": "-QL_QUALITY_CLASSES.TIF",
}

def extract_acquisition_date(xml_path):
    """
    Extrait la date d'acquisition depuis les métadonnées EnMAP.

    Chemin XML :
    base/temporalCoverage/startTime

    Retour :
    str au format YYYY-MM-DD
    """

    xml_path = Path(xml_path)

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Fichier XML introuvable : {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    start_time = root.findtext(
        "./base/temporalCoverage/startTime"
    )

    if not start_time:
        raise ValueError(
            f"Date d'acquisition introuvable dans : {xml_path}"
        )

    acquisition_date = date.fromisoformat(
        start_time[:10]
    )

    return acquisition_date.isoformat()


def generate_json(folder_path):
    folder = Path(folder_path).expanduser().resolve()

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Dossier introuvable : {folder}"
        )

    # Initialisation du JSON
    data = {
        "Path_data": str(folder),
    }

    # Liste des fichiers présents dans le dossier
    files = [f for f in folder.iterdir() if f.is_file()]

    # Recherche automatique des fichiers
    for key, suffix in FILE_SUFFIXES.items():

        matches = [
            f for f in files
            if f.name.upper().endswith(suffix.upper())
        ]

        if len(matches) == 1:
            data[key] = matches[0].name

        elif len(matches) == 0:
            print(f"[WARNING] Fichier manquant : {suffix}")
            data[key] = ""

        else:
            raise ValueError(
                f"Plusieurs fichiers correspondent à {suffix} : "
                f"{[f.name for f in matches]}"
            )


    # Extraction automatique de la date d'acquisition
    if data["xml_path"]:

        xml_file = folder / data["xml_path"]

        acquisition_date = extract_acquisition_date(
            xml_file
        )

        data["acquisition_date"] = acquisition_date

        print(
            f"Date d'acquisition : {acquisition_date}"
        )

    else:
        raise FileNotFoundError(
            "Impossible de récupérer la date : "
            "fichier METADATA.XML manquant."
        )
    # Dossier de résultats propre à chaque image
    results_path = RESULTS_DIR / folder.name

    # Création des dossiers si nécessaire
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # Paramètres complémentaires
    data.update({
        "forest_species_mask": FOREST_SPECIES_MASK,
        "config_forest_path": FOREST_CLASS_PATH,
        "crop": False,
        "Site_coords": [],
        "Path_res": str(results_path),
    })

    # JSON enregistré dans configs/ToProcess
    output_json = CONFIG_DIR / f"{folder.name}.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nJSON généré : {output_json}")
    print(f"Dossier de résultats : {results_path}")

    return output_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Génération automatique du JSON hyperspectral"
    )

    parser.add_argument(
        "folder_path",
        help="Chemin du dossier contenant les données hyperspectrales"
    )

    args = parser.parse_args()

    generate_json(args.folder_path)
