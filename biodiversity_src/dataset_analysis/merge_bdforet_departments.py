from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import pandas as pd



# ============================================================
# CONFIGURATION
# ============================================================

# Dossier contenant tous les départements décompressés
ROOT_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BD_ForetV2_France/BDForetV2_all_departments/extracted")

# Fichier de sortie
OUTPUT_GPKG = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BD_ForetV2_France/BDForetV2_all_departments/BDFORET_V2_FRANCE.gpkg")

OUTPUT_LAYER = "formation_vegetale"

DIAGNOSTIC_CSV = (
    OUTPUT_GPKG.parent
    / "bdforet_departments_diagnostic.csv"
)

READ_ERRORS_CSV = (
    OUTPUT_GPKG.parent
    / "bdforet_read_errors.csv"
)


# ============================================================
# DÉPARTEMENTS MÉTROPOLITAINS ATTENDUS
# ============================================================

EXPECTED_DEPARTMENTS = {
    f"{i:03d}"
    for i in range(1, 96)
    if i != 20
}

EXPECTED_DEPARTMENTS.update({
    "02A",
    "02B",
})


# ============================================================
# EXPRESSIONS RÉGULIÈRES
# ============================================================

# Exemple :
# BDFORET_2-0__SHP_LAMB93_D044_2022-04-01
DEPARTMENT_PATTERN = re.compile(
    r"_D(0\d{2}|02A|02B)_",
    re.IGNORECASE,
)

# Accepte :
# FORMATION_VEGETALE.shp
# FORMATION_VEGETALE_04.shp
# FORMATION_VEGETALE_2A.shp
FORMATION_PATTERN = re.compile(
    r"^FORMATION_VEGETALE(?:_[0-9]{2}|_2A|_2B)?\.shp$",
    re.IGNORECASE,
)


# ============================================================
# EXTRAIRE LE DÉPARTEMENT DEPUIS UN CHEMIN
# ============================================================

def extract_department_code(
    path: str | Path,
) -> str | None:

    path = Path(path)

    for part in reversed(path.parts):

        match = DEPARTMENT_PATTERN.search(part)

        if match:
            return match.group(1).upper()

    return None


# ============================================================
# LECTURE ROBUSTE D'UN SHAPEFILE
# ============================================================

def read_shapefile_with_encoding(
    shp: str | Path,
) -> tuple[gpd.GeoDataFrame, str]:

    shp = Path(shp)

    candidate_encodings = []

    # --------------------------------------------------------
    # Lire éventuellement le .cpg
    # --------------------------------------------------------

    cpg_path = shp.with_suffix(".cpg")

    if cpg_path.exists():

        try:

            cpg_value = (
                cpg_path
                .read_text(
                    encoding="ascii",
                    errors="ignore",
                )
                .strip()
            )

            if cpg_value:

                candidate_encodings.append(
                    cpg_value
                )

                print(
                    f"  -> .cpg : {cpg_value}"
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Encodages de secours
    # --------------------------------------------------------

    candidate_encodings.extend([
        None,
        "utf-8",
        "cp1252",
        "latin1",
    ])

    # Supprimer les doublons en conservant l'ordre
    unique_encodings = []

    for encoding in candidate_encodings:

        if encoding not in unique_encodings:
            unique_encodings.append(
                encoding
            )

    last_error = None

    # --------------------------------------------------------
    # Tentatives
    # --------------------------------------------------------

    for encoding in unique_encodings:

        try:

            if encoding is None:

                gdf = gpd.read_file(
                    shp
                )

                used_encoding = "default"

            else:

                gdf = gpd.read_file(
                    shp,
                    encoding=encoding,
                )

                used_encoding = encoding

            print(
                f"  -> lecture OK "
                f"(encodage : {used_encoding})"
            )

            return (
                gdf,
                used_encoding,
            )

        except UnicodeDecodeError as exc:

            last_error = exc

            print(
                f"  -> échec encodage "
                f"{encoding or 'default'}"
            )

        except Exception as exc:

            last_error = exc

            message = str(exc).lower()

            if (
                "codec" in message
                or "decode" in message
                or "encoding" in message
            ):

                print(
                    f"  -> échec encodage "
                    f"{encoding or 'default'} : "
                    f"{exc}"
                )

                continue

            raise

    raise RuntimeError(
        f"Impossible de lire : {shp}"
    ) from last_error


# ============================================================
# INVENTAIRE DES DOSSIERS DÉPARTEMENTAUX
# ============================================================

def find_department_dirs(
    root_dir: Path,
) -> dict[str, list[Path]]:

    department_dirs = {}

    for path in root_dir.iterdir():

        if not path.is_dir():
            continue

        department = (
            extract_department_code(path)
        )

        if department is None:
            continue

        department_dirs.setdefault(
            department,
            [],
        ).append(path)

    return department_dirs


# ============================================================
# RECHERCHER FORMATION_VEGETALE
# ============================================================

def find_formation_shapefiles(
    department_dirs: dict[str, list[Path]],
) -> dict[str, list[Path]]:

    result = {}

    for department, directories in (
        department_dirs.items()
    ):

        found = []

        for directory in directories:

            for path in directory.rglob(
                "*.shp"
            ):

                if FORMATION_PATTERN.match(
                    path.name
                ):
                    found.append(path)

        result[department] = sorted(
            set(found)
        )

    return result


# ============================================================
# DIAGNOSTIC
# ============================================================

def create_diagnostic(
    department_dirs,
    department_shapefiles,
) -> pd.DataFrame:

    rows = []

    for department in sorted(
        EXPECTED_DEPARTMENTS
    ):

        dirs = department_dirs.get(
            department,
            [],
        )

        files = department_shapefiles.get(
            department,
            [],
        )

        if not dirs:

            status = (
                "department_folder_missing"
            )

        elif not files:

            status = (
                "formation_vegetale_missing"
            )

        elif len(files) > 1:

            status = (
                "multiple_formation_vegetale"
            )

        else:

            status = "ok"

        rows.append(
            {
                "department": department,
                "status": status,
                "n_department_dirs": len(dirs),
                "n_formation_files": len(files),
                "department_dirs": " ; ".join(
                    str(x)
                    for x in dirs
                ),
                "formation_files": " ; ".join(
                    str(x)
                    for x in files
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not ROOT_DIR.exists():

        raise FileNotFoundError(
            f"Dossier introuvable : "
            f"{ROOT_DIR}"
        )

    # ========================================================
    # INVENTAIRE
    # ========================================================

    department_dirs = (
        find_department_dirs(
            ROOT_DIR
        )
    )

    department_shapefiles = (
        find_formation_shapefiles(
            department_dirs
        )
    )

    diagnostic_df = (
        create_diagnostic(
            department_dirs,
            department_shapefiles,
        )
    )

    diagnostic_df.to_csv(
        DIAGNOSTIC_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC")
    print("=" * 80)

    print(
        diagnostic_df[
            [
                "department",
                "status",
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # VÉRIFIER QUE LES 96 DÉPARTEMENTS SONT OK
    # ========================================================

    problems = diagnostic_df[
        diagnostic_df["status"] != "ok"
    ]

    if not problems.empty:

        print("\n" + "=" * 80)
        print(
            "ATTENTION : DÉPARTEMENTS "
            "NON VALIDES"
        )
        print("=" * 80)

        print(
            problems[
                [
                    "department",
                    "status",
                    "formation_files",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Tous les départements ne sont "
            "pas valides. Fusion interrompue."
        )


    print(
        f"\nTous les départements sont OK : "
        f"{len(diagnostic_df)}"
    )


    # ========================================================
    # CONSTRUIRE LA LISTE ORDONNÉE DES SHAPEFILES
    # ========================================================

    shapefiles = []

    for department in sorted(
        EXPECTED_DEPARTMENTS
    ):

        files = (
            department_shapefiles[
                department
            ]
        )

        shapefiles.append(
            files[0]
        )


    print(
        f"{len(shapefiles)} shapefiles "
        "seront fusionnés."
    )


    # ========================================================
    # SUPPRIMER L'ANCIEN GPKG
    # ========================================================

    OUTPUT_GPKG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_GPKG.exists():

        print(
            f"\nSuppression ancien GPKG : "
            f"{OUTPUT_GPKG}"
        )

        OUTPUT_GPKG.unlink()


    # ========================================================
    # ÉCRITURE PROGRESSIVE
    # ========================================================

    first_write = True

    reference_crs = None

    total_entities = 0

    departments_written = []

    read_errors = []


    for i, shp in enumerate(
        shapefiles,
        start=1,
    ):

        department = (
            extract_department_code(
                shp
            )
        )

        print("\n" + "=" * 80)

        print(
            f"[{i}/{len(shapefiles)}] "
            f"D{department}"
        )

        print(
            f"Fichier : {shp}"
        )


        # ====================================================
        # LECTURE
        # ====================================================

        try:

            gdf, encoding = (
                read_shapefile_with_encoding(
                    shp
                )
            )

        except Exception as exc:

            print(
                f"[ERREUR] D{department}"
            )

            print(exc)

            read_errors.append(
                {
                    "department": department,
                    "file": str(shp),
                    "error": str(exc),
                }
            )

            continue


        # ====================================================
        # COUCHE VIDE
        # ====================================================

        if gdf.empty:

            print(
                "  -> couche vide, ignorée"
            )

            continue


        # ====================================================
        # CRS
        # ====================================================

        if gdf.crs is None:

            raise ValueError(
                f"CRS absent : {shp}"
            )


        if reference_crs is None:

            reference_crs = gdf.crs

            print(
                f"  -> CRS de référence : "
                f"{reference_crs}"
            )

        elif gdf.crs != reference_crs:

            print(
                f"  -> reprojection "
                f"{gdf.crs} "
                f"vers "
                f"{reference_crs}"
            )

            gdf = gdf.to_crs(
                reference_crs
            )


        # ====================================================
        # NETTOYAGE GÉOMÉTRIES
        # ====================================================

        before = len(gdf)

        gdf = gdf[
            gdf.geometry.notna()
            & ~gdf.geometry.is_empty
        ].copy()

        removed = (
            before
            - len(gdf)
        )

        if removed:

            print(
                f"  -> géométries vides "
                f"supprimées : {removed}"
            )


        if gdf.empty:

            print(
                "  -> aucune géométrie valide"
            )

            continue


        # ====================================================
        # INFORMATIONS SOURCE
        # ====================================================

        gdf[
            "source_department"
        ] = department

        gdf[
            "source_file"
        ] = shp.name

        gdf[
            "source_encoding"
        ] = encoding


        # ====================================================
        # COMPTE
        # ====================================================

        n_entities = len(gdf)

        total_entities += (
            n_entities
        )

        print(
            f"  -> entités : "
            f"{n_entities:,}"
        )


        # ====================================================
        # EXPORT
        # ====================================================

        if first_write:

            print(
                "  -> création du GeoPackage"
            )

            gdf.to_file(
                OUTPUT_GPKG,
                layer=OUTPUT_LAYER,
                driver="GPKG",
            )

            first_write = False

        else:

            print(
                "  -> ajout au GeoPackage"
            )

            gdf.to_file(
                OUTPUT_GPKG,
                layer=OUTPUT_LAYER,
                driver="GPKG",
                mode="a",
            )


        departments_written.append(
            department
        )


        # ====================================================
        # LIBÉRER LA MÉMOIRE
        # ====================================================

        del gdf


        print(
            f"  -> total écrit jusqu'ici : "
            f"{total_entities:,}"
        )


    # ========================================================
    # ERREURS DE LECTURE
    # ========================================================

    if read_errors:

        pd.DataFrame(
            read_errors
        ).to_csv(
            READ_ERRORS_CSV,
            index=False,
        )

        print(
            f"\nErreurs enregistrées dans : "
            f"{READ_ERRORS_CSV}"
        )


    # ========================================================
    # VÉRIFICATION FINALE
    # ========================================================

    if first_write:

        raise RuntimeError(
            "Aucune donnée n'a été écrite."
        )


    # ========================================================
    # BILAN
    # ========================================================

    print("\n" + "=" * 80)
    print("TERMINÉ")
    print("=" * 80)

    print(
        f"GeoPackage : "
        f"{OUTPUT_GPKG}"
    )

    print(
        f"Couche : "
        f"{OUTPUT_LAYER}"
    )

    print(
        f"CRS : "
        f"{reference_crs}"
    )

    print(
        f"Départements écrits : "
        f"{len(departments_written)}"
    )

    print(
        f"Entités écrites : "
        f"{total_entities:,}"
    )

    print(
        "\nDépartements :"
    )

    print(
        ", ".join(
            f"D{dep}"
            for dep in departments_written
        )
    )


if __name__ == "__main__":
    main()