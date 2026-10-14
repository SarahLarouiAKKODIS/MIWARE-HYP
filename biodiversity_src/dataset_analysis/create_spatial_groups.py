from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from collections import defaultdict
import json

import pandas as pd
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, shape

from utils import load_config


# ============================================================
# CONFIGURATION
# ============================================================

CONFIGS_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess")

OUTPUT_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/dataset_analysis/spatial_groups")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Surface minimale de chevauchement, dans le CRS utilisé pour
# la comparaison.
#
# 0 = le moindre chevauchement suffit à regrouper les images.
MIN_OVERLAP_AREA = 0.0


# ============================================================
# RÉCUPÉRER L'EMPRISE EXACTE D'UN RASTER
# ============================================================

def get_raster_footprint(
    raster_path: str | Path,
    dst_crs: str = "EPSG:4326",
) -> Polygon:
    """
    Retourne l'emprise réelle d'un raster sous forme de polygone.

    Contrairement à src.bounds, cette méthode tient compte
    d'une éventuelle rotation du raster.

    Le polygone est reprojeté vers dst_crs.
    """

    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster introuvable : {raster_path}"
        )

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError(
                f"Le raster n'a pas de CRS : {raster_path}"
            )

        transform = src.transform

        width = src.width
        height = src.height

        # Coins du raster
        corners = [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
            transform * (0, 0),
        ]

        polygon = Polygon(corners)

        geom = transform_geom(
            src.crs,
            dst_crs,
            polygon.__geo_interface__,
        )

    return shape(geom)


# ============================================================
# TEST DE CHEVAUCHEMENT
# ============================================================

def polygons_overlap(
    polygon1: Polygon,
    polygon2: Polygon,
    min_overlap_area: float = 0.0,
) -> bool:
    """
    Vérifie si deux images ont une intersection surfacique.

    Un simple contact par le bord n'est pas considéré
    comme un chevauchement.
    """

    if not polygon1.intersects(polygon2):
        return False

    intersection = polygon1.intersection(polygon2)

    if intersection.is_empty:
        return False

    # Évite de considérer un simple contact comme un overlap
    if intersection.area <= min_overlap_area:
        return False

    return True


# ============================================================
# CONSTRUCTION DU GRAPHE DE CHEVAUCHEMENT
# ============================================================

def build_overlap_graph(
    footprints: dict[str, Polygon],
    min_overlap_area: float = 0.0,
) -> dict[str, set[str]]:
    """
    Construit un graphe où deux images sont connectées
    lorsqu'elles se chevauchent.
    """

    names = list(footprints.keys())

    graph = {
        name: set()
        for name in names
    }

    for i in range(len(names)):

        name1 = names[i]
        geom1 = footprints[name1]

        for j in range(i + 1, len(names)):

            name2 = names[j]
            geom2 = footprints[name2]

            if polygons_overlap(
                geom1,
                geom2,
                min_overlap_area=min_overlap_area,
            ):

                graph[name1].add(name2)
                graph[name2].add(name1)

    return graph


# ============================================================
# GROUPES = COMPOSANTES CONNEXES
# ============================================================

def get_spatial_groups(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    """
    Construit les groupes de telle sorte qu'aucun chevauchement
    n'existe entre deux groupes différents.
    """

    visited = set()
    groups = []

    for image_name in graph:

        if image_name in visited:
            continue

        stack = [image_name]
        group = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            group.append(current)

            stack.extend(
                graph[current] - visited
            )

        groups.append(sorted(group))

    # Les plus gros groupes d'abord
    groups.sort(
        key=len,
        reverse=True,
    )

    return groups


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
        f"\n{len(config_paths)} configuration(s) trouvée(s)."
    )

    footprints = {}

    image_infos = {}

    errors = []

    # ========================================================
    # LECTURE DES CONFIGS
    # ========================================================

    for i, config_path in enumerate(
        config_paths,
        start=1,
    ):

        print(
            f"[{i}/{len(config_paths)}] "
            f"{config_path.name}"
        )

        try:

            config = load_config(config_path)

            image_path = (
                Path(config["Path_data"])
                / Path(config["image_hyp"])
            )

            footprint = get_raster_footprint(
                image_path,
                dst_crs="EPSG:4326",
            )

            image_name = config_path.stem

            footprints[image_name] = footprint

            image_infos[image_name] = {
                "config": str(config_path),
                "image": str(image_path),
            }

        except Exception as exc:

            print(
                f"  [ERREUR] {exc}"
            )

            errors.append({
                "config": config_path.name,
                "error": str(exc),
            })


    # ========================================================
    # CHEVAUCHEMENTS
    # ========================================================

    print("\nConstruction des chevauchements...")

    graph = build_overlap_graph(
        footprints,
        min_overlap_area=MIN_OVERLAP_AREA,
    )


    # ========================================================
    # GROUPES
    # ========================================================

    groups = get_spatial_groups(graph)


    print("\n" + "=" * 70)
    print("GROUPES SPATIAUX")
    print("=" * 70)

    rows = []

    for group_id, group in enumerate(
        groups,
        start=1,
    ):

        print(
            f"\nGroupe {group_id} "
            f"({len(group)} image(s))"
        )

        for image_name in group:

            print(
                f"  - {image_name}"
            )

            rows.append({
                "group_id": group_id,
                "image_name": image_name,
                "config_path": image_infos[image_name]["config"],
                "image_path": image_infos[image_name]["image"],
                "n_overlaps": len(graph[image_name]),
            })


    # ========================================================
    # EXPORT CSV
    # ========================================================

    groups_df = pd.DataFrame(rows)

    groups_df.to_csv(
        OUTPUT_DIR / "spatial_groups.csv",
        index=False,
    )


    # ========================================================
    # EXPORT JSON
    # ========================================================

    groups_json = {
        f"group_{i}": group
        for i, group in enumerate(
            groups,
            start=1,
        )
    }

    with open(
        OUTPUT_DIR / "spatial_groups.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            groups_json,
            f,
            indent=4,
            ensure_ascii=False,
        )


    # ========================================================
    # LISTE DES CHEVAUCHEMENTS
    # ========================================================

    overlap_rows = []

    already_seen = set()

    for image1, neighbours in graph.items():

        for image2 in neighbours:

            pair = tuple(
                sorted([image1, image2])
            )

            if pair in already_seen:
                continue

            already_seen.add(pair)

            intersection = (
                footprints[image1]
                .intersection(footprints[image2])
            )

            overlap_rows.append({
                "image_1": image1,
                "image_2": image2,
                "intersection_area_deg2": intersection.area,
            })

    overlaps_df = pd.DataFrame(overlap_rows)

    overlaps_df.to_csv(
        OUTPUT_DIR / "image_overlaps.csv",
        index=False,
    )


    # ========================================================
    # ERREURS
    # ========================================================

    if errors:

        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )


    # ========================================================
    # BILAN
    # ========================================================

    print("\n" + "=" * 70)
    print("BILAN")
    print("=" * 70)

    print(
        f"Images analysées : {len(footprints)}"
    )

    print(
        f"Groupes spatiaux : {len(groups)}"
    )

    print(
        f"Images isolées   : "
        f"{sum(len(g) == 1 for g in groups)}"
    )

    print(
        f"Paires en chevauchement : "
        f"{len(overlap_rows)}"
    )

    print(
        f"\nRésultats : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()