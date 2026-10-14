from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from rasterio.warp import transform_bounds

def rasterize_forest_classes(
    vector_path: str | Path,
    reference_raster: str | Path,
    output_mask: str | Path,
    class_map: dict[str, int],
    class_field: str,
    *,
    layer_name: str | None = None,
    background_value: int = 0,
    nodata_value: int = 255,
    dtype: str = "uint8",
    all_touched: bool = False,
    compress: str | None = None,
    tiled: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Rasterise un vecteur de classes forestières sur la grille d'un raster de référence.

    Parameters
    ----------
    vector_path : str | Path
        Chemin du fichier vecteur contenant les classes forestières.
    reference_raster : str | Path
        Raster de référence utilisé pour récupérer CRS, transform, shape.
    output_mask : str | Path
        Chemin du masque raster de sortie.
    class_map : dict[str, int]
        Dictionnaire {nom_classe: id_classe}.
    class_field : str
        Nom du champ attributaire contenant la classe.
    layer_name : str | None
        Nom de couche si le fichier vecteur est un GPKG multi-couches.
    background_value : int
        Valeur des pixels sans classe.
    nodata_value : int
        Valeur nodata du raster de sortie.
    dtype : str
        Type du raster de sortie. Recommandé : "uint8".
    all_touched : bool
        Si True, rasterise tous les pixels touchés par les géométries.
        Si False, rasterise seulement les pixels dont le centre est couvert.
    compress : str | None
        Compression GeoTIFF. None = pas de compression.
    tiled : bool
        Active ou non le tiling GeoTIFF.
    verbose : bool
        Affiche les informations de sortie.

    Returns
    -------
    dict
        Résumé de la rasterisation.
    """

    vector_path = Path(vector_path)
    reference_raster = Path(reference_raster)
    output_mask = Path(output_mask)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    if not vector_path.exists():
        raise FileNotFoundError(f"Vecteur introuvable : {vector_path}")

    if not reference_raster.exists():
        raise FileNotFoundError(f"Raster de référence introuvable : {reference_raster}")

    # -----------------------------
    # Lire la grille de référence
    # -----------------------------
    with rasterio.open(reference_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds
        ref_profile = ref.profile.copy()

    if ref_crs is None:
        raise ValueError("Le raster de référence n'a pas de CRS.")

    # ============================================================
    # LECTURE DU VECTEUR LIMITÉE À L'EMPRISE DU RASTER
    # ============================================================

    # Lire uniquement la structure afin de connaître le CRS
    if layer_name is not None:
        gdf_info = gpd.read_file(
            vector_path,
            layer=layer_name,
            rows=1,
        )
    else:
        gdf_info = gpd.read_file(
            vector_path,
            rows=1,
        )

    vector_crs = gdf_info.crs

    if vector_crs is None:
        raise ValueError("Le vecteur n'a pas de CRS.")

    if verbose:
        print("CRS raster  :", ref_crs)
        print("CRS vecteur :", vector_crs)


    # ------------------------------------------------------------
    # Transformer l'emprise du raster vers le CRS du vecteur
    # ------------------------------------------------------------

    if vector_crs != ref_crs:

        vector_bounds = transform_bounds(
            ref_crs,
            vector_crs,
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top,
            densify_pts=21,
        )

    else:

        vector_bounds = (
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top,
        )


    if verbose:
        print("Emprise raster :", ref_bounds)
        print("Emprise pour lecture vecteur :", vector_bounds)


    # ------------------------------------------------------------
    # Lire UNIQUEMENT les polygones de cette emprise
    # ------------------------------------------------------------

    if layer_name is not None:

        gdf = gpd.read_file(
            vector_path,
            layer=layer_name,
            bbox=vector_bounds,
        )

    else:

        gdf = gpd.read_file(
            vector_path,
            bbox=vector_bounds,
        )


    if gdf.empty:
        raise ValueError(
            "Aucune donnée forestière n'intersecte "
            "l'emprise du raster de référence."
        )


    if verbose:
        print(
            f"Polygones BD Forêt chargés : {len(gdf):,}"
        )


    # ------------------------------------------------------------
    # Vérification du champ classe
    # ------------------------------------------------------------

    if class_field not in gdf.columns:

        raise ValueError(
            f"Champ '{class_field}' introuvable dans le vecteur. "
            f"Champs disponibles : {list(gdf.columns)}"
        )


    # ------------------------------------------------------------
    # Reprojection des polygones vers la grille EnMAP
    # ------------------------------------------------------------

    if gdf.crs != ref_crs:

        gdf = gdf.to_crs(ref_crs)


    # ------------------------------------------------------------
    # Clip précis
    # ------------------------------------------------------------

    bbox = box(
        ref_bounds.left,
        ref_bounds.bottom,
        ref_bounds.right,
        ref_bounds.top,
    )

    gdf = gdf[gdf.intersects(bbox)].copy()

    # Nettoyage géométries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("Aucune géométrie valide n'intersecte le raster de référence.")

    classes = gdf[class_field].dropna().unique()

    print(classes) 
    # -----------------------------
    # Construire les couples (géométrie, valeur)
    # -----------------------------
    shapes = []
    ignored_classes = {}

    for geom, cls in zip(gdf.geometry, gdf[class_field].astype(str)):
        if cls in class_map:
            shapes.append((geom, int(class_map[cls])))
        else:
            ignored_classes[cls] = ignored_classes.get(cls, 0) + 1

    if not shapes:
        raise ValueError(
            "Aucune classe du vecteur ne correspond à class_map. "
            f"Classes ignorées : {ignored_classes}"
        )
    

  

    # -----------------------------
    # Rasterisation
    # -----------------------------
    mask = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=background_value,
        dtype=dtype,
        all_touched=all_touched,
    )

    # -----------------------------
    # Profil de sortie
    # -----------------------------
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        height=ref_shape[0],
        width=ref_shape[1],
        count=1,
        dtype=dtype,
        crs=ref_crs,
        transform=ref_transform,
        nodata=nodata_value,
        compress=compress,
        tiled=tiled,
    )

    with rasterio.open(output_mask, "w", **profile) as dst:
        dst.write(mask, 1)

    unique_values = np.unique(mask)

    if verbose:
        print("Masque écrit :", output_mask)
        print("Shape        :", mask.shape)
        print("Classes      :", unique_values)
        if ignored_classes:
            print("Classes ignorées :", ignored_classes)

    return {
        "output_mask": output_mask,
        "shape": mask.shape,
        "unique_values": unique_values.tolist(),
        "n_shapes_rasterized": len(shapes),
        "ignored_classes": ignored_classes,
        "class_field": class_field,
        "class_map": class_map,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from pathlib import Path
    from utils import load_config

    #config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/salsigne_3.json"

    # ============================================================
    # DOSSIER CONTENANT LES CONFIGS
    # ============================================================

    configs_dir = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/ToProcess")

    config_paths = sorted(configs_dir.glob("*.json"))

    if not config_paths:
        raise FileNotFoundError(
            f"Aucun fichier JSON trouvé dans : {configs_dir}"
        )

    print(f"\n{len(config_paths)} fichier(s) de configuration trouvé(s).")


    # ============================================================
    # TRAITEMENT DE CHAQUE CONFIG
    # ============================================================

    for i, config_path in enumerate(config_paths, start=1):

        print("\n" + "=" * 70)
        print(
            f"[{i}/{len(config_paths)}] "
            f"Traitement : {config_path.name}"
        )
        print("=" * 70)

    
        config = load_config(config_path)
        config_forest = load_config(config["config_forest_path"])

        Path_res = Path(config["Path_res"]) / "Forest"
        Path_res.mkdir(parents=True, exist_ok=True)

        image_hyperspectral = Path(config["Path_data"]) / Path(config["image_hyp"])
       
        summary = rasterize_forest_classes(
            vector_path=config["forest_species_mask"],
            reference_raster=image_hyperspectral,
            output_mask=Path_res / "mask_foret_classes.tif",
            class_map=config_forest["CLASS_MAP"],
            class_field=config_forest["CLASS_FIELD"],
            layer_name="formation_vegetale",
            background_value=0,
            nodata_value=255,
            all_touched=False,
            compress=None,
            tiled=False,
        )