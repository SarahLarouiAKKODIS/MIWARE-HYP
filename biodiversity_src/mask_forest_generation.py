from __future__ import annotations

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box


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

    # -----------------------------
    # Lire le vecteur
    # -----------------------------
    if layer_name is not None:
        gdf = gpd.read_file(vector_path, layer=layer_name)
    else:
        gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError(f"Le fichier vecteur est vide : {vector_path}")

    if class_field not in gdf.columns:
        raise ValueError(
            f"Champ '{class_field}' introuvable dans le vecteur. "
            f"Champs disponibles : {list(gdf.columns)}"
        )

    if gdf.crs is None:
        raise ValueError("Le vecteur n'a pas de CRS.")

    # -----------------------------
    # Reprojection vers le CRS raster
    # -----------------------------
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    # -----------------------------
    # Clip sur l'emprise du raster
    # -----------------------------
    bbox = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
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
    from utils.commun_functions import load_config

    config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/salsigne_3.json"

    config = load_config(config_path)
    config_forest = load_config(config["config_forest_path"])

    Path_res = Path(config["Path_res"]) / "Forest"
    Path_res.mkdir(parents=True, exist_ok=True)

    summary = rasterize_forest_classes(
        vector_path=config["forest_species_mask"],
        reference_raster=config["image_hyp"],
        output_mask=Path_res / "mask_foret_classes.tif",
        class_map=config_forest["CLASS_MAP"],
        class_field=config_forest["CLASS_FIELD"],
        layer_name=None,
        background_value=0,
        nodata_value=255,
        all_touched=False,
        compress=None,
        tiled=False,
    )