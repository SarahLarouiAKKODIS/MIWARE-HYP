from pathlib import Path
import numpy as np
import rasterio

from pathlib import Path
import numpy as np
import rasterio


from pathlib import Path
import numpy as np
import rasterio


def concatenate_similarity_best_maps(
    input_dir,
    output_path,
    mode="ace",   # "ace", "sam", "combo", "all"
):
    """
    Concatène les nouvelles cartes de similarité :
      - score ACE/SAM/COMBO
      - espèce associée au meilleur score

    Parameters
    ----------
    input_dir : str | Path
        Dossier contenant les cartes :
        ace_max.tif, ace_best_species.tif,
        sam_min.tif, sam_best_species.tif,
        combo_max.tif, combo_best_species.tif.

    output_path : str | Path
        Chemin du raster multibande de sortie.

    mode : str
        "ace"   -> ace_max + ace_best_species
        "sam"   -> sam_min + sam_best_species
        "combo" -> combo_max + combo_best_species
        "all"   -> les 6 cartes

    Returns
    -------
    stack : np.ndarray
        Cube de forme (n_features, rows, cols)

    feature_names : list[str]
        Noms des bandes concaténées.
    """

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files_by_mode = {
        "ace": [
            "ace_max.tif",
            "ace_best_species.tif",
        ],
        "sam": [
            "sam_min.tif",
            "sam_best_species.tif",
        ],
        "combo": [
            "combo_max.tif",
            "combo_best_species.tif",
        ],
        "all": [
            "ace_max.tif",
            "ace_best_species.tif",
            "sam_min.tif",
            "sam_best_species.tif",
            "combo_max.tif",
            "combo_best_species.tif",
        ],
    }

    if mode not in files_by_mode:
        raise ValueError("mode doit être 'ace', 'sam', 'combo' ou 'all'.")

    tif_files = [input_dir / fname for fname in files_by_mode[mode]]

    missing = [p for p in tif_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers manquants :\n" + "\n".join(str(p) for p in missing)
        )

    stack = []
    feature_names = []

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    for tif in tif_files:
        with rasterio.open(tif) as src:

            if src.height != height or src.width != width:
                raise ValueError(f"Dimensions incompatibles pour {tif.name}")

            if src.transform != transform:
                raise ValueError(f"Géoréférencement différent pour {tif.name}")

            if src.crs != crs:
                raise ValueError(f"CRS différent pour {tif.name}")

            stack.append(src.read(1).astype(np.float32))
            feature_names.append(tif.stem)

    stack = np.stack(stack, axis=0).astype(np.float32)

    profile.update(
        count=stack.shape[0],
        dtype="float32",
        compress="lzw",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)

    print(f"Mode : {mode}")
    print(f"Cube sauvegardé : {output_path}")
    print(f"Dimensions : {stack.shape}")
    print("Features :", feature_names)

    return stack, feature_names

def concatenate_ace_maps(input_dir):
    """
    Concatène toutes les images GeoTIFF commençant par 'ace'
    en un cube (n_images, rows, cols).

    Parameters
    ----------
    input_dir : str | Path
        Dossier contenant les cartes ACE.

    Returns
    -------
    cube : np.ndarray
        Cube de dimensions (n_ace_maps, rows, cols).
    feature_names : list[str]
        Noms des cartes concaténées.
    profile : dict
        Profil rasterio de la première image.
    """

    input_dir = Path(input_dir)

    tif_files = sorted(input_dir.glob("ace*.tif"))

    if len(tif_files) == 0:
        raise ValueError(f"Aucune image 'ace*.tif' trouvée dans {input_dir}")

    cube = []
    feature_names = []

    for tif in tif_files:
        with rasterio.open(tif) as src:
            cube.append(src.read(1).astype(np.float32))

            if len(cube) == 1:
                profile = src.profile.copy()

        feature_names.append(tif.stem)

    cube = np.stack(cube, axis=0)

    print(f"{len(feature_names)} cartes ACE concaténées.")
    print("Dimensions du cube :", cube.shape)

    return cube, feature_names, profile


def concatenate_tifs(
    input_dir,
    output_path,
):
    """
    Concatène toutes les images .tif d'un dossier en un raster multibande.

    Parameters
    ----------
    input_dir : str | Path
        Dossier contenant les images .tif.
    output_path : str | Path
        Chemin du raster de sortie.

    Returns
    -------
    np.ndarray
        Cube concaténé de forme (n_images, rows, cols).
    """

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    tif_files = sorted(input_dir.glob("*.tif"))

    if len(tif_files) == 0:
        raise ValueError(f"Aucune image .tif trouvée dans {input_dir}")

    print(f"{len(tif_files)} images trouvées.")

    stack = []
    feature_names = []

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    for tif in tif_files:

        #print(f"Lecture : {tif.name}")

        with rasterio.open(tif) as src:

            if src.height != height or src.width != width:
                raise ValueError(
                    f"Dimensions incompatibles pour {tif.name}"
                )

            if src.transform != transform:
                raise ValueError(
                    f"Géoréférencement différent pour {tif.name}"
                )

            if src.crs != crs:
                raise ValueError(
                    f"CRS différent pour {tif.name}"
                )

            stack.append(src.read(1).astype(np.float32))
            feature_names.append(tif.name[:-4])

    stack = np.stack(stack, axis=0)

    profile.update(
        count=stack.shape[0],
        dtype="float32",
        compress="lzw",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)

    print(f"\nCube sauvegardé : {output_path}")
    print(f"Dimensions : {stack.shape}")

    return stack, feature_names



import numpy as np
import rasterio


def cube_to_dataset(feature_cube, label_mask, valid_mask_path, ignore_labels=(-1,)):
    """
    Transforme un cube de features (bands, rows, cols)
    en dataset (n_pixels, n_features) en utilisant un masque de validité.

    Parameters
    ----------
    feature_cube : ndarray
        Cube de forme (n_features, rows, cols)

    label_mask : ndarray
        Masque des labels (rows, cols)

    valid_mask_path : str | Path
        Chemin vers enmap_valid_mask.tif
        (True ou 1 = pixel valide)

    ignore_labels : tuple
        Labels à ignorer.

    Returns
    -------
    X : ndarray
        (n_pixels, n_features)

    y : ndarray
        (n_pixels,)
    """

    with rasterio.open(valid_mask_path) as src:
        valid = src.read(1).astype(bool)

    valid &= np.isfinite(feature_cube).all(axis=0)

    for lab in ignore_labels:
        valid &= (label_mask != lab)

    X = feature_cube[:, valid].T.astype(np.float32)
    y = label_mask[valid].astype(np.int32)

    return X, y