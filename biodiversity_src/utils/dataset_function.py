import numpy as np

# ============================================================
# 6) EXTRACTION DES PIXELS
# ============================================================

def build_indices_cube(
    indices_dict,
    target_mask,
    valid_pixels_mask=None,
    ignore_labels=(0,),
    fill_value=np.nan,
):
    """
    Construit un cube de caractéristiques à partir des indices spectraux.

    Parameters
    ----------
    indices_dict : dict
        Dictionnaire des indices spectraux.

    target_mask : ndarray
        Masque des classes.

    valid_pixels_mask : ndarray | None
        Masque des pixels valides.

    ignore_labels : tuple
        Labels à ignorer.

    fill_value : float
        Valeur utilisée pour les pixels invalides.

    Returns
    -------
    X : ndarray
        Cube de features de forme
        (n_features, rows, cols)

    y : ndarray
        Masque des labels de forme
        (rows, cols)

    feature_names : list[str]
        Nom des bandes du cube X.
    """

    feature_names = [
        "ndvi",
        "ndre",
        "ndwi",
        "pri",
        "ari",
        "evi",
        "nbr",
    ]

    rows, cols = target_mask.shape

    valid = np.ones((rows, cols), dtype=bool)

    if valid_pixels_mask is not None:
        valid &= valid_pixels_mask

    for name in feature_names:
        valid &= np.isfinite(indices_dict[name])

    for lab in ignore_labels:
        valid &= (target_mask != lab)

    X = np.full(
        (len(feature_names), rows, cols),
        fill_value,
        dtype=np.float32,
    )

    for i, name in enumerate(feature_names):
        X[i, valid] = indices_dict[name][valid].astype(np.float32)

    y = np.full(
        target_mask.shape,
        fill_value=-1,
        dtype=np.int32,
    )

    y[valid] = target_mask[valid]

    return X, y, feature_names



def extract_pixels(indices_dict, target_mask, valid_pixels_mask=None, ignore_labels=(0,)):
    if valid_pixels_mask is None:
        valid_pixels_mask = np.ones(target_mask.shape, dtype=bool)

    valid = (
        valid_pixels_mask &
        np.isfinite(indices_dict["ndvi"]) &
        np.isfinite(indices_dict["ndre"]) &
        np.isfinite(indices_dict["ndwi"]) &
        np.isfinite(indices_dict["pri"]) &
        np.isfinite(indices_dict["ari"]) &
        np.isfinite(indices_dict["evi"]) &
        np.isfinite(indices_dict["nbr"]) 
    )

    for lab in ignore_labels:
        valid &= (target_mask != lab)

    # coordonnées pixels
    rows, cols = np.where(valid)

    # features
    X = np.stack([
        indices_dict["ndvi"][valid],
        indices_dict["ndre"][valid],
        indices_dict["ndwi"][valid],
        indices_dict["pri"][valid],
        indices_dict["ari"][valid],
        indices_dict["evi"][valid],
        indices_dict["nbr"][valid],
    ], axis=1).astype(np.float32)

    # labels
    y = target_mask[valid].astype(np.int32)

    return X, y, rows, cols