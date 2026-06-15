import numpy as np

# ============================================================
# 6) EXTRACTION DES PIXELS
# ============================================================


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