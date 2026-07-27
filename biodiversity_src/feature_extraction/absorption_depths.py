from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


def find_nearest_band(wavelengths, target_nm):
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def _median_reflectance_in_window(cube, wavelengths, center_nm, window_nm=20):
    """
    Calcule la réflectance médiane autour d'une longueur d'onde cible.
    """

    idx = np.where(
        (wavelengths >= center_nm - window_nm) &
        (wavelengths <= center_nm + window_nm)
    )[0]

    if len(idx) == 0:
        idx = np.array([find_nearest_band(wavelengths, center_nm)])

    refl = np.nanmedian(cube[idx, :, :], axis=0).astype(np.float32)
    wl_eff = float(np.nanmean(wavelengths[idx]))

    return refl, wl_eff, idx


def compute_single_band_depth(
    cube,
    wavelengths,
    center_nm,
    left_nm=None,
    right_nm=None,
    shoulder_offset_nm=70,
    window_nm=20,
    eps=1e-8,
):
    """
    Calcule une carte de profondeur d'absorption autour d'une longueur d'onde donnée.

    BD = 1 - R_center / R_continuum

    Si left_nm et right_nm ne sont pas fournis, ils sont estimés automatiquement :
        left_nm  = center_nm - shoulder_offset_nm
        right_nm = center_nm + shoulder_offset_nm
    """

    if left_nm is None:
        left_nm = center_nm - shoulder_offset_nm

    if right_nm is None:
        right_nm = center_nm + shoulder_offset_nm

    R_left, wl_left_eff, idx_left = _median_reflectance_in_window(
        cube, wavelengths, left_nm, window_nm
    )

    R_center, wl_center_eff, idx_center = _median_reflectance_in_window(
        cube, wavelengths, center_nm, window_nm
    )

    R_right, wl_right_eff, idx_right = _median_reflectance_in_window(
        cube, wavelengths, right_nm, window_nm
    )

    if wl_right_eff <= wl_left_eff:
        raise ValueError(
            f"Continuum invalide pour {center_nm} nm : "
            f"left={wl_left_eff}, right={wl_right_eff}"
        )

    alpha = (wl_center_eff - wl_left_eff) / (wl_right_eff - wl_left_eff)
    continuum = R_left + alpha * (R_right - R_left)

    bd = np.full_like(R_center, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(R_center) &
        np.isfinite(continuum) &
        (np.abs(continuum) > eps)
    )

    bd[valid] = 1.0 - (R_center[valid] / continuum[valid])

    metadata = {
        "center_nm_target": float(center_nm),
        "left_nm_target": float(left_nm),
        "right_nm_target": float(right_nm),
        "center_nm_effective": wl_center_eff,
        "left_nm_effective": wl_left_eff,
        "right_nm_effective": wl_right_eff,
        "center_band_indices": idx_center.tolist(),
        "left_band_indices": idx_left.tolist(),
        "right_band_indices": idx_right.tolist(),
    }

    return bd.astype(np.float32), metadata


def compute_band_depth_cube(
    image_tif,
    wavelengths_csv,
    absorption_wavelengths,
    *,
    shoulder_offset_nm=70,
    window_nm=20,
    custom_shoulders=None,
):
    """
    Calcule un cube de profondeurs d'absorption à partir d'une image hyperspectrale.

    Parameters
    ----------
    image_tif : str | Path
        Chemin du cube hyperspectral GeoTIFF, shape rasterio = (bands, rows, cols).

    wavelengths_csv : str | Path
        CSV contenant la colonne 'wavelength_nm'.

    absorption_wavelengths : list[float]
        Liste des longueurs d'onde centrales pour lesquelles calculer les profondeurs.
        Exemple : [970, 1200, 1450, 1940, 2200]

    shoulder_offset_nm : float
        Distance automatique gauche/droite autour du centre si custom_shoulders=None.

    window_nm : float
        Demi-largeur de fenêtre utilisée pour calculer les médianes.

    custom_shoulders : dict | None
        Permet de définir les épaules manuellement :
        {
            970:  (900, 1070),
            1200: (1120, 1280),
            2200: (2130, 2280)
        }

    Returns
    -------
    band_depth_cube : np.ndarray
        Cube de profondeurs d'absorption :
        shape = (n_absorptions, rows, cols)

    feature_names : list[str]
        Noms des bandes :
        ["bd970", "bd1200", ...]

    metadata : dict
        Informations sur les bandes réellement utilisées.
    """

    image_tif = Path(image_tif)
    wavelengths_csv = Path(wavelengths_csv)

    with rasterio.open(image_tif) as src:
        cube = src.read().astype(np.float32)

    df = pd.read_csv(wavelengths_csv)

    if "wavelength_nm" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'wavelength_nm'.")

    wavelengths = df["wavelength_nm"].to_numpy(dtype=np.float32)

    if cube.shape[0] != len(wavelengths):
        raise ValueError(
            f"Incohérence : cube={cube.shape[0]} bandes, "
            f"CSV={len(wavelengths)} longueurs d'onde."
        )

    band_depth_maps = []
    feature_names = []
    metadata = {}

    for center_nm in absorption_wavelengths:
        center_nm = float(center_nm)

        if custom_shoulders is not None and center_nm in custom_shoulders:
            left_nm, right_nm = custom_shoulders[center_nm]
        else:
            left_nm = center_nm - shoulder_offset_nm
            right_nm = center_nm + shoulder_offset_nm

        bd_map, info = compute_single_band_depth(
            cube=cube,
            wavelengths=wavelengths,
            center_nm=center_nm,
            left_nm=left_nm,
            right_nm=right_nm,
            shoulder_offset_nm=shoulder_offset_nm,
            window_nm=window_nm,
        )

        name = f"bd{int(round(center_nm))}"

        band_depth_maps.append(bd_map)
        feature_names.append(name)
        metadata[name] = info

        print(
            f"{name}: "
            f"{info['left_nm_effective']:.1f} - "
            f"{info['center_nm_effective']:.1f} - "
            f"{info['right_nm_effective']:.1f} nm"
        )

    band_depth_cube = np.stack(band_depth_maps, axis=0).astype(np.float32)

    return band_depth_cube, feature_names, metadata