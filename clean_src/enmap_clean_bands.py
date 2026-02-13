import rasterio
import numpy as np
import pandas as pd

def clean_bands_enmap_from_csv(
    img_path: str,
    bands_csv: str,
    output_path: str,
    output_bands_csv: str | None = None,   # <- NEW: écrit un CSV corrigé
    exclude_ranges_nm=None,
    drop_edges=(0, 0),
    band_id_is_one_based: bool = False,
    use_fwhm_margin: bool = False,
    fwhm_factor: float = 0.5,
    dtype="float32",
    csv_band_id_is_one_based_out: bool = True,  # <- NEW: band_id du CSV de sortie
):
    """
    Retire des bandes (atmosphère / bords) d'une image hyperspectrale en s'appuyant sur un CSV
    contenant les longueurs d'onde réelles.

    CSV attendu (entrée): band_id, wavelength_nm, fwhm_nm, gain, offset (au minimum band_id + wavelength_nm).

    NEW:
      - si output_bands_csv est fourni, écrit un CSV "corrigé" ne contenant QUE les bandes conservées,
        avec band_id ré-indexé pour correspondre au GeoTIFF nettoyé.
    """

    if exclude_ranges_nm is None:
        exclude_ranges_nm = [
            (1340, 1460),  # H2O ~1.4 µm
            (1800, 1960),  # H2O ~1.9 µm
            (0, 420),      # bord VNIR (optionnel)
            (2450, 2600),  # bord SWIR (optionnel)
        ]

    # --- Lire image ---
    with rasterio.open(img_path) as src:
        img = src.read().astype(dtype)
        profile = src.profile

    nbands = img.shape[0]

    # --- Lire CSV ---
    df = pd.read_csv(bands_csv, sep=None, engine="python")  # gère tab/virgule
    required = {"band_id", "wavelength_nm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}. Trouvé: {list(df.columns)}")

    df = df.copy()
    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

    # fwhm optionnel
    has_fwhm = "fwhm_nm" in df.columns
    if use_fwhm_margin and not has_fwhm:
        raise ValueError("use_fwhm_margin=True mais la colonne fwhm_nm n'existe pas dans le CSV.")

    # band_id -> index 0-based correspondant à l'ordre des bandes dans img
    df["band_index"] = df["band_id"] - (1 if band_id_is_one_based else 0)

    if df["band_index"].min() < 0 or df["band_index"].max() >= nbands:
        raise ValueError(
            f"Incohérence band_id/band_index vs image: "
            f"band_index min={df['band_index'].min()}, max={df['band_index'].max()} alors que nbands={nbands}. "
            f"Vérifie band_id_is_one_based."
        )

    # Construire wavelengths alignées sur les bandes image
    wavelengths_nm = np.full(nbands, np.nan, dtype=float)
    wavelengths_nm[df["band_index"].to_numpy()] = df["wavelength_nm"].to_numpy()

    if np.isnan(wavelengths_nm).any():
        missing_idx = np.where(np.isnan(wavelengths_nm))[0]
        raise ValueError(
            f"Le CSV ne couvre pas toutes les bandes: {len(missing_idx)} manquantes (ex: {missing_idx[:10]})."
        )

    # Si on veut utiliser la fwhm pour ajuster les exclusions, on construit un vecteur fwhm aligné
    if use_fwhm_margin:
        df["fwhm_nm"] = df["fwhm_nm"].astype(float)
        fwhm_nm = np.full(nbands, np.nan, dtype=float)
        fwhm_nm[df["band_index"].to_numpy()] = df["fwhm_nm"].to_numpy()

    # --- Déterminer quelles bandes garder ---
    keep = np.ones(nbands, dtype=bool)

    for mn, mx in exclude_ranges_nm:
        if not use_fwhm_margin:
            keep &= ~((wavelengths_nm >= mn) & (wavelengths_nm <= mx))
        else:
            half = fwhm_factor * fwhm_nm
            band_min = wavelengths_nm - half
            band_max = wavelengths_nm + half
            intersects = (band_max >= mn) & (band_min <= mx)
            keep &= ~intersects

    # Retirer bords (après exclusions nm)
    kept_idx_initial = np.where(keep)[0]
    n0, n1 = drop_edges
    if n0 > 0 and len(kept_idx_initial) > n0:
        keep[kept_idx_initial[:n0]] = False
    if n1 > 0 and len(kept_idx_initial) > n1:
        keep[kept_idx_initial[-n1:]] = False

    kept_idx = np.where(keep)[0]
    removed_idx = np.where(~keep)[0]

    # --- Écrire GeoTIFF nettoyé ---
    img_out = img[kept_idx, :, :]
    profile.update(count=img_out.shape[0], dtype=dtype)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(img_out)

    # --- NEW: Écrire un CSV corrigé aligné sur l'image de sortie ---
    if output_bands_csv is not None:
        # On extrait les lignes df qui correspondent aux indices conservés
        kept_set = set(int(i) for i in kept_idx.tolist())
        df_kept = df[df["band_index"].astype(int).isin(kept_set)].copy()

        # Ordonner dans l'ordre des bandes de l'image (important)
        df_kept.sort_values("band_index", inplace=True)

        # Réindexer band_id pour correspondre à la nouvelle image (0..nbands_out-1 ou 1..nbands_out)
        df_kept["band_id_old"] = df_kept["band_id"]
        df_kept["band_index_old"] = df_kept["band_index"].astype(int)

        df_kept["band_index"] = np.arange(len(df_kept), dtype=int)
        df_kept["band_id"] = df_kept["band_index"] + (1 if csv_band_id_is_one_based_out else 0)

        # Nettoyage: on garde un schéma clair (et on conserve les anciennes colonnes en plus)
        # band_id = nouveau (aligné sortie), band_index = nouveau (0-based),
        # band_id_old / band_index_old = correspondance avec l'image d'origine
        # wavelength_nm etc. conservées
        first_cols = ["band_id", "band_index", "wavelength_nm"]
        extra_cols = [c for c in df_kept.columns if c not in first_cols]
        df_kept = df_kept[first_cols + extra_cols]

        df_kept.to_csv(output_bands_csv, index=False)

        print(f"CSV corrigé écrit: {output_bands_csv} (nbands_out={len(df_kept)})")

    print(f"Bands in : {nbands}")
    print(f"Bands out: {img_out.shape[0]}")
    print(f"Removed  : {len(removed_idx)}")

    return {
        "nbands_in": nbands,
        "nbands_out": int(img_out.shape[0]),
        "kept_indices": kept_idx,
        "removed_indices": removed_idx,
        "kept_wavelengths_nm": wavelengths_nm[kept_idx],
        "removed_wavelengths_nm": wavelengths_nm[removed_idx],
    }
