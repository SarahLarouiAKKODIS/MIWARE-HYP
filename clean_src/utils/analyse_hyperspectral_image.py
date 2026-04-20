from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


def analyze_rescaled_cube_with_wavelengths(
    cube_tif: str | Path,
    bands_csv: str | Path,
    *,
    out_csv: str | Path | None = None,
    min_valid: float = -0.1,
    max_valid: float = 1.5,
    nan_threshold_pct: float = 50.0,
    outlier_threshold_pct: float = 5.0,
    band_id_is_one_based: bool = True,
    print_suspicious: bool = True,
) -> dict:
    """
    Analyse un cube hyperspectral rescalé (avec NaN) et associe les anomalies
    aux longueurs d'onde du CSV.

    Parameters
    ----------
    cube_tif : str | Path
        Chemin du cube hyperspectral.
    bands_csv : str | Path
        CSV contenant au minimum: band_id, wavelength_nm.
    out_csv : str | Path | None, optional
        Si fourni, écrit le tableau complet d'analyse par bande.
    min_valid : float, default -0.1
        Seuil minimum acceptable.
    max_valid : float, default 1.5
        Seuil maximum acceptable.
    nan_threshold_pct : float, default 50.0
        % de NaN au-dessus duquel une bande est suspecte.
    outlier_threshold_pct : float, default 5.0
        % de valeurs hors plage au-dessus duquel une bande est suspecte.
    band_id_is_one_based : bool, default True
        True si le CSV a des band_id en 1..N.
    print_suspicious : bool, default True
        Affiche les bandes suspectes dans la console.

    Returns
    -------
    dict
        {
            "global_min": float,
            "global_max": float,
            "global_mean": float,
            "global_std": float,
            "analysis_df": pd.DataFrame,
            "suspicious_df": pd.DataFrame,
            "suspicious_band_indices": list[int],
            "suspicious_band_ids": list[int],
        }
    """
    cube_tif = Path(cube_tif)
    bands_csv = Path(bands_csv)

    if not cube_tif.exists():
        raise FileNotFoundError(f"Fichier introuvable : {cube_tif}")
    if not bands_csv.exists():
        raise FileNotFoundError(f"Fichier introuvable : {bands_csv}")

    # --- Lecture cube ---
    with rasterio.open(cube_tif) as src:
        cube = src.read().astype(np.float32)

    nbands, rows, cols = cube.shape

    # --- Lecture CSV ---
    df = pd.read_csv(bands_csv, sep=None, engine="python").copy()

    required = {"band_id", "wavelength_nm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV: {missing}. Trouvé: {list(df.columns)}"
        )

    df["band_id"] = df["band_id"].astype(int)
    df["wavelength_nm"] = df["wavelength_nm"].astype(float)

    if "fwhm_nm" in df.columns:
        df["fwhm_nm"] = pd.to_numeric(df["fwhm_nm"], errors="coerce")

    # band_id -> index 0-based
    df["band_index"] = df["band_id"] - (1 if band_id_is_one_based else 0)

    if df["band_index"].min() < 0 or df["band_index"].max() >= nbands:
        raise ValueError(
            f"Incohérence band_id/band_index vs image: "
            f"band_index min={df['band_index'].min()}, "
            f"max={df['band_index'].max()}, nbands={nbands}. "
            f"Vérifie band_id_is_one_based."
        )

    # Réordonner par index raster
    df = df.sort_values("band_index").reset_index(drop=True)

    if len(df) != nbands:
        raise ValueError(
            f"Le CSV contient {len(df)} bandes alors que l'image en contient {nbands}."
        )

    expected_idx = np.arange(nbands)
    if not np.array_equal(df["band_index"].to_numpy(), expected_idx):
        raise ValueError(
            "Le CSV ne couvre pas exactement toutes les bandes dans le bon indexage."
        )

    print("===== ANALYSE DU CUBE =====")

    # --- stats globales ---
    finite_vals = cube[np.isfinite(cube)]
    if finite_vals.size == 0:
        raise ValueError("Aucune valeur valide dans le cube.")

    vmin = float(np.nanmin(cube))
    vmax = float(np.nanmax(cube))
    vmean = float(np.nanmean(cube))
    vstd = float(np.nanstd(cube))

    print("\n[GLOBAL]")
    print(f"min  = {vmin:.4f}")
    print(f"max  = {vmax:.4f}")
    print(f"mean = {vmean:.4f}")
    print(f"std  = {vstd:.4f}")

    if vmin < min_valid or vmax > max_valid:
        print("\n[WARNING] Valeurs globales hors plage attendue !")

    # --- stats par bande, robustes aux bandes entièrement NaN ---
    band_min = np.full(nbands, np.nan, dtype=np.float32)
    band_max = np.full(nbands, np.nan, dtype=np.float32)
    band_mean = np.full(nbands, np.nan, dtype=np.float32)
    band_std = np.full(nbands, np.nan, dtype=np.float32)
    band_nan_pct = np.mean(~np.isfinite(cube), axis=(1, 2)) * 100.0
    band_all_nan = np.all(~np.isfinite(cube), axis=(1, 2))

    out_of_range = (cube < min_valid) | (cube > max_valid)
    out_of_range[~np.isfinite(cube)] = False
    band_out_pct = np.mean(out_of_range, axis=(1, 2)) * 100.0

    for i in range(nbands):
        band = cube[i]
        finite = np.isfinite(band)
        if finite.any():
            vals = band[finite]
            band_min[i] = float(vals.min())
            band_max[i] = float(vals.max())
            band_mean[i] = float(vals.mean())
            band_std[i] = float(vals.std())

    suspicious = (
        (band_min < min_valid)
        | (band_max > max_valid)
        | (band_nan_pct > nan_threshold_pct)
        | (band_out_pct > outlier_threshold_pct)
        | band_all_nan
    )

    analysis_df = df.copy()
    analysis_df["band_min"] = band_min
    analysis_df["band_max"] = band_max
    analysis_df["band_mean"] = band_mean
    analysis_df["band_std"] = band_std
    analysis_df["band_nan_pct"] = band_nan_pct
    analysis_df["band_out_pct"] = band_out_pct
    analysis_df["band_all_nan"] = band_all_nan
    analysis_df["suspicious"] = suspicious

    suspicious_df = analysis_df[analysis_df["suspicious"]].copy()

    print("\n[BANDES SUSPECTES]")
    if suspicious_df.empty:
        print("Aucune bande suspecte 👍")
    elif print_suspicious:
        for _, row in suspicious_df.iterrows():
            wl_txt = f"{row['wavelength_nm']:.1f} nm"
            print(
                f"Bande idx={int(row['band_index']):03d} | "
                f"band_id={int(row['band_id']):03d} | "
                f"λ={wl_txt} | "
                f"min={row['band_min']:.3f} | "
                f"max={row['band_max']:.3f} | "
                f"%NaN={row['band_nan_pct']:.1f} | "
                f"%out={row['band_out_pct']:.1f} | "
                f"all_nan={bool(row['band_all_nan'])}"
            )

    print("\n[SUMMARY]")
    print(f"Nombre de bandes        : {nbands}")
    print(f"Bandes suspectes        : {len(suspicious_df)}")

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(out_csv, index=False)
        print(f"Analyse par bande écrite : {out_csv}")

    return {
        "global_min": vmin,
        "global_max": vmax,
        "global_mean": vmean,
        "global_std": vstd,
        "analysis_df": analysis_df,
        "suspicious_df": suspicious_df,
        "suspicious_band_indices": suspicious_df["band_index"].astype(int).tolist(),
        "suspicious_band_ids": suspicious_df["band_id"].astype(int).tolist(),
    }