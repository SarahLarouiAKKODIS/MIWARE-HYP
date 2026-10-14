#!/usr/bin/env python3

"""
Contrôle qualité d'une image hyperspectrale L2A.

Le script analyse :
- dimensions du cube ;
- NaN / Inf ;
- valeurs négatives ;
- valeurs > seuil de réflectance ;
- statistiques par bande ;
- percentiles ;
- bandes potentiellement problématiques ;
- nombre de bandes valides par pixel ;
- pixels valides sur toutes les bandes ;
- pixels ayant au moins une bande valide.

Compatible avec un raster hyperspectral lisible par rasterio.

IMPORTANT
---------
Le script ne modifie jamais l'image.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PARAMÈTRES
# ============================================================

DEFAULT_REFLECTANCE_SCALE = 10000.0
DEFAULT_MAX_REFLECTANCE = 1.2

# Seuils uniquement utilisés pour signaler les bandes suspectes
WARN_NAN_PERCENT = 5.0
WARN_NEGATIVE_PERCENT = 1.0
WARN_HIGH_PERCENT = 1.0


# ============================================================
# OUTILS
# ============================================================

def safe_percent(count, total):
    if total == 0:
        return np.nan
    return 100.0 * count / total


def get_band_wavelengths(csv_path, n_bands):
    """
    Lit éventuellement un CSV spectral contenant par exemple :

    band_id,band_index,wavelength_nm

    Retourne un tableau de longueurs d'onde ou None.
    """

    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)

    if "wavelength_nm" not in df.columns:
        raise ValueError(
            "Le CSV spectral doit contenir une colonne 'wavelength_nm'."
        )

    wavelengths = df["wavelength_nm"].to_numpy(dtype=float)

    if len(wavelengths) != n_bands:
        raise ValueError(
            f"Le CSV contient {len(wavelengths)} bandes "
            f"mais l'image en contient {n_bands}."
        )

    return wavelengths


# ============================================================
# ANALYSE
# ============================================================

def analyse_hyperspectral_quality(
    image_path,
    output_dir,
    wavelength_csv=None,
    reflectance_scale=10000.0,
    max_reflectance=1.2,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Lecture raster
    # --------------------------------------------------------

    print("=" * 80)
    print("CONTRÔLE QUALITÉ HYPERSPECTRAL")
    print("=" * 80)

    print("\nImage :", image_path)

    with rasterio.open(image_path) as src:

        raw = src.read()

        profile = src.profile.copy()

        nodata = src.nodata

        descriptions = src.descriptions

        crs = src.crs

        transform = src.transform

    bands, rows, cols = raw.shape

    n_pixels = rows * cols

    print("\nDimensions")
    print("-" * 80)

    print("Bandes :", bands)
    print("Lignes :", rows)
    print("Colonnes :", cols)
    print("Pixels par bande :", n_pixels)

    print("\nMétadonnées raster")
    print("-" * 80)

    print("dtype :", raw.dtype)
    print("nodata :", nodata)
    print("CRS :", crs)
    print("Transform :", transform)

    # --------------------------------------------------------
    # Longueurs d'onde
    # --------------------------------------------------------

    wavelengths = get_band_wavelengths(
        wavelength_csv,
        bands,
    )

    # --------------------------------------------------------
    # Masque nodata AVANT conversion
    # --------------------------------------------------------

    if nodata is not None and np.isfinite(nodata):
        nodata_mask = raw == nodata
    else:
        nodata_mask = np.zeros(raw.shape, dtype=bool)

    # --------------------------------------------------------
    # Conversion réflectance
    # --------------------------------------------------------

    cube = raw.astype(np.float32)

    if reflectance_scale is not None:
        cube /= float(reflectance_scale)

    cube[nodata_mask] = np.nan

    # Inf éventuels
    cube[~np.isfinite(cube)] = np.nan

    # --------------------------------------------------------
    # STATISTIQUES GLOBALES
    # --------------------------------------------------------

    finite = np.isfinite(cube)

    total_values = cube.size
    finite_values = finite.sum()
    nan_values = total_values - finite_values

    print("\n")
    print("=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)

    print(
        f"Valeurs finies : "
        f"{finite_values:,}/{total_values:,} "
        f"({safe_percent(finite_values, total_values):.2f} %)"
    )

    print(
        f"NaN / nodata : "
        f"{nan_values:,}/{total_values:,} "
        f"({safe_percent(nan_values, total_values):.2f} %)"
    )

    values = cube[finite]

    if values.size > 0:

        print("\nRéflectance globale")

        print(f"Min    : {np.min(values):.6f}")
        print(f"P01    : {np.percentile(values, 1):.6f}")
        print(f"P05    : {np.percentile(values, 5):.6f}")
        print(f"Médiane: {np.median(values):.6f}")
        print(f"P95    : {np.percentile(values, 95):.6f}")
        print(f"P99    : {np.percentile(values, 99):.6f}")
        print(f"Max    : {np.max(values):.6f}")

        negative = np.sum(values < 0)
        high = np.sum(values > max_reflectance)

        print(
            f"\nValeurs < 0 : "
            f"{negative:,} "
            f"({safe_percent(negative, values.size):.4f} %)"
        )

        print(
            f"Valeurs > {max_reflectance} : "
            f"{high:,} "
            f"({safe_percent(high, values.size):.4f} %)"
        )

    # ========================================================
    # STATISTIQUES PAR BANDE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STATISTIQUES PAR BANDE")
    print("=" * 80)

    records = []

    for b in range(bands):

        band = cube[b]

        finite_mask = np.isfinite(band)

        n_valid = finite_mask.sum()

        n_nan = n_pixels - n_valid

        if n_valid > 0:

            x = band[finite_mask]

            n_negative = np.sum(x < 0)
            n_high = np.sum(x > max_reflectance)

            stats = {
                "min": np.min(x),
                "p01": np.percentile(x, 1),
                "p05": np.percentile(x, 5),
                "median": np.median(x),
                "mean": np.mean(x),
                "p95": np.percentile(x, 95),
                "p99": np.percentile(x, 99),
                "max": np.max(x),
                "std": np.std(x),
            }

        else:

            n_negative = 0
            n_high = 0

            stats = {
                key: np.nan
                for key in [
                    "min",
                    "p01",
                    "p05",
                    "median",
                    "mean",
                    "p95",
                    "p99",
                    "max",
                    "std",
                ]
            }

        record = {
            "band_id": b + 1,
            "band_index": b,
            "wavelength_nm": (
                wavelengths[b]
                if wavelengths is not None
                else np.nan
            ),
            "valid_pixels": n_valid,
            "valid_percent": safe_percent(
                n_valid,
                n_pixels,
            ),
            "nan_pixels": n_nan,
            "nan_percent": safe_percent(
                n_nan,
                n_pixels,
            ),
            "negative_pixels": n_negative,
            "negative_percent": safe_percent(
                n_negative,
                n_valid,
            ),
            f"above_{max_reflectance}_pixels": n_high,
            f"above_{max_reflectance}_percent": safe_percent(
                n_high,
                n_valid,
            ),
            **stats,
        }

        records.append(record)

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Détection automatique bandes suspectes
    # --------------------------------------------------------

    high_col = f"above_{max_reflectance}_percent"

    df["warning_nan"] = (
        df["nan_percent"] > WARN_NAN_PERCENT
    )

    df["warning_negative"] = (
        df["negative_percent"] > WARN_NEGATIVE_PERCENT
    )

    df["warning_high"] = (
        df[high_col] > WARN_HIGH_PERCENT
    )

    df["warning"] = (
        df["warning_nan"]
        | df["warning_negative"]
        | df["warning_high"]
    )

    csv_out = output_dir / "quality_per_band.csv"

    df.to_csv(csv_out, index=False)

    # ========================================================
    # AFFICHAGE BANDES SUSPECTES
    # ========================================================

    suspicious = df[df["warning"]]

    print("\n")
    print("=" * 80)
    print("BANDES POTENTIELLEMENT PROBLÉMATIQUES")
    print("=" * 80)

    if suspicious.empty:

        print("Aucune bande signalée.")

    else:

        for _, row in suspicious.iterrows():

            wl = row["wavelength_nm"]

            if np.isfinite(wl):
                wl_txt = f"{wl:.2f} nm"
            else:
                wl_txt = "longueur d'onde inconnue"

            print(
                f"Bande {int(row['band_id']):3d} | "
                f"{wl_txt:12s} | "
                f"NaN={row['nan_percent']:6.2f}% | "
                f"<0={row['negative_percent']:6.2f}% | "
                f">{max_reflectance}="
                f"{row[high_col]:6.2f}%"
            )

    # ========================================================
    # QUALITÉ PAR PIXEL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("COMPLÉTUDE SPECTRALE PAR PIXEL")
    print("=" * 80)

    valid_band_count = np.isfinite(cube).sum(axis=0)

    pixels_any = valid_band_count > 0

    pixels_all = valid_band_count == bands

    print(
        "Pixels valides sur toutes les bandes :",
        pixels_all.sum(),
    )

    print(
        "Pixels ayant au moins une bande valide :",
        pixels_any.sum(),
    )

    if np.any(pixels_any):

        counts = valid_band_count[pixels_any]

        print(
            "Minimum bandes valides :",
            counts.min(),
        )

        print(
            "Maximum bandes valides :",
            counts.max(),
        )

        print(
            "Médiane :",
            np.median(counts),
        )

        print(
            "Moyenne :",
            counts.mean(),
        )

        print("\nDistribution :")

        unique, n = np.unique(
            counts,
            return_counts=True,
        )

        for n_bands, n_pix in zip(unique, n):

            print(
                f"{n_bands:3d}/{bands} bandes : "
                f"{n_pix:8d} pixels "
                f"({100 * n_pix / counts.size:6.2f} %)"
            )

    # ========================================================
    # RASTER DU NOMBRE DE BANDES VALIDES
    # ========================================================

    quality_raster = (
        output_dir
        / "valid_band_count_per_pixel.tif"
    )

    quality_profile = profile.copy()

    quality_profile.update(
        count=1,
        dtype="uint16",
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(
        quality_raster,
        "w",
        **quality_profile,
    ) as dst:

        dst.write(
            valid_band_count.astype(np.uint16),
            1,
        )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    print(
        "Bandes totalement invalides :",
        int((df["valid_pixels"] == 0).sum()),
    )

    print(
        f"Bandes avec > {WARN_NAN_PERCENT}% NaN :",
        int(
            (
                df["nan_percent"]
                > WARN_NAN_PERCENT
            ).sum()
        ),
    )

    print(
        f"Bandes avec > {WARN_NEGATIVE_PERCENT}% "
        f"de valeurs négatives :",
        int(
            (
                df["negative_percent"]
                > WARN_NEGATIVE_PERCENT
            ).sum()
        ),
    )

    print(
        f"Bandes avec > {WARN_HIGH_PERCENT}% "
        f"de valeurs > {max_reflectance} :",
        int(
            (
                df[high_col]
                > WARN_HIGH_PERCENT
            ).sum()
        ),
    )

    print("\nFichiers créés :")
    print("CSV qualité :", csv_out)
    print(
        "Raster complétude spectrale :",
        quality_raster,
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Contrôle qualité d'un cube "
            "hyperspectral L2A."
        )
    )

    parser.add_argument(
        "image",
        help="Image hyperspectrale L2A",
    )

    parser.add_argument(
        "--wavelengths",
        default=None,
        help=(
            "CSV contenant la colonne "
            "'wavelength_nm'"
        ),
    )

    parser.add_argument(
        "--output",
        default="hyperspectral_quality",
        help="Dossier de sortie",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_REFLECTANCE_SCALE,
        help=(
            "Facteur d'échelle de la réflectance. "
            "Défaut = 10000."
        ),
    )

    parser.add_argument(
        "--max-reflectance",
        type=float,
        default=DEFAULT_MAX_REFLECTANCE,
        help=(
            "Seuil utilisé pour détecter les "
            "réflectances élevées."
        ),
    )

    args = parser.parse_args()

    analyse_hyperspectral_quality(
        image_path=args.image,
        output_dir=args.output,
        wavelength_csv=args.wavelengths,
        reflectance_scale=args.scale,
        max_reflectance=args.max_reflectance,
    )