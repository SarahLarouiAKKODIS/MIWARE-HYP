#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


def compute_mask_percentage(
    mask_path: str | Path,
    positive_values: tuple[int, ...] = (1, 255),
    ignore_nodata: bool = True,
) -> tuple[float, int, int]:
    """
    Calcule le pourcentage de pixels actifs dans un masque binaire.

    Parameters
    ----------
    mask_path : str | Path
        Chemin vers le raster masque.
    positive_values : tuple[int, ...], default (1, 255)
        Valeurs considérées comme actives.
    ignore_nodata : bool, default True
        Si True, exclut les pixels nodata du dénominateur.

    Returns
    -------
    tuple[float, int, int]
        (pourcentage_actif, nb_pixels_actifs, nb_pixels_valides)
    """
    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        nodata = src.nodata

    valid = np.ones(mask.shape, dtype=bool)

    if ignore_nodata and nodata is not None:
        valid &= (mask != nodata)

    if np.issubdtype(mask.dtype, np.floating):
        valid &= np.isfinite(mask)

    active = np.isin(mask, positive_values) & valid

    n_active = int(np.sum(active))
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return 0.0, n_active, n_valid

    pct = (n_active / n_valid) * 100.0
    return float(pct), n_active, n_valid


def compute_mask_label_percentages(
    mask_path: str | Path,
    ignore_nodata: bool = True,
) -> dict[int, dict[str, float | int]]:
    """
    Calcule le pourcentage de chaque label dans un masque multiclasses.

    Parameters
    ----------
    mask_path : str | Path
        Chemin vers le raster masque.
    ignore_nodata : bool, default True
        Si True, exclut les pixels nodata.

    Returns
    -------
    dict
        {label: {"count": int, "percentage": float}}
    """
    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        nodata = src.nodata

    valid = np.ones(mask.shape, dtype=bool)

    if ignore_nodata and nodata is not None:
        valid &= (mask != nodata)

    if np.issubdtype(mask.dtype, np.floating):
        valid &= np.isfinite(mask)

    valid_pixels = mask[valid]

    if valid_pixels.size == 0:
        return {}

    labels, counts = np.unique(valid_pixels, return_counts=True)
    total = int(np.sum(counts))

    results: dict[int, dict[str, float | int]] = {}
    for lbl, cnt in zip(labels, counts):
        try:
            label_key = int(lbl)
        except Exception:
            label_key = lbl

        results[label_key] = {
            "count": int(cnt),
            "percentage": float(cnt / total * 100.0),
        }

    return results


def run_mask_analysis_from_config(
    config_path: str | Path,
    *,
    binary_mask_keys: Iterable[str] = (
        "Cloud_mask",
        "Haze_mask",
        "Cirrus_mask",
        "CloudShadow_mask",
        "Snow_mask",
    ),
    multiclass_mask_keys: Iterable[str] = (
        "TestFlags_mask",
    ),
    positive_values_binary: tuple[int, ...] = (1, 255),
    ignore_nodata: bool = True,
) -> dict[str, dict]:
    """
    Analyse les masques définis dans un fichier JSON de config.

    Le config doit contenir par exemple :
    {
      "Cloud_mask": "/path/Cloud_mask.tif",
      "Haze_mask": "/path/Haze_mask.tif",
      "Cirrus_mask": "/path/Cirrus_mask.tif",
      "CloudShadow_mask": "/path/CloudShadow_mask.tif",
      "Snow_mask": "/path/Snow_mask.tif",
      "TestFlags_mask": "/path/TestFlags_mask.tif"
    }

    Returns
    -------
    dict[str, dict]
        Résultats par masque.
    """
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    results: dict[str, dict] = {}

    print("\n=== Analyse des masques ===\n")

    # -----------------------------
    # Masques binaires
    # -----------------------------
    for key in binary_mask_keys:
        if key not in config:
            print(f"[WARN] Clé absente dans le config : {key}\n")
            continue

        mask_path = Path(config[key])

        try:
            pct, n_active, n_valid = compute_mask_percentage(
                mask_path=mask_path,
                positive_values=positive_values_binary,
                ignore_nodata=ignore_nodata,
            )

            results[key] = {
                "type": "binary",
                "path": str(mask_path),
                "active_pixels": n_active,
                "valid_pixels": n_valid,
                "active_percentage": pct,
                "positive_values": list(positive_values_binary),
            }

            print(f"--- {key} ---")
            print(f"Fichier                  : {mask_path}")
            print(f"Valeurs actives          : {positive_values_binary}")
            print(f"Pixels actifs            : {n_active}")
            print(f"Pixels valides           : {n_valid}")
            print(f"Pourcentage actif        : {pct:.2f} %\n")

        except Exception as e:
            results[key] = {
                "type": "binary",
                "path": str(mask_path),
                "error": str(e),
            }
            print(f"[ERREUR] {key} -> {e}\n")

    # -----------------------------
    # Masques multiclasses
    # -----------------------------
    for key in multiclass_mask_keys:
        if key not in config:
            print(f"[WARN] Clé absente dans le config : {key}\n")
            continue

        mask_path = Path(config[key])

        try:
            stats = compute_mask_label_percentages(
                mask_path=mask_path,
                ignore_nodata=ignore_nodata,
            )

            results[key] = {
                "type": "multiclass",
                "path": str(mask_path),
                "labels": stats,
            }

            print(f"--- {key} (multiclasses) ---")
            print(f"Fichier                  : {mask_path}")

            if not stats:
                print("Aucun pixel valide.\n")
                continue

            for lbl, values in stats.items():
                print(
                    f"Label {str(lbl):>5s} : "
                    f"{values['count']:>10d} pixels "
                    f"({values['percentage']:.2f} %)"
                )

            print()

        except Exception as e:
            results[key] = {
                "type": "multiclass",
                "path": str(mask_path),
                "error": str(e),
            }
            print(f"[ERREUR] {key} -> {e}\n")

    return results




# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    config_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/configs/abbaretz.json"  # à adapter
    run_mask_analysis_from_config(config_path)