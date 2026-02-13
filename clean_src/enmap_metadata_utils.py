#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnMAP METADATA.XML -> table des bandes + sélection des bandes les plus proches
de longueurs d’onde cibles (par minéral).

Fonctionne même si les balises XML ont des namespaces et/ou si bandID n’est pas
un enfant direct du "bloc" de bande (parent_map + fallback en profondeur).

Dépendances :
    pip install pandas
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def parse_band_characterisation(xml_path: str | Path) -> pd.DataFrame:
    """
    Va dans <bandCharacterisation> puis, pour chaque <bandID number="...">,
    récupère wavelengthCenterOfBand, FWHMOfBand, GainOfBand, OffsetOfBand.
    Robuste aux namespaces (on matche les fins de tags).
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    # 1) Trouver le noeud <bandCharacterisation>
    band_char = None
    for node in root.iter():
        if isinstance(node.tag, str) and node.tag.endswith("bandCharacterisation"):
            band_char = node
            break

    if band_char is None:
        raise ValueError("Balise <bandCharacterisation> introuvable dans ce XML.")

    def child_text(parent: ET.Element, suffix: str) -> str | None:
        """Texte d'un enfant direct dont le tag finit par `suffix`."""
        for c in list(parent):
            if isinstance(c.tag, str) and c.tag.endswith(suffix):
                return None if c.text is None else c.text.strip()
        return None

    def to_float(x: str | None) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except ValueError:
            return None

    # 2) Parcourir les <bandID number="..."> sous <bandCharacterisation>
    rows = []
    for band in list(band_char):
        if not (isinstance(band.tag, str) and band.tag.endswith("bandID")):
            continue

        band_id = band.attrib.get("number")
        if band_id is None:
            continue

        rows.append({
            "band_id": int(band_id),
            "wavelength_nm": to_float(child_text(band, "wavelengthCenterOfBand")),
            "fwhm_nm": to_float(child_text(band, "FWHMOfBand")),
            "gain": to_float(child_text(band, "GainOfBand")),
            "offset": to_float(child_text(band, "OffsetOfBand")),
        })

    df = (pd.DataFrame(rows)
          .sort_values("band_id")
          .reset_index(drop=True))

    return df

# ---------------------------
# Sélection des bandes proches
# ---------------------------

def closest_band_dict(df: pd.DataFrame, targets_nm: list[float]) -> dict[float, tuple[int, float]]:
    """
    Retourne {target_nm: (band_id, wavelength_nm)}
    """
    out = {}
    for t in targets_nm:
        idx = (df["wavelength_nm"] - float(t)).abs().idxmin()
        out[float(t)] = (
            int(df.loc[idx, "band_id"]),
            float(df.loc[idx, "wavelength_nm"])
        )
    return out
