from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def recover_wavelet_band_info(
    xml_path: str | Path,
    out_csv: str | Path | None = None,
    *,
    fail_if_empty: bool = True
) -> pd.DataFrame:
    """
    EnMAP METADATA.XML -> table des bandes (bandID + wavelength/FWHM/gain/offset).

    Parameters
    ----------
    xml_path : str | Path
        Chemin vers le fichier *-METADATA.XML.
    out_csv : str | Path | None, optional
        Si fourni, enregistre le DataFrame en CSV à cet emplacement.
    fail_if_empty : bool, default True
        Si True, lève une erreur si aucune bande n'est extraite.

    Returns
    -------
    pd.DataFrame
        Colonnes: band_id, wavelength_nm, fwhm_nm, gain, offset
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    # 1) Trouver le noeud <bandCharacterisation> (robuste aux namespaces)
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
    rows: list[dict] = []
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

    df = pd.DataFrame(rows).sort_values("band_id").reset_index(drop=True)

    if df.empty and fail_if_empty:
        raise RuntimeError(
            "Aucune bande extraite. Vérifie que le XML est bien un *-METADATA.XML "
            "et que les balises bandID/wavelengthCenterOfBand existent."
        )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df
