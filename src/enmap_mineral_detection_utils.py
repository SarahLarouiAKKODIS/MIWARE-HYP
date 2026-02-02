import pandas as pd

def load_mineral_targets(csv_path: str, mineral: str) -> dict[float, tuple[int, float]]:
    """
    Retourne un dict:
      {target_nm: (band_id, wavelength_nm)}
    à partir de enmap_selected_bands_by_mineral.csv
    """
    df = pd.read_csv(csv_path)
    sub = df[df["mineral"] == mineral].copy()
    
    if sub.empty:
        raise ValueError(f"Aucune ligne trouvée pour mineral='{mineral}' dans {csv_path}")

    # Normaliser les types
    sub["target_nm"] = sub["target_nm"].astype(float)
    sub["band_id"] = sub["band_id"].astype(int)
    sub["wavelength_nm"] = sub["wavelength_nm"].astype(float)

    return {row.target_nm: (row.band_id, row.wavelength_nm) for row in sub.itertuples(index=False)}
