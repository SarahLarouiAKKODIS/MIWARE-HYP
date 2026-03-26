from enmap_metadata_utils import parse_band_characterisation

### Recover wavelet band information

Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/"
xml_path = Path_data + "SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-METADATA.XML"  

# 1) Lire et sauvegarder la table complète des bandes
df = parse_band_characterisation(xml_path)
if df.empty:
    raise RuntimeError(
        "Aucune bande extraite. Vérifie que le XML est bien un *-METADATA.XML "
        "et que les balises bandID/wavelengthCenterOfBand existent."
    )

df.to_csv("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/enmap_bands_full2.csv", index=False)
print("✅ Table complète enregistrée : enmap_bands_full.csv")
print(df.head(10).to_string(index=False))

