import pandas as pd
from enmap_metadata_utils import pick_bands_for_minerals

# 1) Charger les bandes EnMAP
Path_data = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"
output = Path_data + "enmap_selected_bands_by_mineral.csv"

df = pd.read_csv(Path_data + "enmap_bands_full.csv")

# 2) Définir les minéraux et longueurs d’onde
minerals_targets_nm = {

    # --- olivine ---
    # absorption principale ~1 µm (Fe2+) + contrôle ~2 µm
    "olivine": [860, 1050, 1280, 1800, 2000, 2300],

    # --- pyroxenes ---
    # absorption ~1 µm et ~2 µm (plus marquée que l’olivine)
    "pyroxene": [900, 1000, 1200, 1800, 2000, 2300],

    # --- iron oxides (Fe3+) ---
    # rougeur VIS + absorption ferrique ~0.9 µm
    "iron_oxides": [550, 650, 860, 900, 940],

    # --- carbonates (CO3) ---
    # absorptions diagnostiques ~2.33–2.35 µm (+ optionnel ~2.50 µm)
    "carbonates": [2200, 2330, 2450, 2400, 2500, 2600],

    # --- micas ---
    # Al-OH ~2.20 µm (muscovite / illite)
    # + H2O/OH ~1.90 µm (robustesse)
    "micas": [1800, 1900, 2000, 2100, 2200, 2300],

    # --- argiles ---
    # Al-OH ~2.20 µm (principale)
    # + H2O / OH ~1.90 µm (contrôle)
    "argiles": [1800, 1900, 2000, 2100, 2200, 2300],

    # --- amphiboles ---
    # Mg/Fe-OH ~2.30–2.35 µm
    # + contrôle ~2.00 µm
    "amphiboles": [2250, 2320, 2390, 1900, 2000, 2100],
}



# 3) Sélectionner les bandes
df_sel = pick_bands_for_minerals(df, minerals_targets_nm)

# 4) Résultat
print(df_sel)
df_sel.to_csv(output, index=False)
