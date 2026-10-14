from pathlib import Path
import re

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path("/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/BD_ForetV2_France/BDForetV2_all_departments/extracted")


# ============================================================
# DÉPARTEMENTS ATTENDUS - FRANCE MÉTROPOLITAINE
# ============================================================

departments_expected = {
    f"{i:03d}"
    for i in range(1, 96)
    if i != 20
}

# Corse
departments_expected.update({"02A", "02B"})


# ============================================================
# RECHERCHE DES DÉPARTEMENTS PRÉSENTS
# ============================================================

departments_found = set()

# Accepte par exemple :
# BDFORET_2-0__SHP_LAMB93_D088_2015-04-01
# BDFORET_2-0__SHP_LAMB93_D02A_...
pattern = re.compile(r"_D(0\d{2}|02A|02B)_")

for path in ROOT_DIR.iterdir():

    match = pattern.search(path.name)

    if match:
        departments_found.add(match.group(1))


# ============================================================
# COMPARAISON
# ============================================================

missing = sorted(
    departments_expected - departments_found
)

extra = sorted(
    departments_found - departments_expected
)


# ============================================================
# RÉSULTATS
# ============================================================

print("=" * 60)
print("BD FORET V2 - CONTROLE DES DEPARTEMENTS")
print("=" * 60)

print(f"\nDépartements attendus : {len(departments_expected)}")
print(f"Départements trouvés  : {len(departments_found)}")
print(f"Départements manquants : {len(missing)}")

if missing:
    print("\nDépartements manquants :")

    for dep in missing:
        print(f"  D{dep}")

else:
    print("\nTous les départements métropolitains sont présents.")

if extra:
    print("\nCodes inattendus détectés :")

    for dep in extra:
        print(f"  D{dep}")