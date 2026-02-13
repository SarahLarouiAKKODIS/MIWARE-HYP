from pathlib import Path

##################### ECOSTRESS #####################

spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/ECOSTRESS/ecospeclib-1770820661442"

ref_dir = Path(spectral_library_dir)

minerals = {
    p.name.lower().split(".")[1]
    for p in ref_dir.glob("*.txt")
    if p.name.lower().startswith("mineral.")
    and "ancillary" not in p.name.lower()
}

minerals = sorted(minerals)

print(f"{len(minerals)} minerais uniques trouvés :\n")
for m in minerals:
    print("-", m)


##################### USGS #####################

spectral_library_dir = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/spectral_libraries/USGS/ASCIIdata_splib07a/ChapterM_Minerals/"

ref_dir = Path(spectral_library_dir)

allowed_spectrometers = {"beckb", "asdfrb"}  # lowercase

selected_files = []
minerals = set()

for p in ref_dir.glob("*.txt"):
    parts = p.stem.split("_")  # sans .txt

    if len(parts) < 4:
        continue  # nom inattendu

    mineral = parts[1].lower()
    spectrometer = parts[3].lower()

    if spectrometer in allowed_spectrometers:
        selected_files.append(p)
        minerals.add(mineral)

print(f"{len(selected_files)} spectres conservés")
print(f"{len(minerals)} minerais uniques\n")

print("Minerais :")
for m in sorted(minerals):
    print("-", m)
