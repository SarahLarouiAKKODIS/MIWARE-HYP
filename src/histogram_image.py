import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ======================
# PATH
# ======================
wdi_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Vegatation_indice_outputs/enmap_salsigne_VII_veg.tiff"

# ======================
# READ
# ======================
with rasterio.open(wdi_path) as src:
    wdi = src.read(1).astype(np.float32)

# ======================
# SUPPRIMER NaN
# ======================
wdi_valid = wdi[np.isfinite(wdi)]

print("Nombre de pixels valides :", wdi_valid.size)
print("WDI min / max :", np.min(wdi_valid), np.max(wdi_valid))

p1, p99 = np.percentile(wdi_valid, [1, 99])

plt.figure(figsize=(8, 5))
plt.hist(wdi_valid[(wdi_valid >= p1) & (wdi_valid <= p99)], bins=150)
plt.xlabel("WDI value")
plt.ylabel("Number of pixels")
plt.title("Histogram of WDI values (1–99 percentile)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

