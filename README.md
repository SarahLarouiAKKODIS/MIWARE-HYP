# Hyperspectral Image Processing Project

## Overview
This project focuses on the processing and analysis of hyperspectral images for **mineral detection** and **vegetation health assessment**.

It includes data preprocessing, spectral band analysis, vegetation and water index computation, and classification of vegetation stress levels.
The workflow is designed to be modular and reproducible for large hyperspectral datasets.

---

## Features
- Hyperspectral metadata and wavelet-based information recovery
- Spectral band selection for mineral analysis
- Image cleaning and cropping
- Water and vegetation mask generation
- Computation of vegetation health indices:
  - NDVI
  - MSAVI
  - NDWI
  - VII
  - GNDVI
  - Decorrelated WDI
- Vegetation stress classification based on percentile thresholds
- Mineral detection (example: olivine)

---

## Project Structure
```text
.
├── src/        # Python source code
├── Data/       # Hyperspectral data (not included)
├── Results/    # Generated results (masks, indices, outputs)
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Data
Hyperspectral datasets are **not included** in this repository.

Place your data inside the `Data/` directory before running the scripts.
The expected data format depends on the specific script (e.g. ENMAP products).

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv hypenv
source hypenv/bin/activate  # Linux / macOS
hypenv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Data information recovery
Recover spectral and metadata information using wavelet-based analysis and band selection.
```bash
python src/enmap_metadata_wavelet_recovery.py
python src/enmap_mineral_band_selection.py
```

---

### Data preprocessing
Clean and crop hyperspectral images to prepare them for analysis.
```bash
python src/clean_hyperspectral_image.py
python src/crop_hyperspectral_image.py
```

---

### Water and vegetation indices
Compute binary masks (water and vegetation) and vegetation health indices.
```bash
python src/enmap_water_indices.py
python src/enmap_vegetation_indices.py
```

---

### Vegetation health classification
Compute labeled vegetation health masks using a combination of VII and WDI indices.
Classification is based on percentile thresholds and includes four classes:
1. Combined stress
2. Water stress
3. Chlorophyll stress
4. Healthy vegetation

```bash
python src/vegetation_health_wdi_vii.py
```

---

### Mineral detection
Example of mineral detection applied to **olivine**.
```bash
python src/enmap_olivine_detection.py
```

---

## Outputs
All generated outputs (masks, indices, classification maps) are saved in the `Results/` directory.

These results are generated automatically and are **not tracked by Git**.

---

## Notes
- Large hyperspectral datasets are excluded from version control.
- Results are reproducible by re-running the scripts.
- Ensure sufficient CPU, RAM, and GPU resources when working with large datasets.

---

## License
This project is intended for academic and research purposes.
Add a license file if redistribution or reuse is planned.

---

## Author
Sarah Laroui