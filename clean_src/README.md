# Pipeline EnMAP — Prétraitement, détection minérale par band depths et comparaison spectrale

Ce dépôt regroupe trois scripts principaux permettant de traiter une image hyperspectrale EnMAP et d’en extraire des produits utiles pour la détection minérale.

Les trois scripts correspondent à trois niveaux du pipeline :

1. `main.py`  
   Prétraitement général de l’image EnMAP : récupération des bandes, rescaling, masquage qualité, crop, RGB, nettoyage spectral, QC, masques eau/végétation, extraction des pixels candidats minéraux.

2. `main_banddepth_mineral_detection.py`  
   Détection minérale par indices spectraux et profondeurs de bandes : olivine, pyroxène, amphiboles, carbonates, micas, argiles, oxydes de fer.

3. `main_spectralcomparison_mineral_detection.py`  
   Détection minérale par comparaison à des bibliothèques spectrales : SAM, Matched Filter, ACE et scores combinés.

---

## 1. Organisation générale du pipeline

Le pipeline complet suit la logique suivante :

```text
Image EnMAP brute
        |
        v
main.py
  - récupération des métadonnées spectrales
  - rescaling en réflectance
  - application des masques qualité
  - crop optionnel
  - RGB de contrôle
  - suppression des bandes bruitées
  - contrôle qualité spectral
  - calcul des masques eau et végétation
  - génération du cube candidat minéral
        |
        v
main_banddepth_mineral_detection.py
  - lissage spectral sans normalisation
  - calcul de band depths
  - masques minéraux thématiques
        |
        v
main_spectralcomparison_mineral_detection.py
  - lissage spectral + normalisation L2
  - comparaison aux bibliothèques spectrales
  - SAM, MF, ACE
  - cartes composites de similarité
```

---

## 2. `main.py` — Prétraitement général EnMAP

### Objectif

`main.py` prépare le cube hyperspectral EnMAP pour les étapes de détection minérale. Il produit une image nettoyée spectralement, des masques eau/végétation, une image RGB de contrôle et un cube final contenant les pixels candidats pour l’analyse minérale.

### Entrées principales

Le script lit un fichier de configuration JSON :

```python
config_path = "/home/configs.json"
```

Le fichier de configuration doit contenir au minimum :

```json
{
  "Path_res": "/home/Results/",
  "xml_path": "/path/to/METADATA.XML",
  "image_hyp": "/path/to/SPECTRAL_IMAGE.TIF",
  "Cloud_mask": "/path/to/Cloud_mask.TIF",
  "Haze_mask": "/path/to/Haze_mask.TIF",
  "Cirrus_mask": "/path/to/Cirrus_mask.TIF",
  "CloudShadow_mask": "/path/to/CloudShadow_mask.TIF",
  "Snow_mask": "/path/to/Snow_mask.TIF",
  "TestFlags_mask": "/path/to/TestFlags_mask.TIF",
  "Site_coords": [x_min, y_min, x_max, y_max],
  "crop": true
}
```

### 2.1 Récupération des informations spectrales

Fonction utilisée :

```python
recover_wavelet_band_info(xml_path, out_csv=wavelengths_csv)
```

Sortie :

```text
enmap_bands_full.csv
```

Ce fichier contient notamment :

| Colonne | Description |
|---|---|
| `band_id` | Identifiant de bande |
| `wavelength_nm` | Longueur d’onde centrale |
| `fwhm_nm` | Largeur à mi-hauteur |
| `gain` | Gain radiométrique |
| `offset` | Offset radiométrique |

Ce CSV permet d’associer chaque bande raster à sa longueur d’onde réelle.

### 2.2 Rescaling de l’image

Fonction utilisée :

```python
rescale_enmap_cube_simple(
    input_tif=image_hyperspectrale,
    output_tif=image_hyperspectrale_reflectance,
    scale_factor=10000.0
)
```

Sortie :

```text
image_hyperspectrale_reflectance.tif
```

Cette étape convertit les valeurs numériques brutes en réflectance approximative par division par `10000`.

### 2.3 Application des masques qualité

Fonction utilisée :

```python
mask_enmap_hyperspectral_cube(
    cube_tif=image_hyperspectrale_reflectance,
    mask_files=mask_files,
    out_tif=image_hyperspectrale_clean
)
```

Masques utilisés :

- `Cloud_mask`
- `Haze_mask`
- `Cirrus_mask`
- `CloudShadow_mask`
- `Snow_mask`
- `TestFlags_mask`

Sortie :

```text
image_hyperspectrale_clean.tif
```

Les pixels où au moins un masque est actif sont mis à `NaN` dans toutes les bandes.

### 2.4 Crop optionnel

Fonction utilisée :

```python
crop_hyperspectral_tif(
    image_hyperspectrale_clean,
    image_hyperspectrale_clean_crop,
    Site_coords
)
```

Sortie si `Crop=True` :

```text
image_hyperspectrale_crop.tif
```

Le crop est effectué en coordonnées pixel :

```text
[x_min, y_min, x_max, y_max]
```

### 2.5 Génération RGB

Fonction utilisée :

```python
hyperspectral_to_rgb(
    cube_tif=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    out_rgb_tif=image_hyperspectrale_rgb,
    target_R=650,
    target_G=560,
    target_B=480
)
```

Sortie :

```text
image_hyperspectrale_rgb.tif
```

L’image RGB sert de produit de visualisation rapide. Les bandes les plus proches de 650 nm, 560 nm et 480 nm sont utilisées.

### 2.6 Nettoyage spectral des bandes

Fonction utilisée :

```python
clean_bands_enmap_from_csv(
    img_path=image_hyperspectrale_clean_crop,
    bands_csv=wavelengths_csv,
    output_path=image_hyperspectrale_cleanbands,
    output_bands_csv=clean_wavelengths_csv,
    band_id_is_one_based=True,
    csv_band_id_is_one_based_out=True,
    drop_edges=(2, 2),
    exclude_ranges_nm=[(1340, 1460), (1800, 1960)],
    use_fwhm_margin=True,
    fwhm_factor=0.5
)
```

Sorties :

```text
image_hyperspectrale_cleanbands.tif
enmap_clean_bands_full.csv
```

Bandes supprimées :

- zones d’absorption atmosphérique autour de 1400 nm ;
- zones d’absorption atmosphérique autour de 1900 nm ;
- premières et dernières bandes selon `drop_edges=(2, 2)` ;
- bandes dont l’intervalle spectral recoupe les zones exclues si `use_fwhm_margin=True`.

Le CSV corrigé est réindexé pour correspondre exactement au nouveau cube.

### 2.7 Contrôle qualité spectral

Fonction utilisée :

```python
analyze_rescaled_cube_with_wavelengths(
    cube_tif=image_hyperspectrale_cleanbands,
    bands_csv=clean_wavelengths_csv,
    out_csv=Path_res / "band_qc_report.csv",
    min_valid=-0.1,
    max_valid=1.5,
    nan_threshold_pct=50.0,
    outlier_threshold_pct=5.0,
    band_id_is_one_based=True
)
```

Sortie :

```text
band_qc_report.csv
```

Cette étape produit des statistiques par bande : minimum, maximum, moyenne, pourcentage de NaN, pourcentage de valeurs hors plage et bandes suspectes.

### 2.8 Masque eau

Fonction utilisée :

```python
compute_mndwi_and_water_mask(
    tif_path=image_hyperspectrale_cleanbands,
    wavelengths_csv=clean_wavelengths_csv,
    outdir=Water_results_dir,
    prefix="enmap_salsigne",
    mndwi_th=0.55,
    verbose=True
)
```

Sorties principales :

| Fichier | Format | Description |
|---|---|---|
| `enmap_salsigne_MNDWI_Xu.tiff` | float32 | Carte MNDWI |
| `enmap_salsigne_WATER_MASK.tiff` | int16 | Masque eau |
| `enmap_salsigne_WATER_MASK_VISUAL.tiff` | RGB uint8 | Visualisation du masque |

Convention du masque eau :

| Label | Signification |
|---|---|
| `255` | Eau |
| `0` | Non-eau |
| `-1` | Pixel invalide / exclu |

### 2.9 Masque végétation

Fonction utilisée :

```python
compute_vegetation_indices_wdi_vii(
    tif_path=image_hyperspectrale_cleanbands,
    wavelengths_csv=clean_wavelengths_csv,
    outdir=Vegetation_results_dir,
    prefix="enmap_salsigne",
    ndvi_th=0.3,
    verbose=True
)
```

Sorties principales :

| Fichier | Format | Description |
|---|---|---|
| `enmap_salsigne_NDVI.tiff` | float32 | NDVI |
| `enmap_salsigne_GNDVI.tiff` | float32 | GNDVI |
| `enmap_salsigne_NDRE.tiff` | float32 | NDRE |
| `enmap_salsigne_NDWI_Gao.tiff` | float32 | NDWI Gao |
| `enmap_salsigne_MSAVI.tiff` | float32 | MSAVI |
| `enmap_salsigne_VEG_MASK.tiff` | int16 | Masque végétation |
| `enmap_salsigne_VEG_MASK_VISUAL.tiff` | RGB uint8 | Visualisation du masque |

Convention du masque végétation :

| Label | Signification |
|---|---|
| `255` | Végétation |
| `0` | Non-végétation |
| `-1` | Pixel invalide / exclu |

### 2.10 Extraction du cube candidat minéral

Fonction utilisée :

```python
apply_water_veg_mask(
    img_path=image_hyperspectrale_cleanbands,
    water_mask_path=Water_results_dir / "enmap_salsigne_WATER_MASK.tiff",
    veg_mask_path=Vegetation_results_dir / "enmap_salsigne_VEG_MASK.tiff",
    output_path=enmap_mineral_candidates
)
```

Sortie :

```text
enmap_mineral_candidates.tif
```

Ce cube exclut les pixels d’eau, les pixels de végétation et les pixels invalides. Il correspond aux pixels potentiellement exploitables pour la détection minérale.

---

## 3. `main_banddepth_mineral_detection.py` — Détection minérale par band depths

### Objectif

Ce script réalise une détection minérale basée sur des indices spectraux ciblés, principalement des profondeurs de bandes d’absorption. Il utilise un cube déjà nettoyé spectralement.

### Entrées principales

```python
Path_res = "/home/Results/"
clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"
image_hyperspectrale_cleanbands = Path_res + "image_hyperspectrale_cleanbands.tif"
```

### Étape 1 — Lissage spectral sans normalisation

Fonction utilisée :

```python
savgol_smooth_and_normalize(
    img_path=image_hyperspectrale_cleanbands,
    output_path=image_hyperspectrale_cleanbands_smooth,
    normalize=None
)
```

Sortie :

```text
image_hyperspectrale_cleanbands_smooth.tif
```

Choix méthodologique :

- `normalize=None` est recommandé pour les band depths ;
- la normalisation L2 modifierait les amplitudes relatives utiles aux indices d’absorption ;
- le lissage Savitzky–Golay réduit le bruit spectral tout en conservant la forme générale du spectre.

### Étape 2 — Détections minérales

Toutes les fonctions utilisent :

```python
tif_path=image_hyperspectrale_cleanbands_smooth
bands_csv=clean_wavelengths_csv
```

#### Olivine

Fonction :

```python
detect_olivine_bd1050_bd2000_clean(...)
```

Méthode :

- `BD1050 = band_depth(860, 1050, 1280)`
- `BD2000 = band_depth(1800, 2000, 2300)`

Condition de masque :

```text
BD1050 > 0.05
BD2000 < 0.02
```

Sorties :

| Fichier | Description |
|---|---|
| `BD1050_olivine.tif` | Absorption principale autour de 1050 nm |
| `BD2000_control.tif` | Critère de rejet autour de 2000 nm |
| `olivine_mask.tif` | Masque binaire |
| `olivine_probability.tif` | Score continu |

#### Pyroxène

Fonction :

```python
detect_pyroxene_bd1um_bd2um_clean(...)
```

Méthode :

- `BD1um = band_depth(900, 1000, 1200)`
- `BD2um = band_depth(1800, 2000, 2300)`

Condition :

```text
BD1um > 0.05
BD2um > 0.03
```

#### Amphiboles

Fonction :

```python
detect_amphiboles_bd2320_clean(...)
```

Méthode :

- `BD2320 = band_depth(2250, 2320, 2390)`
- contrôle optionnel `BD2000 = band_depth(1900, 2000, 2100)`

#### Carbonates

Fonction :

```python
detect_carbonates_bd2330_bd2500_clean(...)
```

Méthode :

- `BD2330 = band_depth(2200, 2330, 2450)`
- optionnel : `BD2500 = band_depth(2400, 2500, 2600)`

Par défaut, `use_bd2500=False`.

#### Micas

Fonction :

```python
detect_micas_bd2200_clean(...)
```

Méthode :

- `BD2200 = band_depth(2100, 2200, 2300)`
- contrôle `BD1900 = band_depth(1800, 1900, 2000)`

Ici, le contrôle à 1900 nm est utilisé comme critère de rejet.

#### Argiles

Fonction :

```python
detect_argiles_bd2200_clean(...)
```

Méthode :

- `BD2200 = band_depth(2100, 2200, 2300)`
- contrôle `BD1900 = band_depth(1800, 1900, 2000)`

Contrairement aux micas, le signal autour de 1900 nm est ici utilisé comme critère positif d’hydratation.

Attention : dans la version actuelle, `write_outputs=False` par défaut. La fonction retourne donc les tableaux en mémoire, mais n’écrit pas forcément les rasters sur disque.

#### Oxydes de fer

Fonction :

```python
detect_iron_oxides_bd900_redness_clean(...)
```

Méthode :

- `BD900 = band_depth(860, 900, 940)`
- `REDNESS = (R650 - R550) / (R650 + R550)`

Condition :

```text
BD900 > 0.04
REDNESS > 0.05
```

### Convention générale des masques minéraux

| Label | Signification |
|---|---|
| `255` | Pixel détecté |
| `0` | Pixel non détecté ou invalide |

Attention : ces masques ne distinguent pas explicitement les pixels invalides des pixels valides non détectés. Pour cela, il faut consulter les rasters continus associés, qui conservent des `NaN`.

---

## 4. `main_spectralcomparison_mineral_detection.py` — Détection par comparaison spectrale

### Objectif

Ce script détecte un minéral cible par comparaison des spectres image à des spectres de référence issus de bibliothèques spectrales.

Deux approches sont utilisées :

1. `SAM + MF`
2. `SAM + ACE`

### Entrées principales

```python
Path_res = "/home/Results/"
clean_wavelengths_csv = Path_res + "enmap_clean_bands_full.csv"
image_hyperspectrale_cleanbands = Path_res + "image_hyperspectrale_cleanbands.tif"
spectral_library_dir = "Data/spectral_libraries/RELAB/"
mineral = "arsenopyrite"
```

La bibliothèque RELAB doit être organisée sous forme :

```text
RELAB/
    arsenopyrite/
        ref1.tab
        ref2.tab
    chalcopyrite/
        ref1.tab
        ref2.tab
```

### Étape 1 — Lissage et normalisation L2

Fonction :

```python
savgol_smooth_and_normalize(
    img_path=image_hyperspectrale_cleanbands,
    output_path=image_hyperspectrale_cleanbands_smooth_norm,
    window_length=9,
    polyorder=2,
    normalize="l2"
)
```

Sortie :

```text
image_hyperspectrale_cleanbands_smooth_norm.tif
```

Choix méthodologique :

- le lissage réduit le bruit spectral ;
- la normalisation L2 est adaptée aux méthodes de similarité spectrale ;
- elle rend les spectres comparables en forme, indépendamment de leur amplitude globale.

### Étape 2 — SAM + Matched Filter

Fonction :

```python
run_single_mineral_detection_from_tab_refs(
    img_tif_path=image_hyperspectrale_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_mf/" + mineral
)
```

Sorties :

| Fichier | Description |
|---|---|
| `sam_<mineral>_min.tif` | Carte SAM agrégée |
| `mf_<mineral>_max.tif` | Carte Matched Filter agrégée |
| `combo_sam_mf_<mineral>.tif` | Carte combinée SAM + MF |

Interprétation :

| Produit | Lecture |
|---|---|
| `SAM_min` | Plus la valeur est faible, plus la similarité spectrale est forte |
| `MF_max` | Plus la valeur est élevée, plus la réponse au spectre cible est forte |
| `COMBO` | Plus la valeur est élevée, plus le pixel est compatible avec la cible |

### Étape 3 — SAM + ACE

Fonction :

```python
run_single_mineral_ace_from_tab_refs(
    img_tif_path=image_hyperspectrale_cleanbands_smooth_norm,
    bands_csv=clean_wavelengths_csv,
    ref_root_dir=spectral_library_dir,
    mineral=mineral,
    out_dir=out_dir + "sam_ace/" + mineral
)
```

Sorties :

| Fichier | Description |
|---|---|
| `sam_<mineral>_min.tif` | Carte SAM agrégée |
| `ace_<mineral>_max.tif` | Carte ACE agrégée |
| `combo_sam_ace_<mineral>.tif` | Carte combinée SAM + ACE |

Interprétation :

| Produit | Lecture |
|---|---|
| `SAM_min` | Petite valeur = forte similarité |
| `ACE_max` | Grande valeur = forte détection statistique |
| `COMBO` | Grande valeur = pixel fortement compatible avec la cible |

### Gestion multi-références

Pour chaque minéral, plusieurs spectres de référence peuvent être utilisés.

Le pipeline calcule une carte par référence, puis agrège :

| Méthode | Agrégation |
|---|---|
| SAM | minimum des angles |
| MF | maximum des réponses |
| ACE | maximum des réponses |

Les spectres sont rejetés si :

- le fichier ne peut pas être lu ;
- le recouvrement spectral avec le cube est insuffisant ;
- le spectre de référence contient trop peu de bandes exploitables.

### Produits continus, pas des masques

Les sorties SAM, MF, ACE et COMBO ne sont pas des masques binaires.

Elles doivent être interprétées comme des cartes continues de score :

| Valeur | Interprétation |
|---|---|
| élevée pour MF / ACE / COMBO | forte compatibilité avec le minéral |
| faible pour SAM | forte similarité spectrale |
| NaN | pixel invalide |

Si un masque binaire est souhaité, il faut appliquer un seuil sur les cartes de score.

---

## 5. Différence entre les trois scripts

| Script | Rôle | Produit principal |
|---|---|---|
| `main.py` | Prétraitement général | Cube propre, masques eau/végétation, cube candidat minéral |
| `main_banddepth_mineral_detection.py` | Détection par indices ciblés | Masques minéraux et cartes de band depths |
| `main_spectralcomparison_mineral_detection.py` | Détection par bibliothèque spectrale | Cartes SAM, MF, ACE et scores combinés |

---

## 6. Choix méthodologiques importants

### Lissage sans normalisation pour les band depths

Les band depths exploitent les amplitudes relatives autour d’une absorption. Une normalisation L2 pourrait modifier ces relations. C’est pourquoi `normalize=None` est utilisé dans `main_banddepth_mineral_detection.py`.

### Lissage avec normalisation L2 pour SAM/MF/ACE

Les méthodes de comparaison spectrale s’intéressent davantage à la forme globale du spectre. La normalisation L2 rend les signatures comparables entre pixels et références.

### Nettoyage spectral avant toutes les détections

Les zones d’absorption atmosphérique autour de 1400 nm et 1900 nm sont supprimées avant les analyses afin de limiter les faux signaux liés à l’atmosphère.

### Exclusion eau / végétation

Le cube `enmap_mineral_candidates.tif` permet de concentrer les analyses minérales sur les pixels les plus pertinents, c’est-à-dire hors eau, hors végétation et hors pixels invalides.

---

## 7. Axes d’amélioration

### 7.1 Validation terrain

Les cartes produites devraient être comparées à des observations terrain, relevés géologiques ou cartes de référence indépendantes.

### 7.2 Optimisation des seuils

Les seuils utilisés pour les masques minéraux pourraient être optimisés automatiquement, par exemple à partir de données de validation ou par analyse statistique des distributions.

### 7.3 Meilleure gestion des pixels invalides

Les masques binaires minéraux codent actuellement `0` à la fois pour les pixels non détectés et les pixels invalides. Une convention à trois classes serait plus explicite :

| Label | Signification |
|---|---|
| `255` | Détecté |
| `0` | Non détecté |
| `-1` | Invalide |

### 7.4 Fusion des approches

Les résultats issus des band depths, SAM, MF et ACE pourraient être fusionnés dans une carte de probabilité commune.

### 7.5 Information spatiale

Le pipeline actuel est majoritairement pixel-wise. Des filtres morphologiques, une segmentation ou des contraintes spatiales pourraient réduire le bruit de type “salt-and-pepper”.

### 7.6 Pondération des spectres de référence

Toutes les références sont actuellement traitées de façon relativement équivalente. Un clustering ou une pondération des spectres de référence pourrait améliorer la robustesse.

### 7.7 Passage à l’échelle

Pour des scènes complètes, un traitement par tuiles, une parallélisation ou une approche Dask/GPU pourrait améliorer les performances.

---

## 8. Résumé final

Ce pipeline fournit une chaîne complète de traitement d’images hyperspectrales EnMAP :

- préparation radiométrique et spectrale ;
- masquage des pixels non exploitables ;
- calcul de produits de contrôle ;
- extraction des zones candidates minérales ;
- détection minérale par indices spectraux ;
- comparaison à des bibliothèques spectrales ;
- production de cartes continues et de masques interprétables.

Les trois scripts sont complémentaires et doivent être utilisés dans l’ordre suivant :

```bash
python main.py
python main_banddepth_mineral_detection.py
python main_spectralcomparison_mineral_detection.py
```
