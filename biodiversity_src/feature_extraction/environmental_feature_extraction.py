from pathlib import Path
import requests
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from scipy.ndimage import uniform_filter


def download_copernicus_dem_for_image(
    image_tif,
    out_dir,
    api_key,
    dem_filename="copernicus_dem_glo30.tif",
):
    image_tif = Path(image_tif)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path = out_dir / dem_filename

    if dem_path.exists():
        print("DEM déjà existant :", dem_path)
        return dem_path

    with rasterio.open(image_tif) as src:
        bounds = src.bounds
        crs = src.crs

        west, south, east, north = transform_bounds(
            crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=COP30"
        f"&west={west}"
        f"&south={south}"
        f"&east={east}"
        f"&north={north}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )

    print("Téléchargement du DEM Copernicus GLO-30...")
    response = requests.get(url)
    response.raise_for_status()

    with open(dem_path, "wb") as f:
        f.write(response.content)

    print("DEM sauvegardé :", dem_path)

    return dem_path


def reproject_dem_to_image_grid(image_tif, dem_tif):
    image_tif = Path(image_tif)
    dem_tif = Path(dem_tif)

    with rasterio.open(image_tif) as src_img:
        profile = src_img.profile.copy()
        dst_crs = src_img.crs
        dst_transform = src_img.transform
        dst_height = src_img.height
        dst_width = src_img.width

    dem_resampled = np.full(
        (dst_height, dst_width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(dem_tif) as src_dem:
        reproject(
            source=rasterio.band(src_dem, 1),
            destination=dem_resampled,
            src_transform=src_dem.transform,
            src_crs=src_dem.crs,
            src_nodata=src_dem.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dem_resampled, profile


def compute_slope_aspect(dem, transform):
    xres = abs(transform.a)
    yres = abs(transform.e)

    dz_dy, dz_dx = np.gradient(dem, yres, xres)

    slope = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    ).astype(np.float32)

    aspect = np.degrees(
        np.arctan2(-dz_dx, dz_dy)
    )

    aspect = ((aspect + 360) % 360).astype(np.float32)

    slope[~np.isfinite(dem)] = np.nan
    aspect[~np.isfinite(dem)] = np.nan

    return slope, aspect


def compute_tpi(dem, window_size=9):
    dem_filled = np.where(
        np.isfinite(dem),
        dem,
        np.nanmean(dem),
    )

    local_mean = uniform_filter(
        dem_filled,
        size=window_size,
        mode="nearest",
    )

    tpi = dem - local_mean
    tpi[~np.isfinite(dem)] = np.nan

    return tpi.astype(np.float32)


def compute_simple_twi(dem, slope, eps=1e-6):
    slope_rad = np.radians(slope)

    twi = np.log(
        1.0 / np.maximum(np.tan(slope_rad), eps)
    )

    twi[~np.isfinite(dem)] = np.nan
    twi[~np.isfinite(slope)] = np.nan

    return twi.astype(np.float32)



from scipy.ndimage import generic_filter, laplace


def compute_tri(dem, window_size=3):
    """
    Calcule le Terrain Ruggedness Index (TRI).

    TRI = moyenne des différences absolues entre l'altitude du pixel central
    et les altitudes de ses voisins.
    """

    dem = dem.astype(np.float32)

    def tri_func(values):
        center = values[len(values) // 2]

        if not np.isfinite(center):
            return np.nan

        neighbors = np.delete(values, len(values) // 2)
        neighbors = neighbors[np.isfinite(neighbors)]

        if neighbors.size == 0:
            return np.nan

        return np.mean(np.abs(neighbors - center))

    tri = generic_filter(
        dem,
        function=tri_func,
        size=window_size,
        mode="nearest",
    )

    tri[~np.isfinite(dem)] = np.nan

    return tri.astype(np.float32)


def compute_curvature(dem, transform):
    """
    Calcule une courbure topographique simple à partir du DEM.

    Valeurs positives/négatives :
    - selon la convention, elles indiquent des zones convexes ou concaves.
    - utile pour distinguer crêtes, replats et thalwegs.
    """

    dem = dem.astype(np.float32)

    xres = abs(transform.a)
    yres = abs(transform.e)

    # Laplacien normalisé par la résolution spatiale
    curvature = laplace(dem) / ((xres + yres) / 2.0) ** 2

    curvature[~np.isfinite(dem)] = np.nan

    return curvature.astype(np.float32)


def extract_topographic_features(
    image_tif,
    path_res,
    api_key,
    output_tif=None,
    tpi_window_size=9,
    tri_window_size=3,
):
    """
    Télécharge le DEM Copernicus GLO-30 sur l'emprise de l'image hyperspectrale,
    le reprojette sur la grille de l'image, puis calcule :

    - altitude
    - pente
    - exposition
    - TPI
    - TWI approximatif
    - TRI
    - courbure

    Returns
    -------
    topo_cube : np.ndarray
        Cube de forme (7, rows, cols)

    feature_names : list[str]
        Noms des variables topographiques.
    """

    image_tif = Path(image_tif)
    path_res = Path(path_res)

    dem_dir = path_res / "Copernicus_dem_glo30"
    dem_dir.mkdir(parents=True, exist_ok=True)

    dem_path = download_copernicus_dem_for_image(
        image_tif=image_tif,
        out_dir=dem_dir,
        api_key=api_key,
    )

    dem_resampled, profile = reproject_dem_to_image_grid(
        image_tif=image_tif,
        dem_tif=dem_path,
    )

    slope, aspect = compute_slope_aspect(
        dem_resampled,
        profile["transform"],
    )

    tpi = compute_tpi(
        dem_resampled,
        window_size=tpi_window_size,
    )

    twi = compute_simple_twi(
        dem_resampled,
        slope,
    )

    tri = compute_tri(
        dem_resampled,
        window_size=tri_window_size,
    )

    curvature = compute_curvature(
        dem_resampled,
        profile["transform"],
    )

    feature_names = [
        "altitude",
        "slope",
        "aspect",
        "tpi",
        "twi",
        "tri",
        "curvature",
    ]

    topo_cube = np.stack(
        [
            dem_resampled.astype(np.float32),
            slope.astype(np.float32),
            aspect.astype(np.float32),
            tpi.astype(np.float32),
            twi.astype(np.float32),
            tri.astype(np.float32),
            curvature.astype(np.float32),
        ],
        axis=0,
    )

    if output_tif is None:
        output_tif = dem_dir / "topographic_features.tif"
    else:
        output_tif = Path(output_tif)
        
    # On crée le dossier AVANT l'enregistrement du fichier
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        count=topo_cube.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    with rasterio.open(output_tif, "w", **out_profile) as dst:
        dst.write(topo_cube)

        for i, name in enumerate(feature_names, start=1):
            dst.set_band_description(i, name)

    print("Cube topographique sauvegardé :", output_tif)
    print("Shape :", topo_cube.shape)
    print("Features :", feature_names)

    return topo_cube, feature_names



from pathlib import Path
import requests
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def chelsa_bioclim_url(bio_id):
    return (
        "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/"
        f"climatologies/1981-2010/bio/CHELSA_bio{bio_id}_1981-2010_V.2.1.tif"
    )


def download_chelsa_bioclim(bio_id, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"CHELSA_bio{bio_id}_1981-2010_V.2.1.tif"

    if out_file.exists():
        print("Fichier déjà existant :", out_file)
        return out_file

    url = chelsa_bioclim_url(bio_id)
    print("Téléchargement :", url)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(out_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return out_file


def reproject_raster_to_image_grid(src_raster, image_profile):
    dst = np.full(
        (image_profile["height"], image_profile["width"]),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(src_raster) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=image_profile["transform"],
            dst_crs=image_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dst.astype(np.float32)


def extract_chelsa_features(
    image_tif,
    path_res,
    output_tif=None,
):
    """
    Extrait les variables climatiques CHELSA v2.1 alignées sur l'image hyperspectrale.

    Variables utilisées :
    - BIO1  : mean_annual_temperature
    - BIO4  : temperature_seasonality
    - BIO12 : annual_precipitation
    - BIO15 : precipitation_seasonality

    Returns
    -------
    climate_cube : np.ndarray
        Cube de forme (4, rows, cols)

    feature_names : list[str]
        Noms des variables climatiques.
    """

    image_tif = Path(image_tif)
    path_res = Path(path_res)

    climate_dir = path_res / "CHELSA"
    climate_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_tif) as src_img:
        image_profile = src_img.profile.copy()

    selected_bioclim = {
        "mean_annual_temperature": 1,
        "temperature_seasonality": 4,
        "annual_precipitation": 12,
        "precipitation_seasonality": 15,
    }

    climate_maps = []
    feature_names = []

    for feature_name, bio_id in selected_bioclim.items():

        chelsa_tif = download_chelsa_bioclim(
            bio_id=bio_id,
            out_dir=climate_dir,
        )

        arr = reproject_raster_to_image_grid(
            src_raster=chelsa_tif,
            image_profile=image_profile,
        )

        # Correction des unités CHELSA
        if bio_id == 1:
            arr = arr / 10.0 - 273.15

        elif bio_id == 4:
            arr = arr / 100.0

        elif bio_id == 12:
            arr = arr

        elif bio_id == 15:
            arr = arr / 10.0

        climate_maps.append(arr.astype(np.float32))
        feature_names.append(feature_name)

    climate_cube = np.stack(climate_maps, axis=0).astype(np.float32)

    if output_tif is None:
        output_tif = climate_dir / "chelsa_features.tif"
    else:
        output_tif = Path(output_tif)

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    out_profile = image_profile.copy()
    out_profile.update(
        count=climate_cube.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    with rasterio.open(output_tif, "w", **out_profile) as dst:
        dst.write(climate_cube)

        for i, name in enumerate(feature_names, start=1):
            dst.set_band_description(i, name)

    print("Cube CHELSA sauvegardé :", output_tif)
    print("Shape :", climate_cube.shape)
    print("Features :", feature_names)

    return climate_cube, feature_names


from pathlib import Path
import requests
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds


def download_terraclimate_file(variable, year, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"TerraClimate_{variable}_{year}.nc"

    if out_file.exists():
        print("Fichier déjà existant :", out_file)
        return out_file

    url = (
        "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/"
        f"TerraClimate_{variable}_{year}.nc"
    )

    print("Téléchargement :", url)

    response = requests.get(url)
    response.raise_for_status()

    with open(out_file, "wb") as f:
        f.write(response.content)

    return out_file


def reproject_climate_to_image_grid(
    climate_array,
    climate_transform,
    climate_crs,
    image_profile,
):
    dst = np.full(
        (image_profile["height"], image_profile["width"]),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=climate_array.astype(np.float32),
        destination=dst,
        src_transform=climate_transform,
        src_crs=climate_crs,
        dst_transform=image_profile["transform"],
        dst_crs=image_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return dst.astype(np.float32)

from datetime import date

def extract_climate_features(
    image_tif,
    path_res,
    acquisition_date,
    climate_window=10,
    output_tif=None,
):
    """
    Extraction de 8 caractéristiques bioclimatiques TerraClimate.

    Paramètres calculés :
        BIO1  : Température moyenne annuelle
        BIO4  : Saisonnalité des températures
        BIO6  : Température minimale du mois le plus froid
        BIO10 : Température moyenne du trimestre le plus chaud
        BIO12 : Précipitations annuelles
        BIO15 : Saisonnalité des précipitations
        BIO17 : Précipitations du trimestre le plus sec
        DEF   : Déficit hydrique estival (juin-août)

    Les indicateurs sont calculés annuellement, puis moyennés
    sur les climate_window années précédant l'acquisition.

    Returns
    -------
    climate_cube : np.ndarray
        Cube (8, rows, cols)

    feature_names : list[str]
        Noms des caractéristiques climatiques.
    """

    image_tif = Path(image_tif)
    path_res = Path(path_res)

    # =====================================================
    # 1. DATE D'ACQUISITION
    # =====================================================

    acquisition_date = date.fromisoformat(
        str(acquisition_date)[:10]
    )

    end_year = acquisition_date.year
    start_year = end_year - climate_window
    last_year = end_year - 1

    print(f"Date d'acquisition : {acquisition_date}")
    print(f"Période climatique : {start_year}-{last_year}")

    # =====================================================
    # 2. DOSSIER TERRACLIMATE
    # =====================================================

    climate_dir = path_res / "TerraClimate"
    climate_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 3. EMPRISE DE L'IMAGE ENMAP
    # =====================================================

    with rasterio.open(image_tif) as src:

        image_profile = src.profile.copy()

        image_bounds = src.bounds
        image_crs = src.crs

        west, south, east, north = transform_bounds(
            image_crs,
            "EPSG:4326",
            image_bounds.left,
            image_bounds.bottom,
            image_bounds.right,
            image_bounds.top,
        )

    # =====================================================
    # 4. VARIABLES TERRACLIMATE
    # =====================================================

    variables = ["tmax", "tmin", "ppt", "def"]

    feature_names = [
        "bio1_mean_annual_temperature",
        "bio4_temperature_seasonality",
        "bio6_min_temperature_coldest_month",
        "bio10_mean_temperature_warmest_quarter",
        "bio12_annual_precipitation",
        "bio15_precipitation_seasonality",
        "bio17_precipitation_driest_quarter",
        "summer_water_deficit",
    ]

    feature_lists = {
        name: [] for name in feature_names
    }

    # =====================================================
    # 5. FONCTION : TRIMESTRES GLISSANTS
    # =====================================================

    def rolling_quarter(da, operation="mean"):

        rolling = da.rolling(
            time=3,
            min_periods=3
        )

        if operation == "mean":
            return rolling.mean().isel(time=slice(2, None))

        if operation == "sum":
            return rolling.sum().isel(time=slice(2, None))

        raise ValueError(f"Opération inconnue : {operation}")

    # =====================================================
    # 6. TRAITEMENT DES ANNÉES
    # =====================================================

    for year in range(start_year, last_year + 1):

        print(f"\nTraitement année {year}")

        nc_paths = {
            var: download_terraclimate_file(
                variable=var,
                year=year,
                out_dir=climate_dir,
            )
            for var in variables
        }

        datasets = []

        def load_var(var_name):

            ds = xr.open_dataset(nc_paths[var_name])
            datasets.append(ds)

            da = ds[var_name]

            da = da.sel(
                lon=slice(west, east),
                lat=slice(north, south),
            )

            # Charger en mémoire avant fermeture des fichiers
            return da.load()

        try:

            tmax = load_var("tmax")
            tmin = load_var("tmin")
            ppt = load_var("ppt")
            deficit = load_var("def")

            # Conversion des valeurs brutes si nécessaire :
            # cette fonction suppose que les données chargées
            # sont déjà en °C et mm.

            tmean = (tmax + tmin) / 2.0

            # =============================================
            # BIO1 : TEMPÉRATURE MOYENNE ANNUELLE
            # =============================================

            bio1 = tmean.mean(dim="time")

            # =============================================
            # BIO4 : SAISONNALITÉ DES TEMPÉRATURES
            # =============================================

            bio4 = tmean.std(dim="time") * 100.0

            # =============================================
            # BIO6 : TEMPÉRATURE MINIMALE
            # DU MOIS LE PLUS FROID
            # =============================================

            bio6 = tmin.min(dim="time")

            # =============================================
            # BIO10 : TEMPÉRATURE DU TRIMESTRE
            # LE PLUS CHAUD
            # =============================================

            quarterly_temp = rolling_quarter(
                tmean,
                operation="mean"
            )

            bio10 = quarterly_temp.max(dim="time")

            # =============================================
            # BIO12 : PRÉCIPITATIONS ANNUELLES
            # =============================================

            bio12 = ppt.sum(dim="time")

            # =============================================
            # BIO15 : SAISONNALITÉ DES PRÉCIPITATIONS
            # =============================================

            ppt_mean = ppt.mean(dim="time")
            ppt_std = ppt.std(dim="time")

            bio15 = xr.where(
                ppt_mean > 0,
                (ppt_std / ppt_mean) * 100.0,
                np.nan
            )

            # =============================================
            # BIO17 : PRÉCIPITATIONS DU TRIMESTRE
            # LE PLUS SEC
            # =============================================

            quarterly_ppt = rolling_quarter(
                ppt,
                operation="sum"
            )

            bio17 = quarterly_ppt.min(dim="time")

            # =============================================
            # DÉFICIT HYDRIQUE ESTIVAL
            # =============================================

            summer_deficit = deficit.sel(
                time=deficit["time"].dt.month.isin(
                    [6, 7, 8]
                )
            ).sum(dim="time")

            # =============================================
            # STOCKAGE
            # =============================================

            annual_features = {
                "bio1_mean_annual_temperature": bio1,
                "bio4_temperature_seasonality": bio4,
                "bio6_min_temperature_coldest_month": bio6,
                "bio10_mean_temperature_warmest_quarter": bio10,
                "bio12_annual_precipitation": bio12,
                "bio15_precipitation_seasonality": bio15,
                "bio17_precipitation_driest_quarter": bio17,
                "summer_water_deficit": summer_deficit,
            }

            for name, da in annual_features.items():
                feature_lists[name].append(da)

        finally:

            for ds in datasets:
                ds.close()

    # =====================================================
    # 7. MOYENNES PLURIANNUELLES
    # =====================================================

    climate_features = {}

    for name in feature_names:

        climate_features[name] = xr.concat(
            feature_lists[name],
            dim="year"
        ).mean(dim="year")

    # =====================================================
    # 8. REPROJECTION SUR LA GRILLE ENMAP
    # =====================================================

    climate_maps = []

    for name in feature_names:

        da = climate_features[name]

        arr = da.values.astype(np.float32)

        lat = da["lat"].values
        lon = da["lon"].values

        if len(lat) < 2 or len(lon) < 2:
            raise ValueError(
                "Emprise TerraClimate trop petite : "
                "moins de deux pixels sur un axe."
            )

        lon_res = abs(float(np.diff(lon).mean()))
        lat_res = abs(float(np.diff(lat).mean()))

        climate_transform = from_bounds(
            west=float(lon.min()) - lon_res / 2,
            south=float(lat.min()) - lat_res / 2,
            east=float(lon.max()) + lon_res / 2,
            north=float(lat.max()) + lat_res / 2,
            width=arr.shape[1],
            height=arr.shape[0],
        )

        arr_reprojected = reproject_climate_to_image_grid(
            climate_array=arr,
            climate_transform=climate_transform,
            climate_crs="EPSG:4326",
            image_profile=image_profile,
        )

        climate_maps.append(arr_reprojected)

    climate_cube = np.stack(
        climate_maps,
        axis=0
    ).astype(np.float32)

    # =====================================================
    # 9. SAUVEGARDE
    # =====================================================

    if output_tif is None:

        output_tif = (
            climate_dir
            / f"bioclim_features_{start_year}_{last_year}.tif"
        )

    else:
        output_tif = Path(output_tif) / f"_{start_year}_{last_year}.tif"

    output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_profile = image_profile.copy()

    out_profile.update(
        count=climate_cube.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    with rasterio.open(
        output_tif,
        "w",
        **out_profile
    ) as dst:

        dst.write(climate_cube)

        for i, name in enumerate(feature_names, start=1):
            dst.set_band_description(i, name)

    print("\nCube climatique sauvegardé :", output_tif)
    print("Shape :", climate_cube.shape)
    print("Features :", feature_names)

    return climate_cube, feature_names


def extract_climate_features_old(
    image_tif,
    path_res,
    end_year,
    climate_window=10,
    output_tif=None,
):
    """
    Extrait les variables climatiques TerraClimate sans Google Earth Engine.

    Pour la classification des essences, on utilise une moyenne pluriannuelle
    sur les climate_window années précédant l'année de l'image.

    Exemple :
    end_year=2025, climate_window=10
    -> période utilisée : 2015 à 2024

    Variables finales :
    - mean_temperature
    - annual_precipitation
    - summer_water_deficit
    - annual_pet
    - temperature_seasonality
    - precipitation_seasonality

    Returns
    -------
    climate_cube : np.ndarray
        Cube de forme (6, rows, cols)

    feature_names : list[str]
        Noms des variables climatiques.
    """

    image_tif = Path(image_tif)
    path_res = Path(path_res)

    climate_dir = path_res / "TerraClimate"
    climate_dir.mkdir(parents=True, exist_ok=True)

    print('climate_dir ======', climate_dir)

    start_year = end_year - climate_window
    last_year = end_year - 1

    print(f"Période climatique utilisée : {start_year}-{last_year}")

    variables = ["tmax", "tmin", "ppt", "def", "pet"]

    with rasterio.open(image_tif) as src:
        image_profile = src.profile.copy()
        image_bounds = src.bounds
        image_crs = src.crs

        west, south, east, north = transform_bounds(
            image_crs,
            "EPSG:4326",
            image_bounds.left,
            image_bounds.bottom,
            image_bounds.right,
            image_bounds.top,
        )

    annual_tmean_list = []
    annual_ppt_list = []
    summer_def_list = []
    annual_pet_list = []
    temp_seasonality_list = []
    precip_seasonality_list = []

    for year in range(start_year, last_year + 1):

        print(f"\nTraitement année {year}")

        nc_paths = {
            var: download_terraclimate_file(
                variable=var,
                year=year,
                out_dir=climate_dir,
            )
            for var in variables
        }

        def load_var(var_name):
            ds = xr.open_dataset(nc_paths[var_name])
            da = ds[var_name]

            # TerraClimate : latitude décroissante généralement
            da = da.sel(
                lon=slice(west, east),
                lat=slice(north, south),
            )

            return da

        tmax = load_var("tmax")
        tmin = load_var("tmin")
        ppt = load_var("ppt")
        deficit = load_var("def")
        pet = load_var("pet")

        # Température moyenne mensuelle
        tmean_monthly = (tmax + tmin) / 2.0

        annual_tmean_list.append(
            tmean_monthly.mean(dim="time")
        )

        annual_ppt_list.append(
            ppt.sum(dim="time")
        )

        summer_def_list.append(
            deficit.sel(
                time=deficit["time"].dt.month.isin([6, 7, 8])
            ).sum(dim="time")
        )

        annual_pet_list.append(
            pet.sum(dim="time")
        )

        temp_seasonality_list.append(
            tmean_monthly.std(dim="time")
        )

        precip_seasonality_list.append(
            ppt.std(dim="time")
        )

    mean_temperature = xr.concat(
        annual_tmean_list,
        dim="year"
    ).mean(dim="year")

    annual_precipitation = xr.concat(
        annual_ppt_list,
        dim="year"
    ).mean(dim="year")

    summer_water_deficit = xr.concat(
        summer_def_list,
        dim="year"
    ).mean(dim="year")

    annual_pet = xr.concat(
        annual_pet_list,
        dim="year"
    ).mean(dim="year")

    temperature_seasonality = xr.concat(
        temp_seasonality_list,
        dim="year"
    ).mean(dim="year")

    precipitation_seasonality = xr.concat(
        precip_seasonality_list,
        dim="year"
    ).mean(dim="year")

    climate_features = {
        "mean_temperature": mean_temperature,
        "annual_precipitation": annual_precipitation,
        "summer_water_deficit": summer_water_deficit,
        "annual_pet": annual_pet,
        "temperature_seasonality": temperature_seasonality,
        "precipitation_seasonality": precipitation_seasonality,
    }

    climate_maps = []
    feature_names = []

    for name, da in climate_features.items():

        arr = da.values.astype(np.float32)

        lat = da["lat"].values
        lon = da["lon"].values

        climate_transform = from_bounds(
            west=float(lon.min()),
            south=float(lat.min()),
            east=float(lon.max()),
            north=float(lat.max()),
            width=arr.shape[1],
            height=arr.shape[0],
        )

        arr_reprojected = reproject_climate_to_image_grid(
            climate_array=arr,
            climate_transform=climate_transform,
            climate_crs="EPSG:4326",
            image_profile=image_profile,
        )

        climate_maps.append(arr_reprojected)
        feature_names.append(name)

    climate_cube = np.stack(climate_maps, axis=0).astype(np.float32)

    if output_tif is None:
        output_tif = (
            climate_dir
            / f"climate_features_{start_year}_{last_year}.tif"
        )
    else:
        output_tif = Path(output_tif)

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    out_profile = image_profile.copy()
    out_profile.update(
        count=climate_cube.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    with rasterio.open(output_tif, "w", **out_profile) as dst:
        dst.write(climate_cube)

        for i, name in enumerate(feature_names, start=1):
            dst.set_band_description(i, name)

    print("\nCube climatique sauvegardé :", output_tif)
    print("Shape :", climate_cube.shape)
    print("Features :", feature_names)

    return climate_cube, feature_names