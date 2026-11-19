"""
Streamlit front-end for the EnMAP preprocessing pipeline (see main_gui.py).

Launch with:
    streamlit run clean_src/00-archives/main_streamlit.py
"""
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for the Streamlit server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.enmap_band_utils import recover_wavelet_band_info
from enmap_quality_mask import mask_enmap_hyperspectral_cube
from utils.enmap_crop_image import crop_hyperspectral_tif
from utils.analyse_hyperspectral_image import analyze_rescaled_cube_with_wavelengths, check_input_raster_validity
from enmap_water_indices import compute_mndwi_and_water_mask
from enmap_vegetation_indices import compute_vegetation_indices_wdi_vii
from preprocessing.mask_enmap import apply_water_veg_mask
from preprocessing.enmap_clean_bands import clean_bands_enmap_from_csv
from preprocessing.rescaling_image import rescale_enmap_cube_simple
from utils.enmap_rgb_extraction import hyperspectral_to_rgb

DEFAULT_CONFIGS_DIR = "/home/yassine/programming/MIWARE-HYP/preprocess_gui"

st.set_page_config(page_title="MIWARE-HYP – EnMAP preprocessing", layout="wide")


# ============================================================
# Image helpers
# ============================================================


@st.cache_data(show_spinner=False)
def render_index_map(path: str, mtime: float, title: str, cmap: str = "viridis") -> bytes:
    """Render a single-band index raster with matplotlib (2-98 % stretch + colorbar) as PNG bytes.

    NaN / non-finite pixels are drawn in black. `mtime` invalidates the cache when the file changes.
    """
    arr = read_raster(path, mtime).astype("float32")
    finite = np.isfinite(arr)
    vmin, vmax = (np.percentile(arr[finite], [2, 98]) if finite.any() else (0.0, 1.0))

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("black")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """2-98 % stretch per channel, (H, W, 3) -> uint8."""
    rgb = rgb.astype("float32")
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for ch in range(3):
        band = rgb[..., ch]
        valid = np.isfinite(band) & (band != 0)
        if not valid.any():
            continue
        lo, hi = np.percentile(band[valid], [2, 98])
        out[..., ch] = (np.clip((band - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    return out


def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Return an array as (H, W) or (H, W, C)."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return arr


@st.cache_data(show_spinner=False)
def read_raster(path: str, mtime: float) -> np.ndarray:
    """Read a small (1-3 band) raster. `mtime` invalidates the cache when the file changes."""
    try:
        return to_hwc(tifffile.imread(path))
    except Exception:
        with rasterio.open(path) as src:
            return to_hwc(src.read())


@st.cache_data(show_spinner=False)
def read_cube_rgb(cube_path: str, bands_csv: str, mtime: float, targets=(650, 560, 480)) -> np.ndarray:
    """Build a true-colour composite of a hyperspectral cube from the closest bands."""
    df = pd.read_csv(bands_csv)
    wv = df["wavelength_nm"].to_numpy()
    with rasterio.open(cube_path) as src:
        idx = [int(np.argmin(np.abs(wv - t))) + 1 for t in targets]
        idx = [min(i, src.count) for i in idx]
        rgb = np.stack([src.read(i).astype("float32") for i in idx], axis=-1)
        if src.nodata is not None and not np.isnan(src.nodata):
            rgb[rgb == src.nodata] = np.nan
    return stretch_rgb(np.nan_to_num(rgb, nan=0.0))


def show_image(path: Path, caption: str, kind: str = "auto", cmap: str = "viridis"):
    """Render a raster file in the page."""
    if not path.exists():
        st.info(f"Not produced yet: `{path.name}`")
        return
    arr = read_raster(str(path), path.stat().st_mtime)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        img = arr[..., :3]
        if img.dtype != np.uint8:
            img = stretch_rgb(img)
        st.image(img, caption=caption, width="stretch")
        return
    if kind == "mask":
        img = np.full((*arr.shape, 3), 160, dtype=np.uint8)
        img[arr == 255] = (255, 255, 255)
        img[arr == -1] = (200, 30, 30)
        st.image(img, caption=caption, width="stretch")
        return
    st.image(render_index_map(str(path), path.stat().st_mtime, caption, cmap), width="stretch")


# ============================================================
# Pipeline
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def output_paths(config: dict, prefix: str) -> dict:
    res = Path(config["Path_res"])
    water_dir = res / "Water_indice_outputs"
    veg_dir = res / "Vegetation_indice_outputs"
    return {
        "res": res,
        "bands_csv": res / "enmap_bands_full.csv",
        "reflectance": res / "image_hyperspectrale_reflectance.tif",
        "clean": res / "image_hyperspectrale_clean.tif",
        "crop": res / "image_hyperspectrale_crop.tif",
        "rgb": res / "image_hyperspectrale_rgb.tif",
        "cleanbands": res / "image_hyperspectrale_cleanbands.tif",
        "clean_bands_csv": res / "enmap_clean_bands_full.csv",
        "qc_csv": res / "band_qc_report.csv",
        "water_dir": water_dir,
        "veg_dir": veg_dir,
        "water_mask": water_dir / f"{prefix}_WATER_MASK.tiff",
        "veg_mask": veg_dir / f"{prefix}_VEG_MASK.tiff",
        "mineral": res / "enmap_mineral_candidates.tif",
    }


def build_steps(config: dict, p: dict, params: dict):
    """Return the ordered list of (name, output_to_check, callable) pipeline steps."""
    data = Path(config["Path_data"])
    image_hyp = data / config["image_hyp"]
    mask_files = [data / config[k] for k in
                  ("Cloud_mask", "Haze_mask", "Cirrus_mask", "CloudShadow_mask", "Snow_mask", "TestFlags_mask")]
    crop = params["crop"]
    cube_for_next = p["crop"] if crop else p["clean"]

    def s0():
        check_input_raster_validity(image_hyp)

    def s1():
        df = recover_wavelet_band_info(data / config["xml_path"], out_csv=p["bands_csv"])
        print(df.head(10).to_string(index=False))

    def s2():
        rescale_enmap_cube_simple(input_tif=image_hyp, output_tif=p["reflectance"], scale_factor=10000.0)

    def s3():
        mask_enmap_hyperspectral_cube(cube_tif=p["reflectance"], mask_files=mask_files, out_tif=p["clean"])

    def s4():
        crop_hyperspectral_tif(p["clean"], p["crop"], config["Site_coords"])

    def s5():
        hyperspectral_to_rgb(cube_tif=cube_for_next, bands_csv=p["bands_csv"], out_rgb_tif=p["rgb"],
                             target_R=params["R"], target_G=params["G"], target_B=params["B"])

    def s6():
        summary = clean_bands_enmap_from_csv(
            img_path=cube_for_next, bands_csv=p["bands_csv"], output_path=p["cleanbands"],
            output_bands_csv=p["clean_bands_csv"], band_id_is_one_based=True,
            csv_band_id_is_one_based_out=True, drop_edges=(2, 2),
            exclude_ranges_nm=[(1340, 1460), (1800, 1960)], use_fwhm_margin=True, fwhm_factor=0.5)
        print(summary["nbands_in"], "->", summary["nbands_out"])
        print("Removed wavelengths (nm):", summary["removed_wavelengths_nm"])

    def s7():
        analyze_rescaled_cube_with_wavelengths(
            cube_tif=p["cleanbands"], bands_csv=p["clean_bands_csv"], out_csv=p["qc_csv"],
            min_valid=-0.1, max_valid=1.5, nan_threshold_pct=50.0, outlier_threshold_pct=5.0,
            band_id_is_one_based=True)

    def s8():
        compute_mndwi_and_water_mask(tif_path=p["cleanbands"], wavelengths_csv=p["clean_bands_csv"],
                                     outdir=p["water_dir"], prefix=params["prefix"],
                                     mndwi_th=params["mndwi_th"], verbose=True)

    def s9():
        compute_vegetation_indices_wdi_vii(tif_path=p["cleanbands"], wavelengths_csv=p["clean_bands_csv"],
                                           outdir=p["veg_dir"], prefix=params["prefix"],
                                           ndvi_th=params["ndvi_th"], verbose=True)

    def s10():
        apply_water_veg_mask(img_path=p["cleanbands"], water_mask_path=p["water_mask"],
                             veg_mask_path=p["veg_mask"], output_path=p["mineral"])

    steps = [
        ("0) Input validity check", None, s0),
        ("1) Band information (XML → CSV)", p["bands_csv"], s1),
        ("2) Rescaling to reflectance", p["reflectance"], s2),
        ("3) Quality masks (cloud, haze, …)", p["clean"], s3),
    ]
    if crop:
        steps.append(("4) Crop", p["crop"], s4))
    steps += [
        ("5) RGB composite", p["rgb"], s5),
        ("6) Clean bands", p["cleanbands"], s6),
        ("7) Band QC", p["qc_csv"], s7),
        ("8) Water index & mask", p["water_mask"], s8),
        ("9) Vegetation indices & mask", p["veg_mask"], s9),
        ("10) Water + vegetation mask → mineral candidates", p["mineral"], s10),
    ]
    return steps


def run_pipeline(config: dict, p: dict, params: dict, reuse: bool):
    p["res"].mkdir(parents=True, exist_ok=True)
    steps = build_steps(config, p, params)
    progress = st.progress(0.0)
    for i, (name, out, fn) in enumerate(steps, start=1):
        if reuse and out is not None and Path(out).exists():
            st.write(f"⏭️ {name} — reused existing `{Path(out).name}`")
            progress.progress(i / len(steps))
            continue
        with st.status(name, expanded=False) as status:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fn()
            except Exception:
                st.code(buf.getvalue() + "\n" + traceback.format_exc())
                status.update(label=f"❌ {name}", state="error", expanded=True)
                st.error(f"Pipeline stopped at step: {name}")
                return False
            log = buf.getvalue().strip()
            if log:
                st.code(log)
            status.update(label=f"✅ {name}", state="complete")
        progress.progress(i / len(steps))
    return True


# ============================================================
# UI
# ============================================================

st.title("MIWARE-HYP · EnMAP preprocessing")

with st.sidebar:
    st.header("Configuration")
    configs_dir = Path(st.text_input("Config folder", DEFAULT_CONFIGS_DIR))
    config_paths = sorted(configs_dir.glob("*.json")) if configs_dir.is_dir() else []
    if not config_paths:
        st.error(f"No JSON config found in {configs_dir}")
        st.stop()
    config_path = st.selectbox("Config file", config_paths, format_func=lambda x: x.name)
    config = load_config(config_path)

    st.header("Parameters")
    prefix = st.text_input("Output prefix", "enmap_salsigne")
    crop = st.checkbox("Crop to Site_coords", value=bool(config.get("crop")),
                       disabled=not config.get("Site_coords"))
    mndwi_th = st.number_input("MNDWI water threshold", value=0.55, step=0.05, format="%.2f")
    ndvi_th = st.number_input("NDVI vegetation threshold", value=0.30, step=0.05, format="%.2f")
    with st.expander("RGB target wavelengths (nm)"):
        R = st.number_input("Red", value=650)
        G = st.number_input("Green", value=560)
        B = st.number_input("Blue", value=480)
    reuse = st.checkbox("Skip steps whose output already exists", value=True)
    run = st.button("▶ Run pipeline", type="primary", width="stretch")

    with st.expander("Raw config"):
        st.json(config)

params = dict(prefix=prefix, crop=crop, mndwi_th=mndwi_th, ndvi_th=ndvi_th, R=R, G=G, B=B)
p = output_paths(config, prefix)

if run:
    with st.container(border=True):
        st.subheader("Run log")
        ok = run_pipeline(config, p, params, reuse)
    if ok:
        st.success(f"Pipeline finished. Results in `{p['res']}`")

st.caption(f"Results folder: `{p['res']}`")

tab_rgb, tab_water, tab_veg, tab_min, tab_qc = st.tabs(
    ["RGB", "Water", "Vegetation", "Mineral candidates", "Bands & QC"])

with tab_rgb:
    c1, c2 = st.columns(2)
    with c1:
        show_image(p["rgb"], "RGB composite (after quality masking)")
    with c2:
        if p["reflectance"].exists() and p["bands_csv"].exists():
            st.image(read_cube_rgb(str(p["reflectance"]), str(p["bands_csv"]),
                                   p["reflectance"].stat().st_mtime, (R, G, B)),
                     caption="Reflectance before quality masking", width="stretch")
        else:
            st.info("Reflectance image not produced yet.")

with tab_water:
    c1, c2 = st.columns(2)
    with c1:
        show_image(p["water_dir"] / f"{prefix}_MNDWI_Xu.tiff", "MNDWI (Xu)", cmap="BrBG")
    with c2:
        show_image(p["water_dir"] / f"{prefix}_WATER_MASK_VISUAL.tiff",
                   "Water mask — blue: water · grey: non-water · red: excluded")

with tab_veg:
    show_image(p["veg_dir"] / f"{prefix}_VEG_MASK_VISUAL.tiff",
               "Vegetation mask — green: vegetation · grey: non-vegetation · red: excluded")
    st.markdown("##### Vegetation indices")
    indices = [("NDVI", "RdYlGn"), ("GNDVI", "RdYlGn"), ("NDRE", "RdYlGn"), ("MSAVI", "RdYlGn"),
               ("NDWI_Gao", "BrBG"), ("VII_Zhang2012", "viridis"), ("LWAI_968", "viridis"),
               ("LWAI_1181", "viridis"), ("WDI_article", "viridis")]
    cols = st.columns(3)
    for i, (name, cmap) in enumerate(indices):
        with cols[i % 3]:
            show_image(p["veg_dir"] / f"{prefix}_{name}.tiff", name, cmap=cmap)

with tab_min:
    if p["mineral"].exists() and p["clean_bands_csv"].exists():
        st.image(read_cube_rgb(str(p["mineral"]), str(p["clean_bands_csv"]),
                               p["mineral"].stat().st_mtime, (R, G, B)),
                 caption="Mineral candidates (bare soil: water & vegetation masked out)",
                 width="stretch")
    else:
        st.info("Mineral candidates image not produced yet.")

with tab_qc:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### All bands")
        if p["bands_csv"].exists():
            st.dataframe(pd.read_csv(p["bands_csv"]), height=300)
    with c2:
        st.markdown("##### Clean bands")
        if p["clean_bands_csv"].exists():
            st.dataframe(pd.read_csv(p["clean_bands_csv"]), height=300)
    st.markdown("##### Band QC report")
    if p["qc_csv"].exists():
        qc = pd.read_csv(p["qc_csv"])
        st.dataframe(qc, height=350)
        num = [c for c in qc.columns if c not in ("band_id", "band_index") and pd.api.types.is_numeric_dtype(qc[c])]
        if "wavelength_nm" in qc.columns and num:
            metric = st.selectbox("Plot metric vs wavelength", [c for c in num if c != "wavelength_nm"])
            st.line_chart(qc.set_index("wavelength_nm")[metric])
    else:
        st.info("QC report not produced yet.")
