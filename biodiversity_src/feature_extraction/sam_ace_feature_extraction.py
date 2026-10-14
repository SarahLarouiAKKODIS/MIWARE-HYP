from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import rasterio
from typing import Optional

from spectral.algorithms import spectral_angles


def safe_filename(name: str) -> str:
    name = name.lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def sam_to_score(sam: np.ndarray, sam_scale: float = 0.1) -> np.ndarray:
    """
    Convertit l'angle SAM en score [0..1] (plus grand = meilleur).
    sam_scale en radians ~ 0.1 est un bon départ (à ajuster).
    """
    sam = sam.astype(np.float32)
    out = np.exp(-sam / float(sam_scale)).astype(np.float32)
    out[~np.isfinite(sam)] = np.nan
    return out

def combine_sam_ace(
    sam_min: np.ndarray,
    ace_max: np.ndarray,
    sam_scale: float = 0.1,
    mode: str = "product",   # "product" ou "mean"
) -> np.ndarray:
    s_score = sam_to_score(sam_min, sam_scale=sam_scale)

    # ACE est déjà pseudo-normalisé entre 0 et 1
    a_score = ace_max.astype(np.float32).copy()
    a_score[~np.isfinite(a_score)] = np.nan
    a_score = np.clip(a_score, 0.0, 1.0)

    if mode == "mean":
        combo = 0.5 * (s_score + a_score)
    else:
        combo = s_score * a_score

    combo[~np.isfinite(combo)] = np.nan
    return combo.astype(np.float32)

# -----------------------------
# ACE
# -----------------------------
def ace_map_from_background_stats(
    img_brc: np.ndarray,          # (bands, rows, cols)
    target_b: np.ndarray,         # (bands,)
    bg_mean_b: np.ndarray,        # (bands,)
    bg_cov_bb: np.ndarray,        # (bands, bands)
    valid_mask_rc: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calcule une carte ACE à partir des stats de fond.
    Retourne une carte (rows, cols) en float32 dans [0, 1] en pratique.
    """
    img_brc = np.asarray(img_brc, dtype=np.float32)
    target_b = np.asarray(target_b, dtype=np.float32).reshape(-1)
    bg_mean_b = np.asarray(bg_mean_b, dtype=np.float32).reshape(-1)
    bg_cov_bb = np.asarray(bg_cov_bb, dtype=np.float32)

    bands, rows, cols = img_brc.shape
    if target_b.shape[0] != bands:
        raise ValueError(f"target_b incompatible: {target_b.shape[0]} vs bands={bands}")
    if bg_mean_b.shape[0] != bands:
        raise ValueError(f"bg_mean_b incompatible: {bg_mean_b.shape[0]} vs bands={bands}")
    if bg_cov_bb.shape != (bands, bands):
        raise ValueError(f"bg_cov_bb incompatible: {bg_cov_bb.shape} vs ({bands}, {bands})")

    X = img_brc.reshape(bands, -1).T   # (n_pix, bands)
    finite = np.isfinite(X).all(axis=1)

    if valid_mask_rc is not None:
        finite &= valid_mask_rc.reshape(-1)

    ace = np.full(X.shape[0], np.nan, dtype=np.float32)
    if not np.any(finite):
        return ace.reshape(rows, cols)

    Xv = X[finite, :] - bg_mean_b[None, :]
    tv = target_b - bg_mean_b

    inv_cov = np.linalg.pinv(bg_cov_bb).astype(np.float32)

    inv_t = inv_cov @ tv
    t_inv_t = float(tv @ inv_t)

    if not np.isfinite(t_inv_t) or t_inv_t <= eps:
        return ace.reshape(rows, cols)

    X_inv = Xv @ inv_cov                # (n_valid, bands)
    x_inv_x = np.sum(X_inv * Xv, axis=1)
    x_inv_t = Xv @ inv_t

    denom = t_inv_t * x_inv_x
    good = np.isfinite(denom) & (denom > eps)

    ace_valid = np.full(Xv.shape[0], np.nan, dtype=np.float32)
    ace_valid[good] = (x_inv_t[good] ** 2) / denom[good]

    ace_valid = np.clip(ace_valid, 0.0, 1.0).astype(np.float32)
    ace[finite] = ace_valid

    return ace.reshape(rows, cols).astype(np.float32)

def compute_best_sam_ace_species_from_reference_csvs(
    img,
    profile,
    reference_csv_paths,
    out_dir,
    *,
    class_id_map=None,      # ex: {"Hêtre": 8, "Douglas": 6}
    valid_pixels_mask=None,
    nodata=np.nan,
    compress="lzw",
    sam_scale=0.1,
):
    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(f"img doit être (bands, rows, cols), reçu {img.shape}")

    reference_csv_paths = [Path(p) for p in reference_csv_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands, rows, cols = img.shape

    #############################################################################

    # print("Shape image :", img.shape)

    # finite_per_band = np.isfinite(img).sum(axis=(1, 2))

    # print("Pixels par bande :", rows * cols)
    # print("Bandes totalement NaN :", np.where(finite_per_band == 0)[0] + 1)
    # print("Nombre de bandes totalement NaN :", np.sum(finite_per_band == 0))

    # print(
    #     "Pixels valides sur toutes les bandes :",
    #     np.isfinite(img).all(axis=0).sum()
    # )

    # print(
    #     "Pixels ayant au moins une bande valide :",
    #     np.isfinite(img).any(axis=0).sum()
    # )

    # finite_count = np.isfinite(img).sum(axis=0)

    # # On regarde uniquement les pixels qui ont au moins une donnée
    # veg = finite_count > 0

    # values = finite_count[veg]

    # print("Nombre pixels végétation :", values.size)
    # print("Min bandes valides       :", values.min())
    # print("Max bandes valides       :", values.max())
    # print("Médiane                   :", np.median(values))
    # print("Moyenne                   :", values.mean())

    # values_unique, counts = np.unique(values, return_counts=True)

    # print("\nDistribution :")

    # for n, count in zip(values_unique, counts):
    #     print(
    #         f"{n:3d}/213 bandes : "
    #         f"{count:5d} pixels "
    #         f"({100 * count / values.size:.2f} %)"
    #     )

    
    #############################################################################


    valid = np.isfinite(img).all(axis=0)

    if valid_pixels_mask is not None:
        valid_pixels_mask = np.asarray(valid_pixels_mask).astype(bool)
        if valid_pixels_mask.shape != (rows, cols):
            raise ValueError(
                f"valid_pixels_mask={valid_pixels_mask.shape}, image={(rows, cols)}"
            )
        valid &= valid_pixels_mask

    if not np.any(valid):
        raise ValueError("Aucun pixel valide dans l'image.")

    prof_float = profile.copy()
    prof_float.update(count=1, dtype="float32", nodata=nodata, compress=compress)

    prof_class = profile.copy()
    prof_class.update(count=1, dtype="int16", nodata=0, compress=compress)

    ref_dfs = [pd.read_csv(p) for p in reference_csv_paths]

    class_columns = [
        c for c in ref_dfs[0].columns
        if c != "wavelength_nm"
    ]

    if class_id_map is None:
        class_id_map = {
            class_name: i + 1
            for i, class_name in enumerate(class_columns)
        }

    for csv_path, df in zip(reference_csv_paths, ref_dfs):
        if "wavelength_nm" not in df.columns:
            raise ValueError(f"Colonne wavelength_nm absente dans {csv_path}")

        if len(df) != bands:
            raise ValueError(
                f"Nombre de bandes incohérent pour {csv_path} : "
                f"{len(df)} lignes CSV vs {bands} bandes image"
            )

    # Fond ACE
    X_bg = img[:, valid].T
    bg_mean = np.nanmean(X_bg, axis=0).astype(np.float32)
    bg_cov = np.cov(X_bg, rowvar=False).astype(np.float32)
    bg_cov += np.eye(bg_cov.shape[0], dtype=np.float32) * 1e-3

    # Cartes globales finales
    ace_max = np.full((rows, cols), -np.inf, dtype=np.float32)
    ace_best_species = np.zeros((rows, cols), dtype=np.int16)

    sam_min = np.full((rows, cols), np.inf, dtype=np.float32)
    sam_best_species = np.zeros((rows, cols), dtype=np.int16)

    combo_max = np.full((rows, cols), -np.inf, dtype=np.float32)
    combo_best_species = np.zeros((rows, cols), dtype=np.int16)

    outputs = {}

    for class_name in class_columns:
        print(f"\n=== Classe : {class_name} ===")

        class_id = int(class_id_map[class_name])

        sam_maps = []
        ace_maps = []

        for csv_path, df in zip(reference_csv_paths, ref_dfs):

            if class_name not in df.columns:
                print(f"[WARN] Classe absente dans {csv_path.name} : {class_name}")
                continue

            ref_spectrum = df[class_name].to_numpy(dtype=np.float32)

            good = np.isfinite(ref_spectrum)

            if np.count_nonzero(good) < 10:
                print(f"[WARN] Spectre trop incomplet : {csv_path.name} / {class_name}")
                continue

            img_good = img[good, :, :]
            img_good_rcb = np.transpose(img_good, (1, 2, 0))
            ref_good = ref_spectrum[good].astype(np.float32)

            # SAM
            sam_map = spectral_angles(
                img_good_rcb,
                ref_good.reshape(1, -1)
            ).astype(np.float32)[:, :, 0]

            sam_map[~valid] = np.nan
            sam_maps.append(sam_map)

            # ACE
            mean_good = bg_mean[good]
            cov_good = bg_cov[np.ix_(good, good)]

            ace_map = ace_map_from_background_stats(
                img_brc=img_good,
                target_b=ref_good,
                bg_mean_b=mean_good,
                bg_cov_bb=cov_good,
                valid_mask_rc=valid,
            )

            ace_map[~valid] = np.nan
            ace_maps.append(ace_map)

        if len(sam_maps) == 0 or len(ace_maps) == 0:
            print(f"[WARN] Aucune référence utilisable pour {class_name}")
            continue

        # Multi-référence par classe
        sam_final = np.nanmin(np.stack(sam_maps, axis=0), axis=0).astype(np.float32)
        ace_final = np.nanmax(np.stack(ace_maps, axis=0), axis=0).astype(np.float32)

        combo_final = combine_sam_ace(
            sam_min=sam_final,
            ace_max=ace_final,
            sam_scale=sam_scale,
            mode="product",
        )

        # Mise à jour ACE max global
        better_ace = np.isfinite(ace_final) & (ace_final > ace_max)
        ace_max[better_ace] = ace_final[better_ace]
        ace_best_species[better_ace] = class_id

        # Mise à jour SAM min global
        better_sam = np.isfinite(sam_final) & (sam_final < sam_min)
        sam_min[better_sam] = sam_final[better_sam]
        sam_best_species[better_sam] = class_id

        # Mise à jour combo max global
        better_combo = np.isfinite(combo_final) & (combo_final > combo_max)
        combo_max[better_combo] = combo_final[better_combo]
        combo_best_species[better_combo] = class_id

        outputs[class_name] = {
            "class_id": class_id,
            "n_refs_used": len(sam_maps),
        }

    # Nettoyage inf
    ace_max[~np.isfinite(ace_max)] = np.nan
    sam_min[~np.isfinite(sam_min)] = np.nan
    combo_max[~np.isfinite(combo_max)] = np.nan

    # Sauvegarde
    ace_max_out = out_dir / "ace_max.tif"
    ace_species_out = out_dir / "ace_best_species.tif"

    sam_min_out = out_dir / "sam_min.tif"
    sam_species_out = out_dir / "sam_best_species.tif"

    combo_max_out = out_dir / "combo_max.tif"
    combo_species_out = out_dir / "combo_best_species.tif"

    with rasterio.open(ace_max_out, "w", **prof_float) as dst:
        dst.write(ace_max, 1)

    with rasterio.open(ace_species_out, "w", **prof_class) as dst:
        dst.write(ace_best_species, 1)

    with rasterio.open(sam_min_out, "w", **prof_float) as dst:
        dst.write(sam_min, 1)

    with rasterio.open(sam_species_out, "w", **prof_class) as dst:
        dst.write(sam_best_species, 1)

    with rasterio.open(combo_max_out, "w", **prof_float) as dst:
        dst.write(combo_max, 1)

    with rasterio.open(combo_species_out, "w", **prof_class) as dst:
        dst.write(combo_best_species, 1)

    # Sauvegarde correspondance id ↔ espèce
    mapping_df = pd.DataFrame([
        {"class_id": int(v), "class_name": k}
        for k, v in class_id_map.items()
    ])

    mapping_csv = out_dir / "species_id_mapping.csv"
    mapping_df.to_csv(mapping_csv, index=False)

    print("\nCartes finales sauvegardées :")
    print("ACE max        :", ace_max_out)
    print("ACE espèce     :", ace_species_out)
    print("SAM min        :", sam_min_out)
    print("SAM espèce     :", sam_species_out)
    print("Combo max      :", combo_max_out)
    print("Combo espèce   :", combo_species_out)
    print("Mapping espèces:", mapping_csv)

    return {
        "ace_max": str(ace_max_out),
        "ace_best_species": str(ace_species_out),
        "sam_min": str(sam_min_out),
        "sam_best_species": str(sam_species_out),
        "combo_max": str(combo_max_out),
        "combo_best_species": str(combo_species_out),
        "species_mapping": str(mapping_csv),
        "details": outputs,
    }

def compute_final_sam_ace_from_reference_csvs(
    img,
    profile,
    reference_csv_paths,
    out_dir,
    *,
    valid_pixels_mask=None,
    nodata=np.nan,
    compress="lzw",
    sam_scale=0.1,
):
    """
    Calcule, pour chaque classe végétale :
      - SAM_final = min(SAM_ref1, SAM_ref2, ...)
      - ACE_final = max(ACE_ref1, ACE_ref2, ...)
      - combo_sam_ace = combinaison SAM + ACE

    Parameters
    ----------
    img : np.ndarray
        Cube hyperspectral de forme (bands, rows, cols).
    profile : dict
        Profil rasterio de l'image de référence.
    reference_csv_paths : list[str | Path]
        Liste des CSV de spectres moyens de référence.
    out_dir : str | Path
        Dossier de sortie.
    valid_pixels_mask : np.ndarray | None
        Masque booléen optionnel, True = pixel valide.
    """

    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(f"img doit être de forme (bands, rows, cols), reçu {img.shape}")

    reference_csv_paths = [Path(p) for p in reference_csv_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands, rows, cols = img.shape

    valid = np.isfinite(img).all(axis=0)

    if valid_pixels_mask is not None:
        valid_pixels_mask = np.asarray(valid_pixels_mask).astype(bool)

        if valid_pixels_mask.shape != (rows, cols):
            raise ValueError(
                f"valid_pixels_mask shape={valid_pixels_mask.shape} incompatible "
                f"avec image shape={(rows, cols)}"
            )

        valid &= valid_pixels_mask


    print("Shape image :", img.shape)

    finite_per_band = np.isfinite(img).sum(axis=(1, 2))

    print("Pixels par bande :", rows * cols)
    print("Bandes totalement NaN :", np.where(finite_per_band == 0)[0] + 1)
    print("Nombre de bandes totalement NaN :", np.sum(finite_per_band == 0))

    print(
        "Pixels valides sur toutes les bandes :",
        np.isfinite(img).all(axis=0).sum()
    )

    print(
        "Pixels ayant au moins une bande valide :",
        np.isfinite(img).any(axis=0).sum()
    )

    if not np.any(valid):
        raise ValueError("Aucun pixel valide dans l'image hyperspectrale.")

    prof_out = profile.copy()
    prof_out.update(
        count=1,
        dtype="float32",
        nodata=nodata,
        compress=compress,
    )

    # Chargement CSV
    ref_dfs = [pd.read_csv(p) for p in reference_csv_paths]

    class_columns = [
        c for c in ref_dfs[0].columns
        if c != "wavelength_nm"
    ]

    if not class_columns:
        raise ValueError("Aucune classe trouvée dans les CSV de référence.")

    # Vérification cohérence bandes
    for csv_path, df in zip(reference_csv_paths, ref_dfs):

        if "wavelength_nm" not in df.columns:
            raise ValueError(f"Colonne 'wavelength_nm' absente dans {csv_path}")

        if len(df) != bands:
            raise ValueError(
                f"Nombre de bandes incohérent pour {csv_path} : "
                f"{len(df)} lignes dans le CSV, {bands} bandes dans l'image."
            )

    outputs = {}

    # Fond pour ACE
    X_bg = img[:, valid].T

    bg_mean = np.nanmean(X_bg, axis=0).astype(np.float32)
    bg_cov = np.cov(X_bg, rowvar=False).astype(np.float32)
    bg_cov += np.eye(bg_cov.shape[0], dtype=np.float32) * 1e-3

    for class_name in class_columns:

        print(f"\n=== Classe : {class_name} ===")

        sam_maps = []
        ace_maps = []

        for csv_path, df in zip(reference_csv_paths, ref_dfs):

            if class_name not in df.columns:
                print(f"[WARN] Classe absente dans {csv_path.name} : {class_name}")
                continue

            ref_spectrum = df[class_name].to_numpy(dtype=np.float32)

            if ref_spectrum.shape[0] != bands:
                print(f"[WARN] Spectre ignoré, longueur incorrecte : {csv_path.name}")
                continue

            good = np.isfinite(ref_spectrum)

            if np.count_nonzero(good) < 10:
                print(f"[WARN] Spectre trop incomplet : {csv_path.name} / {class_name}")
                continue

            img_good = img[good, :, :]
            img_good_rcb = np.transpose(img_good, (1, 2, 0))
            ref_good = ref_spectrum[good].astype(np.float32)

            # SAM
            sam_map = spectral_angles(
                img_good_rcb,
                ref_good.reshape(1, -1)
            ).astype(np.float32)[:, :, 0]

            sam_map[~valid] = np.nan
            sam_maps.append(sam_map)

            # ACE
            mean_good = bg_mean[good]
            cov_good = bg_cov[np.ix_(good, good)]

            ace_map = ace_map_from_background_stats(
                img_brc=img_good,
                target_b=ref_good,
                bg_mean_b=mean_good,
                bg_cov_bb=cov_good,
                valid_mask_rc=valid,
            )

            ace_map[~valid] = np.nan
            ace_maps.append(ace_map)

        if len(sam_maps) == 0 or len(ace_maps) == 0:
            print(f"[WARN] Aucune référence utilisable pour {class_name}")
            continue

        sam_final = np.nanmin(np.stack(sam_maps, axis=0), axis=0).astype(np.float32)
        ace_final = np.nanmax(np.stack(ace_maps, axis=0), axis=0).astype(np.float32)

        combo = combine_sam_ace(
            sam_min=sam_final,
            ace_max=ace_final,
            sam_scale=sam_scale,
            mode="product",
        )

        safe_name = safe_filename(class_name)

        sam_out = out_dir / f"sam_final_{safe_name}.tif"
        ace_out = out_dir / f"ace_final_{safe_name}.tif"
        combo_out = out_dir / f"combo_sam_ace_final_{safe_name}.tif"

        with rasterio.open(sam_out, "w", **prof_out) as dst:
            dst.write(sam_final, 1)

        with rasterio.open(ace_out, "w", **prof_out) as dst:
            dst.write(ace_final, 1)

        with rasterio.open(combo_out, "w", **prof_out) as dst:
            dst.write(combo, 1)

        outputs[class_name] = {
            "sam_final": str(sam_out),
            "ace_final": str(ace_out),
            "combo": str(combo_out),
            "n_refs_used": len(sam_maps),
        }

        print("SAM final   ->", sam_out)
        print("ACE final   ->", ace_out)
        print("Combo final ->", combo_out)
        print("Références utilisées :", len(sam_maps))

    return outputs