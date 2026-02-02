import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import tifffile as tiff


def save_colored_quality_mask(
    mask_path: str,
    out_png: str = "quality_mask_colored.png",
    out_tif: str | None = "quality_mask_colored.tif",
    band_index: int = 1,
    nodata_values: set[int] | None = None,
    title: str = "Quality layer (classes)",
    show: bool = True,
) -> dict[int, tuple[int, int, int]]:
    """
    Ouvre un masque de classes (quality layer), crée une visualisation très lisible
    (couleurs discrètes + légende), sauvegarde en PNG et optionnellement en TIFF RGB.

    Retourne un dict {classe: (R,G,B)}.
    """
    if nodata_values is None:
        nodata_values = set()

    # Labels (EN) pour les classes EnMAP
    class_meanings_en: dict[int, str] = {
        0: "Unclassified pixels",
        1: "Land",
        2: "Water",
        3: "Out of scene",
    }

    # 1) lecture
    with rasterio.open(mask_path) as src:
        q = src.read(band_index)

        # nodata depuis le fichier si dispo
        if src.nodata is not None:
            try:
                nodata_values.add(int(src.nodata))
            except Exception:
                pass

    q = np.asarray(q)
    classes = np.unique(q)

    # 2) retirer nodata si demandé (on le mettra en noir)
    classes_no_nodata = [c for c in classes if int(c) not in nodata_values]

    if len(classes_no_nodata) == 0:
        raise ValueError("Aucune classe trouvée (hors NoData). Vérifie le band_index et nodata_values.")

    # 3) palette : très contrastée et “visuelle”
    # On pioche dans tab20, puis on boucle si > 20 classes.
    base = plt.get_cmap("tab20").colors  # tuples RGB [0..1]
    rgb_colors = []
    for i in range(len(classes_no_nodata)):
        c = base[i % len(base)]
        rgb_colors.append((int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))

    # NoData en noir (si présent)
    class_to_rgb: dict[int, tuple[int, int, int]] = {}
    for cls, rgb in zip(classes_no_nodata, rgb_colors):
        class_to_rgb[int(cls)] = rgb
    for nd in nodata_values:
        class_to_rgb[int(nd)] = (0, 0, 0)

    # 4) construire une image RGB
    h, w = q.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in class_to_rgb.items():
        rgb_img[q == cls] = rgb

    # 5) préparer affichage labelisé (cmap discret + norm)
    # On garde l’ordre des classes (hors nodata) pour une légende propre
    ordered_classes = classes_no_nodata[:]  # liste
    ordered_colors01 = [(r / 255, g / 255, b / 255) for (r, g, b) in rgb_colors]

    cmap = ListedColormap(ordered_colors01, name="quality_classes")
    # bornes entre classes (discret)
    bounds = np.arange(len(ordered_classes) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    # On convertit q -> indices 0..N-1 pour l'affichage discret
    class_to_idx = {int(c): i for i, c in enumerate(ordered_classes)}
    q_idx = np.full(q.shape, fill_value=-1, dtype=np.int32)
    for c, i in class_to_idx.items():
        q_idx[q == c] = i

    # 6) sauver PNG (avec légende)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Pixels nodata (idx=-1) affichés en noir via un overlay RGB simple
    ax.imshow(rgb_img)
    ax.set_title(title)
    ax.axis("off")

    # Légende (patches) – très lisible
    import matplotlib.patches as mpatches

    patches = []
    for c in ordered_classes:
        r, g, b = class_to_rgb[int(c)]
        meaning = class_meanings_en.get(int(c), "Unknown")
        patches.append(
            mpatches.Patch(
                color=(r / 255, g / 255, b / 255),
                label=f"Class {int(c)} = {meaning}",
            )
        )
    if nodata_values:
        patches.append(mpatches.Patch(color=(0, 0, 0), label="NoData"))

    ax.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Classes",
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 7) sauver TIFF RGB (optionnel, sans légende)
    if out_tif:
        tiff.imwrite(out_tif, rgb_img, photometric="rgb")

    return class_to_rgb


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Remplace par ton fichier quality layer
    mask_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLASSES.TIF"
    out_path = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/"
    # Si tu sais qu'une valeur représente du NoData (ex: 255), mets-la ici
    nodata_vals = {255}

    mapping = save_colored_quality_mask(
        mask_path=mask_path,
        out_png=out_path + "quality_layer_colored.png",
        out_tif=out_path + "quality_layer_colored.tif",
        band_index=1,
        nodata_values=nodata_vals,
        title="EnMAP L2A Quality Layer (Classes)",
        show=True
    )

    print("Mapping classe -> couleur (RGB):")
    for k in sorted(mapping.keys()):
        print(k, "->", mapping[k])
