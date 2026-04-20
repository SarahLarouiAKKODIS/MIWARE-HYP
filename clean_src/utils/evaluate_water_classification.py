from __future__ import annotations

from typing import Iterable
import numpy as np
import rasterio
from fix_georef import fix_raster_georef

def print_evaluation_results(results: dict):
    print("\n=== Résultats de l'évaluation (masque eau) ===\n")

    print("---- Comptage des pixels ----")
    print(f"{'Pixels valides':30s}: {results['total_valid_pixels']}")
    print(f"{'Pixels eau (GT)':30s}: {results['gt_water_pixels']}")
    print(f"{'Pixels eau (prédiction)':30s}: {results['pred_water_pixels']}")

    print("\n---- Matrice de confusion ----")
    print(f"{'Vrais positifs (TP)':30s}: {results['true_positives']}")
    print(f"{'Faux positifs (FP)':30s}: {results['false_positives']}")
    print(f"{'Faux négatifs (FN)':30s}: {results['false_negatives']}")
    print(f"{'Vrais négatifs (TN)':30s}: {results['true_negatives']}")

    print("\n---- Métriques ----")
    print(f"{'Accuracy':30s}: {results['accuracy']:.4f}")
    print(f"{'Precision':30s}: {results['precision']:.4f}")
    print(f"{'Recall':30s}: {results['recall']:.4f}")
    print(f"{'F1-score':30s}: {results['f1_score']:.4f}")
    print(f"{'IoU (eau)':30s}: {results['iou_water']:.4f}")
    print(f"{'Specificity':30s}: {results['specificity']:.4f}")
    print(f"{'False positive rate':30s}: {results['false_positive_rate']:.4f}")

    # -----------------------------
    # Interprétation rapide
    # -----------------------------
    print("\n---- Interprétation rapide ----")

    precision = results["precision"]
    recall = results["recall"]
    fpr = results["false_positive_rate"]

    if precision > 0.9:
        print("✔ Très peu de faux positifs (bonne détection de l’eau).")
    elif precision < 0.7:
        print("⚠ Beaucoup de faux positifs (sur-détection de l’eau).")

    if recall > 0.9:
        print("✔ La majorité de l’eau est détectée.")
    elif recall < 0.7:
        print("⚠ Une partie importante de l’eau est manquée.")

    if fpr > 0.1:
        print("⚠ Taux de faux positifs élevé → risque de masquer trop de zones.")

    print("\n===========================================\n")

def evaluate_water_classification(
    ground_truth_tif: str,
    prediction_tif: str,
    *,
    gt_water_label: int = 2,
    pred_water_label: int = 255,
    pred_invalid_label: int = -1,
    gt_valid_labels: Iterable[int] | None = None,
    check_alignment: bool = True,
) -> dict:
    """
    Évalue une classification de l'eau entre une vérité terrain et une prédiction,
    en excluant les pixels invalides directement à partir du raster de prédiction.

    Convention attendue pour prediction_tif (issu de compute_mndwi_and_water_mask)
    ------------------------------------------------------------------------------
    - 255 : eau
    - 0   : non-eau
    - -1  : pixel invalide / à exclure

    Parameters
    ----------
    ground_truth_tif : str
        Chemin vers le raster de vérité terrain.
    prediction_tif : str
        Chemin vers le raster de prédiction (WATER_MASK).
    gt_water_label : int, default 2
        Label de la classe eau dans la vérité terrain.
    pred_water_label : int, default 255
        Label de la classe eau dans la prédiction.
    pred_invalid_label : int, default -1
        Label des pixels invalides dans la prédiction.
    gt_valid_labels : Iterable[int] | None, default None
        Si fourni, seuls ces labels de vérité terrain seront évalués.
        Exemple : [1, 2] si 1=non-eau et 2=eau.
        Très utile si la GT contient des classes "non classé", "incertain", etc.
    check_alignment : bool, default True
        Si True, vérifie shape, CRS et transform avant comparaison.

    Returns
    -------
    dict
        Dictionnaire contenant :
        - matrice de confusion
        - métriques classiques
        - IoU eau
        - comptages de pixels
    """
    with rasterio.open(ground_truth_tif) as gt_src, rasterio.open(prediction_tif) as pred_src:
        gt = gt_src.read(1)
        pred = pred_src.read(1)

        # -----------------------------
        # Vérifications géométriques
        # -----------------------------
        if gt.shape != pred.shape:
            raise ValueError(
                f"Les dimensions ne correspondent pas : "
                f"GT {gt.shape}, prédiction {pred.shape}"
            )

        if check_alignment:
            if gt_src.crs != pred_src.crs:
                raise ValueError(
                    f"CRS différents : GT={gt_src.crs}, prédiction={pred_src.crs}"
                )

            if gt_src.transform != pred_src.transform:
                raise ValueError(
                    "Transforms différents entre GT et prédiction. "
                    "Les rasters ne sont pas parfaitement alignés."
                )

        # -----------------------------
        # Construction du masque valide
        # -----------------------------
        valid_mask = np.ones(gt.shape, dtype=bool)

        # Exclure nodata GT
        if gt_src.nodata is not None:
            valid_mask &= (gt != gt_src.nodata)

        # Exclure nodata prédiction
        if pred_src.nodata is not None:
            valid_mask &= (pred != pred_src.nodata)

        # Exclure NaN / inf éventuels
        if np.issubdtype(gt.dtype, np.floating):
            valid_mask &= np.isfinite(gt)
        if np.issubdtype(pred.dtype, np.floating):
            valid_mask &= np.isfinite(pred)

        # Exclure pixels invalides de la prédiction
        valid_mask &= (pred != pred_invalid_label)

        # Exclure labels GT non désirés si demandé
        if gt_valid_labels is not None:
            gt_valid_labels = set(gt_valid_labels)
            valid_mask &= np.isin(gt, list(gt_valid_labels))

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            raise ValueError("Aucun pixel valide à comparer après masquage.")

        # -----------------------------
        # Binarisation eau / non-eau
        # -----------------------------
        gt_water = (gt == gt_water_label)
        pred_water = (pred == pred_water_label)

        gt_water_valid = gt_water[valid_mask]
        pred_water_valid = pred_water[valid_mask]

        # -----------------------------
        # Matrice de confusion
        # -----------------------------
        tp = int(np.sum(gt_water_valid & pred_water_valid))
        fp = int(np.sum((~gt_water_valid) & pred_water_valid))
        fn = int(np.sum(gt_water_valid & (~pred_water_valid)))
        tn = int(np.sum((~gt_water_valid) & (~pred_water_valid)))

        total = tp + fp + fn + tn

        # -----------------------------
        # Métriques
        # -----------------------------
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        iou_water = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        gt_water_pixels = int(np.sum(gt_water_valid))
        pred_water_pixels = int(np.sum(pred_water_valid))

        return {
            "total_valid_pixels": total,
            "gt_water_pixels": gt_water_pixels,
            "pred_water_pixels": pred_water_pixels,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "false_positive_rate": float(false_positive_rate),
            "iou_water": float(iou_water),
            "params": {
                "gt_water_label": gt_water_label,
                "pred_water_label": pred_water_label,
                "pred_invalid_label": pred_invalid_label,
                "gt_valid_labels": list(gt_valid_labels) if gt_valid_labels is not None else None,
                "check_alignment": check_alignment,
            },
        }

if __name__ == "__main__":

    # ground_truth_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/SALSIGNE/dims_op_oc_oc-en_702726665_1/ENMAP.HSI.L2A/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z/ENMAP01-____L2A-DT0000165944_20251129T112021Z_002_V010505_20251130T042131Z-QL_QUALITY_CLASSES.TIF"
    # prediction_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/SALSIGNE/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"
   
    
    # ground_truth_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/CMA/ENMAP01-____L2A-DT0000183870_20260317T112025Z_002_V010506_20260318T023754Z/ENMAP01-____L2A-DT0000183870_20260317T112025Z_002_V010506_20260318T023754Z-QL_QUALITY_CLASSES.TIF"
    # prediction_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/CMA/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"
    
    ground_truth_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Data/Abbaretz/ENMAP.HSI.L2A/Abbaretz_2/ENMAP01-____L2A-DT0000166114_20251130T114431Z_002_V010505_20251201T013553Z/ENMAP01-____L2A-DT0000166114_20251130T114431Z_002_V010505_20251201T013553Z-QL_QUALITY_CLASSES.TIF"
    prediction_tif = "/home/sarah.laroui/Bureau/MIWARE-HYP/Python_code/Results/ABBARETZ/Water_indice_outputs/enmap_salsigne_WATER_MASK.tiff"

    fix_raster_georef(
        raster_to_fix=prediction_tif,
        reference_raster=ground_truth_tif
    )

    results = evaluate_water_classification(
        ground_truth_tif=ground_truth_tif,
        prediction_tif=prediction_tif,
        gt_valid_labels=[1, 2]
    )

    print_evaluation_results(results)