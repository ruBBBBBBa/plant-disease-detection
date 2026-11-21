"""Herramienta de inferencia (T07): una foto → máscara, regiones candidatas, predicción y motivo.

Encadena lo ya congelado en fases anteriores, sin ajustar nada nuevo:

1. normalización de T06 (lado mayor a 256 px);
2. segmentación: K-means de T03 (modo automático) o GrabCut inicializado con un rectángulo
   (modo ROI, práctica de segmentación);
3. «no concluyente» de T06 (D031/D035): máscara vacía o calidad fuera de los rangos de PlantVillage;
4. características ``v1`` (T04) y modelo ``exp2`` (T05) de la variante (B por defecto, D036).

Como B es un SVM **lineal** sobre características estandarizadas, su puntuación de decisión se
descompone exactamente en una suma de contribuciones por característica (``linear_contributions``).
La puntuación **no es una probabilidad** (regla 8 de ``AGENTS.md``).
"""

from __future__ import annotations

import json

import cv2
import joblib
import numpy as np

from src import external as ex
from src import features as ft
from src import segmentation as seg
from src import training as tr

SCORE_NOTE = "puntuación de decisión del SVM, no una probabilidad"
FIXED_WARNINGS = [
    "Las regiones candidatas son zonas de color anómalo, no «dónde ve la enfermedad» el modelo.",
    "El sistema solo es fiable con una hoja sola sobre fondo uniforme (D035).",
]
LABELS = {0: "healthy", 1: "diseased"}

# --------------------------------------------------------------------------- ROI


def parse_roi(text: str) -> tuple[int, int, int, int]:
    """Convierte ``"x,y,w,h"`` en una tupla de enteros; ancho y alto deben ser positivos."""
    x, y, w, h = (int(v) for v in text.split(","))
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI con ancho o alto no positivo: {text}")
    return x, y, w, h


def expand_roi(
    roi: tuple[int, int, int, int], margin: float, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Caja de recorte ``(x0, y0, x1, y1)``: la ROI más ``margin`` de su ancho/alto por cada lado.

    Se recorta a los bordes de la imagen (no se rellena con píxeles inventados).
    """
    x, y, w, h = roi
    dx, dy = round(margin * w), round(margin * h)
    height, width = shape[:2]
    return max(0, x - dx), max(0, y - dy), min(width, x + w + dx), min(height, y + h + dy)


def crop_roi(
    image_bgr: np.ndarray, roi: tuple[int, int, int, int], margin: float, long_side: int = 256
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Recorta la ROI con margen, normaliza el recorte (lado mayor ``long_side``) y lleva la ROI.

    Devuelve ``(recorte normalizado, caja de recorte en la original, ROI en el recorte)``.
    La ROI interior se lleva a coordenadas del recorte restando el origen de la caja y
    multiplicando por la escala de la normalización.
    """
    box = expand_roi(roi, margin, image_bgr.shape)
    x0, y0, x1, y1 = box
    crop = ex.normalize_size(image_bgr[y0:y1, x0:x1], long_side)
    sx, sy = crop.shape[1] / (x1 - x0), crop.shape[0] / (y1 - y0)
    x, y, w, h = roi
    inner = (round((x - x0) * sx), round((y - y0) * sy), round(w * sx), round(h * sy))
    return crop, box, inner


def segment_roi(crop_bgr: np.ndarray, inner: tuple[int, int, int, int], seg_config: dict):
    """GrabCut inicializado con el rectángulo ``inner`` y después el postprocesado de T03.

    Técnica: GrabCut (práctica de segmentación): fuera del rectángulo es «fondo seguro», dentro
    «probable hoja»; los modelos de color (mezclas de gaussianas) y el corte de grafo deciden.
    Se deja al menos 1 px de fondo seguro en cada lado: si la ROI llega al borde del recorte,
    GrabCut no tendría píxeles con los que modelar el fondo.
    """
    height, width = crop_bgr.shape[:2]
    x, y, w, h = inner
    x0, y0 = max(1, x), max(1, y)
    x1, y1 = min(width - 1, x + w), min(height - 1, y + h)
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    iterations = seg_config["methods"]["grabcut"]["iterations"]
    cv2.setRNGSeed(0)
    cv2.grabCut(crop_bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    out = seg.to_mask((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    if seg_config["default_postprocess"]:
        out = seg.postprocess(out, seg_config["postprocess_kernel"])
    return out


def mask_to_original(
    mask: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, ...]
) -> np.ndarray:
    """Devuelve una máscara del recorte a coordenadas de la imagen original (vecino más próximo)."""
    x0, y0, x1, y1 = box
    out = np.zeros(shape[:2], np.uint8)
    out[y0:y1, x0:x1] = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


# --------------------------------------------------------------------------- configuración


def load_configs() -> dict:
    """Configuraciones congeladas: segmentación (T03), características (T04), T06 y T07."""
    root = tr.REPO_ROOT / "configs"
    names = {"seg": "segmentation", "feat": "features", "external": "external", "own": "own"}
    return {k: json.loads((root / f"{v}.json").read_text()) for k, v in names.items()}


def load_pipeline(variant_cfg: dict, exp: str = "exp2"):
    """Carga el pipeline entrenado en T05 (``models/<exp>/<grupo>__<modelo>.joblib``)."""
    path = tr.REPO_ROOT / "models" / exp / f"{variant_cfg['group']}__{variant_cfg['model']}.joblib"
    if not path.is_file():
        raise SystemExit(
            f"Error: falta el modelo {path} (se genera con scripts/train_classifiers.py)"
        )
    return joblib.load(path)


# --------------------------------------------------------------------------- análisis


def segment_and_extract(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    seg_config: dict,
    feat_config: dict,
    long_side: int,
    margin: float,
) -> dict:
    """Segmenta (automático o ROI) y extrae las características ``v1``.

    Devuelve la imagen analizada (normalizada o recorte normalizado), la máscara de la hoja,
    la de regiones candidatas, las características y, en modo ROI, la caja y la ROI transformada.
    Esta parte no depende de la variante: se calcula una vez y se predice con varias.
    """
    if roi is None:
        image = ex.normalize_size(image_bgr, long_side)
        mask, box, inner = ft.leaf_mask(image, seg_config), None, None
    else:
        image, box, inner = crop_roi(image_bgr, roi, margin, long_side)
        mask = segment_roi(image, inner, seg_config)
    feats = ft.extract_features(image, mask, feat_config)
    spots = np.zeros_like(mask)
    if not feats["mask_empty"]:
        spots = ft.spot_mask(
            image, ft.prepare_mask(image, mask, feat_config["mask_variant"]), feat_config
        )
    return {
        "mode": "auto" if roi is None else "roi",
        "image": image,
        "mask": mask,
        "spots": spots,
        "features": feats,
        "roi": roi,
        "crop_box": box,
        "roi_in_crop": inner,
    }


def inconclusive_reasons(feats: dict, ranges: dict) -> list[str]:
    """Motivos de «no concluyente» (D031): máscara vacía o qué calidad se sale y de qué rango."""
    if feats["mask_empty"]:
        return ["máscara vacía"]
    return [
        f"{c} = {feats[c]:.3f} fuera de [{ranges[c][0]:.3f}, {ranges[c][1]:.3f}]"
        for c in ex.QUALITY_COLUMNS
        if not (ranges[c][0] <= feats[c] <= ranges[c][1])
    ]


def linear_contributions(pipeline, x: np.ndarray, names: list[str]) -> tuple[list[dict], float]:
    """Contribución de cada característica a la puntuación de un modelo lineal estandarizado.

    Con z = (x − μ) / σ (``StandardScaler`` ajustado en train) la puntuación es f = w·z + b, así
    que cᵢ = wᵢ·zᵢ y Σ cᵢ = f − b. Positiva empuja hacia ``diseased``; negativa hacia ``healthy``.
    Devuelve las contribuciones ordenadas por |cᵢ| y el término independiente b.
    """
    z = pipeline["scaler"].transform(x.reshape(1, -1))[0]
    w = pipeline["clf"].coef_[0]
    rows = [
        {"feature": n, "value": float(v), "z": float(zi), "contribution": float(wi * zi)}
        for n, v, zi, wi in zip(names, x, z, w, strict=True)
    ]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows, float(pipeline["clf"].intercept_[0])


def predict_variant(analysis: dict, pipeline, group: str, ranges: dict, top: int = 5) -> dict:
    """Predicción de una variante sobre un análisis; si es no concluyente, sin predicción."""
    feats = analysis["features"]
    reasons = inconclusive_reasons(feats, ranges)
    names = tr.FEATURE_GROUPS[group]
    x = np.array([feats[n] for n in names], dtype=np.float64)
    if not reasons and not np.isfinite(x).all():
        reasons = ["características no finitas"]
    out = {
        "inconclusive": bool(reasons),
        "reasons": reasons,
        "prediction": None,
        "score": None,
        "score_note": SCORE_NOTE,
        "intercept": None,
        "top_contributions": [],
    }
    if reasons:
        return out
    score = float(pipeline.decision_function(x.reshape(1, -1))[0])
    contributions, intercept = linear_contributions(pipeline, x, names)
    out |= {
        "prediction": LABELS[int(pipeline.predict(x.reshape(1, -1))[0])],
        "score": score,
        "intercept": intercept,
        "top_contributions": contributions[:top],
    }
    return out


def result_json(analysis: dict, prediction: dict, variant: str, variant_cfg: dict) -> dict:
    """Diccionario serializable con todo lo que muestra la figura."""
    feats = analysis["features"]
    return {
        "mode": analysis["mode"],
        "roi": list(analysis["roi"]) if analysis["roi"] else None,
        "crop_box": list(analysis["crop_box"]) if analysis["crop_box"] else None,
        "variant": variant,
        "group": variant_cfg["group"],
        "model": variant_cfg["model"],
        **prediction,
        "quality": {c: feats.get(c) for c in ["mask_empty", *ex.QUALITY_COLUMNS]},
        "spot_area_frac": feats.get("spot_area_frac"),
        "warnings": FIXED_WARNINGS,
    }


# --------------------------------------------------------------------------- figura


def figure_lines(result: dict) -> list[str]:
    """Texto del panel de resultados de la figura (también lo usan los tests)."""
    lines = [
        f"Modo: {result['mode']}   Variante: {result['variant']} "
        f"({result['group']} + {result['model']})"
    ]
    if result["inconclusive"]:
        lines += ["NO CONCLUYENTE: sin predicción", *[f"  motivo: {r}" for r in result["reasons"]]]
    else:
        lines += [
            f"Predicción: {result['prediction']}",
            f"Puntuación: {result['score']:+.3f} ({SCORE_NOTE})",
            f"Término independiente b = {result['intercept']:+.3f}",
            "5 contribuciones mayores (coef. × valor estandarizado; + → diseased):",
            *[
                f"  {c['feature']}: {c['contribution']:+.3f} "
                f"(valor {c['value']:.3g}, z {c['z']:+.2f})"
                for c in result["top_contributions"]
            ],
        ]
    return [*lines, "", *result["warnings"]]


def overlay(image_bgr: np.ndarray, mask: np.ndarray, spots: np.ndarray) -> np.ndarray:
    """Contorno de la hoja en verde y regiones candidatas en magenta, sobre la imagen (RGB)."""
    out = image_bgr.copy()
    out[spots > 0] = (0.5 * out[spots > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def save_figure(original_bgr: np.ndarray, analysis: dict, result: dict, path) -> None:
    """Figura: original (con la ROI), máscara, regiones candidatas y panel de texto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = original_bgr.copy()
    if analysis["roi"]:
        x, y, w, h = analysis["roi"]
        thick = max(2, round(max(shown.shape[:2]) / 300))
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 255), thick)
    fig = plt.figure(figsize=(14, 6))
    panels = [
        (
            cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
            "Original" + (" + ROI" if analysis["roi"] else ""),
        ),
        (analysis["mask"], "Máscara de la hoja"),
        (overlay(analysis["image"], analysis["mask"], analysis["spots"]), "Regiones candidatas"),
    ]
    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.axis("off")
    ax.text(0, 1, "\n".join(figure_lines(result)), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- interacción


def display_scale(shape: tuple[int, ...], max_side: int = 1000) -> float:
    """Factor para que la foto quepa en pantalla (nunca amplía)."""
    return min(1.0, max_side / max(shape[:2]))


def to_original(rois: list, scale: float) -> list[tuple[int, int, int, int]]:
    """Rectángulos dibujados sobre la foto reducida → coordenadas de la imagen original."""
    return [tuple(round(v / scale) for v in r) for r in rois if r[2] > 0 and r[3] > 0]


def select_rois(image_bgr: np.ndarray, window: str, multiple: bool) -> list:
    """Dibuja una o varias ROI (``cv2.selectROI(s)``) en la foto reducida; las da en la original."""
    scale = display_scale(image_bgr.shape)
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if multiple:
        rois = cv2.selectROIs(window, small, showCrosshair=False)
    else:
        rois = [cv2.selectROI(window, small, showCrosshair=False)]
    cv2.destroyWindow(window)
    return to_original([tuple(int(v) for v in r) for r in rois], scale)
