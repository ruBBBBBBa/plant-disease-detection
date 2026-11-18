"""Test externo en PlantDoc (T06): normalización, «no concluyente», métricas e hipótesis.

Todo lo que hay aquí se fija **antes** de procesar PlantDoc (regla de oro de T06). Los umbrales
salen solo de PlantVillage (``configs/external.json``); PlantDoc no ajusta nada.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import features as ft
from src import training as tr

QUALITY_COLUMNS = ["leaf_area_frac", "border_touch_frac", "mask_solidity"]
READINGS = ("all", "conclusive")

# --------------------------------------------------------------------------- imagen


def read_bgr(path: Path) -> np.ndarray | None:
    """Lee una imagen como BGR de 3 canales (gris y 4 canales se convierten); None si falla.

    ``IMREAD_COLOR`` descarta el canal alfa y replica el gris, igual que en PlantVillage.
    """
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def normalize_size(image_bgr: np.ndarray, long_side: int = 256) -> np.ndarray:
    """Reescala para que el lado mayor mida ``long_side`` px, conservando la proporción."""
    h, w = image_bgr.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- no concluyente


def quality_ranges(table: pd.DataFrame, low: float = 0.5, high: float = 99.5) -> dict:
    """Rango [p_low, p_high] de cada característica de calidad (solo máscaras no vacías)."""
    valid = table[~table["mask_empty"].astype(bool)]
    return {
        c: [float(np.percentile(valid[c], low)), float(np.percentile(valid[c], high))]
        for c in QUALITY_COLUMNS
    }


def is_inconclusive_range(row: pd.Series | dict, ranges: dict) -> bool:
    """No concluyente si la máscara está vacía o alguna calidad cae fuera de su rango."""
    if row["mask_empty"]:
        return True
    return any(not (ranges[c][0] <= row[c] <= ranges[c][1]) for c in QUALITY_COLUMNS)


def inconclusive_mask(table: pd.DataFrame, ranges: dict) -> np.ndarray:
    """Vector booleano de «no concluyente» para cada fila de una tabla de características."""
    return np.array([is_inconclusive_range(r, ranges) for r in table.to_dict("records")])


def extract_image(path: Path, seg_config: dict, config: dict, sigma: float, long_side: int):
    """Normalización → segmentación T03 → características T04, para una imagen.

    Devuelve ``(características, características con textura suavizada)``. Si la imagen no se
    puede leer, se devuelve un vector vacío con ``mask_empty=True`` y ``read_error=True``.
    """
    image = read_bgr(path)
    if image is None:
        empty = {"mask_empty": True} | ft._zero_features(config)
        return {**empty, "read_error": True}, {**empty, "read_error": True}
    image = normalize_size(image, long_side)
    mask = ft.leaf_mask(image, seg_config)
    feats = ft.extract_features(image, mask, config) | {"read_error": False}
    blurred = {**feats}
    if not feats["mask_empty"]:
        # Mismo suavizado que en T05 (generate_v1_blur): solo cambia la textura.
        prepared = ft.prepare_mask(image, mask, config["mask_variant"])
        gray = cv2.cvtColor(cv2.GaussianBlur(image, (0, 0), sigma), cv2.COLOR_BGR2GRAY)
        tex = config["texture"]
        blurred |= ft.glcm_features(gray, prepared, tex["glcm_levels"], tex["glcm_distance"])
        blurred |= ft.lbp_features(gray, prepared, tex["lbp_points"], tex["lbp_radius"])
    return feats, blurred


# --------------------------------------------------------------------------- métricas


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Media de los recalls de sanas y enfermas (NaN si falta una clase)."""
    rec = [
        float((y_pred[y_true == c] == c).mean()) if (y_true == c).any() else np.nan for c in (0, 1)
    ]
    return float(np.mean(rec))


def _ba_replicas(y_true: np.ndarray, y_pred: np.ndarray, index: np.ndarray) -> np.ndarray:
    """BA de cada réplica bootstrap (``index``: réplicas × n), vectorizado."""
    yt, ok = y_true[index], (y_pred == y_true)[index]
    out = []
    for c in (0, 1):
        n_c = (yt == c).sum(axis=1)
        out.append(np.where(n_c > 0, (ok & (yt == c)).sum(axis=1) / np.maximum(n_c, 1), np.nan))
    return (out[0] + out[1]) / 2


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap emparejado de BA(a) − BA(b): mismos índices de imagen en las dos variantes."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    diff = _ba_replicas(y_true, pred_a, index) - _ba_replicas(y_true, pred_b, index)
    diff = diff[np.isfinite(diff)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "diff": balanced_accuracy(y_true, pred_a) - balanced_accuracy(y_true, pred_b),
        "ci95": [float(lo), float(hi)],
    }


def bootstrap_ci(values_fn, n: int, n_boot: int = 2000, seed: int = 42) -> list[float]:
    """IC 95 % por percentiles de ``values_fn(índices)`` remuestreando imágenes."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        try:
            v = values_fn(rng.integers(0, n, n))
        except ValueError:  # réplica sin una de las dos clases
            continue
        if np.isfinite(v):
            vals.append(v)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [np.nan, np.nan]


def reading_metrics(
    pred: pd.DataFrame, reading: str, n_boot: int = 2000, seed: int = 42, with_ci: bool = True
) -> dict:
    """Métricas de una variante en una lectura, siempre con su cobertura.

    ``pred``: columnas ``y_true, y_pred, score, inconclusive``. Lectura ``all``: las no
    concluyentes cuentan como error (``y_pred`` ya lo trae así). Lectura ``conclusive``: solo
    las analizadas.
    """
    if reading not in READINGS:
        raise ValueError(f"lectura desconocida: {reading}")
    coverage = float(1 - pred["inconclusive"].mean())
    rows = pred if reading == "all" else pred[~pred["inconclusive"]]
    y, p, s = (rows[c].to_numpy() for c in ("y_true", "y_pred", "score"))
    try:
        metrics = tr.compute_metrics(y, p, s)
    except ValueError:  # una sola clase: sin ROC-AUC
        metrics = tr.compute_metrics(y, p)
    metrics |= {"reading": reading, "n": int(len(rows)), "coverage": coverage}
    if with_ci:
        auc = lambda i: roc_auc_score(y[i], s[i])  # noqa: E731
        ba = lambda i: balanced_accuracy(y[i], p[i])  # noqa: E731
        rec = lambda c: lambda i: float((p[i][y[i] == c] == c).mean())  # noqa: E731
        metrics["ci95"] = {
            "balanced_accuracy": bootstrap_ci(ba, len(y), n_boot, seed),
            "recall_healthy": bootstrap_ci(rec(0), len(y), n_boot, seed),
            "recall_diseased": bootstrap_ci(rec(1), len(y), n_boot, seed),
            "roc_auc": bootstrap_ci(auc, len(y), n_boot, seed),
        }
    return metrics


def recall_by_disease(pred: pd.DataFrame, reading: str) -> pd.DataFrame:
    """Recall por ``disease_name`` (aciertos / n; en sanas, las predichas como sanas)."""
    rows = pred if reading == "all" else pred[~pred["inconclusive"]]
    ok = rows["y_pred"] == rows["y_true"]
    return ok.groupby(rows["disease_name"]).agg(recall="mean", n="size").reset_index()


def special_cases(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Casos especiales de T06 (filas de predicción), para informarlos aparte."""
    name = pred["disease_name"]
    return {
        "mosaic_virus": pred[name == "mosaic_virus"],
        "yellow_virus": pred[name == "yellow_virus"],
        "spider_mites": pred[name == "spider_mites"],
        "tomato_V8.jpg": pred[pred["image_path"].str.endswith("tomato_V8.jpg")],
    }


# --------------------------------------------------------------------------- hipótesis


def val_balanced_accuracy(val_predictions: pd.DataFrame) -> float:
    """BA de val de PlantVillage a partir de ``val__<grupo>__<modelo>.csv.gz``."""
    return balanced_accuracy(
        val_predictions["y_true"].to_numpy(), val_predictions["y_pred"].to_numpy()
    )


def verdict(ci95: list[float]) -> str:
    """Apoyada si el IC excluye 0 con signo positivo; contraria si lo excluye en negativo."""
    if ci95[0] > 0:
        return "apoyada"
    if ci95[1] < 0:
        return "no apoyada (efecto contrario)"
    return "no concluyente"


def evaluate_hypothesis(worse: dict, better: dict, n_boot: int = 10000, seed: int = 42) -> dict:
    """Contraste de «``better`` cae menos que ``worse``»: Δ_worse − Δ_better > 0.

    Cada variante es ``{"val_ba", "y_true", "y_pred"}`` (``y_pred`` en la lectura «todas»).
    Δ = BA(val) − BA(PlantDoc). La BA de val se toma como fija: el IC solo recoge la
    incertidumbre de PlantDoc (limitación que se anota en el informe).
    """
    boot = paired_bootstrap_diff(worse["y_true"], worse["y_pred"], better["y_pred"], n_boot, seed)
    shift = worse["val_ba"] - better["val_ba"]  # Δ_w − Δ_b = shift − (BA_pd_w − BA_pd_b)
    ci = [shift - boot["ci95"][1], shift - boot["ci95"][0]]
    return {"diff_delta": shift - boot["diff"], "ci95": ci, "verdict": verdict(ci)}


def delta(
    val_ba: float, y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 10000, seed: int = 42
):
    """Δ = BA(val) − BA(PlantDoc) de una variante, con IC 95 % (val fijo)."""
    index = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    reps = _ba_replicas(y_true, y_pred, index)
    reps = reps[np.isfinite(reps)]
    ba = balanced_accuracy(y_true, y_pred)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"ba_plantdoc": ba, "delta": val_ba - ba, "ci95": [val_ba - hi, val_ba - lo]}


def reference_flag_rate(table: pd.DataFrame, ranges: dict) -> dict:
    """% de imágenes de PlantVillage (train + val) que marcaría la regla, por motivo."""
    flagged = inconclusive_mask(table, ranges)
    out = {"any": float(flagged.mean()) * 100, "n": int(len(table))}
    out["mask_empty"] = float(table["mask_empty"].astype(bool).mean()) * 100
    for c in QUALITY_COLUMNS:
        out[c] = float(((table[c] < ranges[c][0]) | (table[c] > ranges[c][1])).mean()) * 100
    return out
