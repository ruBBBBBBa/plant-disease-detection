"""Entrenamiento, CV agrupada y evaluación de clasificadores clásicos (T05).

- CV: StratifiedGroupKFold por leaf_group_id (sin fuga entre hojas).
- Escalado: siempre dentro de Pipeline (StandardScaler + modelo).
- Hiperparámetros: rejillas pequeñas y justificadas (Tema 6).
- Métricas: confusion matrix, precision/recall/F1, accuracy, balanced accuracy,
  ROC-AUC, recall diseased, IC 95% bootstrap agrupado por hoja.
- control_nitidez no es candidato a mejor modelo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- feature groups

FEATURE_GROUPS = {
    "color": [
        "h_circ_mean_deg",
        "h_circ_var",
        "s_mean",
        "s_std",
        "v_mean",
        "v_std",
        "l_mean",
        "l_std",
        "a_mean",
        "a_std",
        "b_mean",
        "b_std",
        *[f"h_hist_{i:02d}" for i in range(12)],
        *[f"a_hist_{i:02d}" for i in range(8)],
        *[f"b_hist_{i:02d}" for i in range(8)],
        "frac_yellow",
        "frac_brown",
        "frac_dark",
    ],
    "textura": [
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        *[f"lbp_{i:02d}" for i in range(18)],
    ],
    "manchas": [
        "spot_count",
        "spot_area_frac",
        "spot_mean_area_frac",
        "spot_max_area_frac",
        "spot_mean_solidity",
        "spot_mean_eccentricity",
        "spot_mean_circularity",
    ],
    "combinadas": None,  # se rellena abajo
    "color_manchas": None,
    "control_nitidez": ["sharpness_laplacian_var"],
    "textura_suavizada": None,  # se rellena con las mismas columnas de textura pero de v1_blur
}

FEATURE_GROUPS["combinadas"] = (
    FEATURE_GROUPS["color"] + FEATURE_GROUPS["textura"] + FEATURE_GROUPS["manchas"]
)
FEATURE_GROUPS["color_manchas"] = FEATURE_GROUPS["color"] + FEATURE_GROUPS["manchas"]
# Variante C: color_manchas sin l_std ni v_std (para testar si v_std era huella de captura)
FEATURE_GROUPS["color_manchas_sin_l_v"] = [
    c for c in FEATURE_GROUPS["color_manchas"] if c not in ("l_std", "v_std")
]

# Candidatos a "mejor modelo" (control_nitidez excluido)
CANDIDATE_GROUPS = [
    "color",
    "textura",
    "manchas",
    "combinadas",
    "color_manchas",
    "color_manchas_sin_l_v",
    "textura_suavizada",
]


# --------------------------------------------------------------------------- models & grids


def get_models_and_grids(seed: int = 42) -> dict[str, tuple[Any, dict]]:
    """Devuelve {nombre: (estimador_base, grid_params)}.

    Todas las rejillas son pequeñas y están justificadas por el temario (Tema 6).
    """
    return {
        "knn": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean"],
            },
        ),
        "svm_lineal": (
            LinearSVC(random_state=seed, max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.1, 1, 10]},
        ),
        "svm_rbf": (
            SVC(random_state=seed, class_weight="balanced"),  # sin probability=True
            {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
            {
                "clf__max_depth": [3, 5, 7],
                "clf__min_samples_split": [2, 5],
                "clf__criterion": ["gini"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }


# --------------------------------------------------------------------------- baseline por reglas


def baseline_rules_predict(X: np.ndarray, feature_names: list[str], thresholds: dict) -> np.ndarray:
    """Baseline interpretable: umbrales sobre spot_area_frac y frac_brown (p90 de sanas en train).

    Regla: enferma si (spot_area_frac > thr_spot) OR (frac_brown > thr_brown).
    Los umbrales se fijan con train y se pasan en `thresholds`.
    """
    idx_spot = feature_names.index("spot_area_frac") if "spot_area_frac" in feature_names else -1
    idx_brown = feature_names.index("frac_brown") if "frac_brown" in feature_names else -1

    pred = np.zeros(len(X), dtype=int)
    if idx_spot >= 0:
        pred |= (X[:, idx_spot] > thresholds.get("spot_area_frac", 0)).astype(int)
    if idx_brown >= 0:
        pred |= (X[:, idx_brown] > thresholds.get("frac_brown", 0)).astype(int)
    return pred


def compute_baseline_thresholds(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> dict:
    """Percentil 90 de spot_area_frac y frac_brown en clase healthy (0) de train."""
    healthy = X_train[y_train == 0]
    thresholds = {}
    for fname in ("spot_area_frac", "frac_brown"):
        if fname in feature_names:
            idx = feature_names.index(fname)
            thresholds[fname] = float(np.percentile(healthy[:, idx], 90))
        else:
            thresholds[fname] = 0.0
    return thresholds


# --------------------------------------------------------------------------- metrics


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict:
    """Métricas completas para binario healthy(0)/diseased(1)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_healthy": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_healthy": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_healthy": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_diseased": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_diseased": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_diseased": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def bootstrap_ci_grouped(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    groups: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """IC 95% bootstrap remuestreando hojas (groups), no imágenes.

    Devuelve (lower, upper) del percentil 2.5 y 97.5.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    values = []

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        mask = np.isin(groups, sampled_groups)
        if mask.sum() == 0:
            continue
        try:
            val = metric_fn(
                y_true[mask], y_pred[mask], y_score[mask] if y_score is not None else None
            )
            if np.isfinite(val):
                values.append(val)
        except Exception:
            continue

    if len(values) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


# --------------------------------------------------------------------------- CV & training


def cv_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    base_estimator,
    param_grid: dict,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """GridSearchCV con StratifiedGroupKFold.

    Devuelve: best_params, best_score (balanced_accuracy), cv_results (media/std por fold).
    """
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    # Pre-computar splits como lista (generador no se puede picklear con n_jobs=-1)
    cv_splits = list(cv.split(X, y, groups))

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clone(base_estimator)),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splits,
        scoring="balanced_accuracy",
        n_jobs=1,  # n_jobs=-1 causa problemas de pickling/warnings con loky
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)

    # Métricas por fold en el mejor estimador
    fold_metrics = []
    for _fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        pipe = clone(search.best_estimator_)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[val_idx])
        y_score = None
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(X[val_idx])[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            y_score = pipe.decision_function(X[val_idx])
        fold_metrics.append(compute_metrics(y[val_idx], y_pred, y_score))

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "fold_metrics": fold_metrics,
        "best_estimator": search.best_estimator_,
    }


def train_full(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    base_estimator,
    best_params: dict,
    seed: int = 42,
) -> Pipeline:
    """Entrena en todo train con los mejores hiperparámetros."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                clone(base_estimator).set_params(
                    **{k.replace("clf__", ""): v for k, v in best_params.items()}
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


# --------------------------------------------------------------------------- evaluation on val/test


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict:
    """Evalúa un modelo ya entrenado, con IC bootstrap si hay groups."""
    y_pred = model.predict(X)
    y_score = None
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = model.decision_function(X)

    metrics = compute_metrics(y, y_pred, y_score)

    if groups is not None:
        for metric_name in ("balanced_accuracy", "recall_diseased", "roc_auc"):
            if metric_name in metrics:

                def m_fn(y_t, y_p, y_s, m_name=metric_name):
                    if m_name == "balanced_accuracy":
                        return balanced_accuracy_score(y_t, y_p)
                    if m_name == "recall_diseased":
                        return recall_score(y_t, y_p, pos_label=1, zero_division=0)
                    if m_name == "roc_auc" and y_s is not None:
                        return roc_auc_score(y_t, y_s)
                    return float("nan")

                ci = bootstrap_ci_grouped(y, y_pred, y_score, groups, m_fn)
                metrics[f"{metric_name}_ci95"] = ci

    return metrics


# --------------------------------------------------------------------------- utils


def get_git_commit_hash() -> str:
    """Hash del commit actual (HEAD)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def check_git_clean() -> bool:
    """True si no hay cambios sin commitear."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() == ""
    except Exception:
        return False


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- texture blur


def compute_blur_sigma(
    features_csv: Path,
    sigma_candidates: list[float] | None = None,
    seed: int = 42,
) -> tuple[float, dict]:
    if sigma_candidates is None:
        sigma_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    """Elige sigma: menor que hace que la mediana de sharpness en sanas quede
    dentro del IQR (Q1-Q3) de las enfermas, TRAS aplicar el mismo blur a todas.

    Si ninguna cumple, usa la mayor probada e infórmalo.
    """
    # Cargar metadatos y features
    from src import features as ft
    from src import segmentation as seg

    metadata = pd.read_csv(REPO_ROOT / "data/metadata.csv", keep_default_na=False)
    train_meta = metadata[
        (metadata["split"] == "train") & (metadata["in_subset"].astype(str) == "True")
    ]

    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    results = {}

    for sigma in sigma_candidates:
        h_vals, d_vals = [], []
        for _, row in train_meta.iterrows():
            image = seg.read_image(REPO_ROOT / row.image_path)
            mask = ft.leaf_mask(image, seg_config)
            if not (mask > 0).any():
                continue
            # Aplicar blur a la imagen completa
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            # Medir sharpness en la hoja (mascara erosionada 2px como en sharpness_feature)
            inner = cv2.erode(mask, seg.ellipse_kernel(5)) > 0
            if not inner.any():
                continue
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            val = float(lap[inner].var())
            if row["binary_label"] == "healthy":
                h_vals.append(val)
            else:
                d_vals.append(val)

        if not h_vals or not d_vals:
            results[sigma] = {
                "healthy_median": None,
                "diseased_q1": None,
                "diseased_q3": None,
                "ok": False,
            }
            continue

        h_med = np.median(h_vals)
        d_q1, d_q3 = np.percentile(d_vals, [25, 75])
        ok = d_q1 <= h_med <= d_q3
        results[sigma] = {
            "healthy_median": float(h_med),
            "diseased_q1": float(d_q1),
            "diseased_q3": float(d_q3),
            "ok": bool(ok),
        }
        if ok:
            # AUC post-blur para ver si el sesgo desaparece
            all_vals = np.array(h_vals + d_vals)
            all_labels = np.array([0] * len(h_vals) + [1] * len(d_vals))
            auc = float(roc_auc_score(all_labels, all_vals))
            results[sigma]["auc_post_blur"] = auc
            return sigma, results

    # Ninguna sigma cumple → usar la mayor
    best_sigma = sigma_candidates[-1]
    results[best_sigma]["forced"] = True
    # AUC post-blur para la mayor
    all_vals = np.array(h_vals + d_vals)
    all_labels = np.array([0] * len(h_vals) + [1] * len(d_vals))
    if len(all_vals) > 0:
        results[best_sigma]["auc_post_blur"] = float(roc_auc_score(all_labels, all_vals))
    return best_sigma, results


def generate_v1_blur(
    input_csv: Path,
    output_csv: Path,
    sigma: float,
    seed: int = 42,
) -> None:
    """Regenera características de textura (GLCM + LBP) tras GaussianBlur(sigma).

    Usa la misma máscara y config que v1; solo recalcula textura. El resto de columnas
    se copian del CSV original.
    """
    from src import features as ft
    from src import segmentation as seg

    df = pd.read_csv(input_csv)
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    feat_config = json.loads((REPO_ROOT / "configs/features.json").read_text())
    metadata = pd.read_csv(REPO_ROOT / "data/metadata.csv", keep_default_na=False).set_index(
        "image_id"
    )

    new_rows = []
    for _, row in df.iterrows():
        meta = metadata.loc[row.image_id]
        image = seg.read_image(REPO_ROOT / meta.image_path)
        raw_mask = ft.leaf_mask(image, seg_config)
        mask = ft.prepare_mask(image, raw_mask, feat_config["mask_variant"])
        if not (mask > 0).any():
            mask = raw_mask

        # Blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Recalcular solo textura
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        tex_cfg = feat_config["texture"]
        glcm = ft.glcm_features(gray, mask, tex_cfg["glcm_levels"], tex_cfg["glcm_distance"])
        lbp = ft.lbp_features(gray, mask, tex_cfg["lbp_points"], tex_cfg["lbp_radius"])
        new_tex = {**glcm, **lbp}

        # Copiar fila y reemplazar textura
        new_row = row.to_dict()
        new_row.update(new_tex)
        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(output_csv, index=False, float_format="%.6g")


# --------------------------------------------------------------------------- feature group helpers


def get_feature_columns(group_name: str, df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de columnas para un grupo, validando que existan."""
    if group_name == "textura_suavizada":
        # Mismas columnas que textura
        cols = FEATURE_GROUPS["textura"]
    else:
        cols = FEATURE_GROUPS[group_name]
    if cols is None:
        raise ValueError(f"Grupo {group_name} no definido")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return cols


def prepare_xy(
    df: pd.DataFrame, group_name: str, exclude_mask_empty: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepara X, y, groups para un grupo de características.

    Excluye filas con mask_empty=True si exclude_mask_empty.
    """
    work = df.copy()
    if exclude_mask_empty and "mask_empty" in work.columns:
        work = work[~work["mask_empty"]].copy()

    feature_cols = get_feature_columns(group_name, work)
    X = work[feature_cols].to_numpy(dtype=np.float64)
    y = (work["binary_label"] != "healthy").astype(int).to_numpy()
    groups = (
        work["leaf_group_id"].to_numpy()
        if "leaf_group_id" in work.columns
        else np.arange(len(work))
    )
    return X, y, groups, feature_cols
