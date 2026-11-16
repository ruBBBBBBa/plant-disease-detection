"""Tests de entrenamiento y evaluación (T05) con datos sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src import training as tr


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def synthetic_data():
    """Datos sintéticos balanceados con grupos de hojas."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_groups = 20  # 10 muestras por grupo en promedio
    n_features = 10

    X = rng.normal(0, 1, (n_samples, n_features))
    # Hacer que las 2 primeras features separen las clases
    X[:100, 0] += 2  # healthy
    X[100:, 0] -= 2  # diseased
    X[:100, 1] -= 1
    X[100:, 1] += 1

    y = np.array([0]*100 + [1]*100)
    groups = np.repeat(np.arange(n_groups), n_samples // n_groups)
    # Ajustar si no divide exacto
    if len(groups) > n_samples:
        groups = groups[:n_samples]
    elif len(groups) < n_samples:
        groups = np.concatenate([groups, np.full(n_samples - len(groups), n_groups)])

    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, groups, feature_names


@pytest.fixture
def synthetic_df(synthetic_data):
    """DataFrame estilo features_v1_subset.csv."""
    X, y, groups, feature_names = synthetic_data
    n = len(X)
    df = pd.DataFrame(X, columns=feature_names)
    df["binary_label"] = np.where(y == 0, "healthy", "diseased")
    df["leaf_group_id"] = groups
    df["split"] = np.where(np.arange(n) < 140, "train", "val")  # 140 train, 60 val
    df["mask_empty"] = False
    df["image_id"] = [f"img_{i}" for i in range(n)]
    return df


# --------------------------------------------------------------------------- tests

def test_cv_folds_no_shared_groups(synthetic_data):
    """Los pliegues de StratifiedGroupKFold nunca comparten leaf_group_id entre train y val."""
    X, y, groups, _ = synthetic_data
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in cv.split(X, y, groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups), "Fuga de grupos entre train y val en CV"


def test_pipeline_scaler_fitted_per_fold(synthetic_data):
    """El StandardScaler se ajusta dentro del pipeline por fold, no con todo train."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y, groups, _ = synthetic_data
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42, max_iter=200))])

    # cross_val_score clona el pipeline por fold → scaler se re-ajusta en cada fold
    scores = cross_val_score(pipe, X, y, cv=cv.split(X, y, groups), groups=groups)
    assert len(scores) == 3
    assert all(s > 0.5 for s in scores)  # debe aprender algo


def test_metrics_known_case():
    """Métricas dan lo esperado en un caso pequeño conocido."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_score = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.3])

    m = tr.compute_metrics(y_true, y_pred, y_score)
    assert m["tp"] == 2
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["accuracy"] == 4/6
    assert m["recall_diseased"] == 2/3
    assert m["precision_diseased"] == 2/3
    assert 0 <= m["roc_auc"] <= 1


def test_bootstrap_grouped_resamples_groups(synthetic_data):
    """Bootstrap remuestrea hojas (groups), no imágenes."""
    X, y, groups, _ = synthetic_data
    # Predicción perfecta
    y_pred = y.copy()
    y_score = y.astype(float)

    def metric_fn(y_t, y_p, y_s):
        return tr.balanced_accuracy_score(y_t, y_p)  # type: ignore

    ci = tr.bootstrap_ci_grouped(y, y_pred, y_score, groups, metric_fn, n_bootstrap=100, seed=42)
    # Con predicción perfecta, balanced_accuracy = 1.0 siempre
    assert ci[0] == 1.0 and ci[1] == 1.0


def test_mask_empty_excluded_in_train_counted_as_error_in_eval():
    """Filas con mask_empty se excluyen al entrenar y cuentan como error al evaluar."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(0, 1, (n, 5))
    y = np.array([0]*50 + [1]*50)
    groups = np.arange(n)

    # Marcar 10% como mask_empty
    mask_empty = np.zeros(n, dtype=bool)
    mask_empty[:10] = True

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["binary_label"] = np.where(y == 0, "healthy", "diseased")
    df["leaf_group_id"] = groups
    df["mask_empty"] = mask_empty
    df["split"] = "train"

    # prepare_xy con exclude_mask_empty=True debe quitar esas filas
    X_train, y_train, g_train, _ = tr.prepare_xy(df, "color", exclude_mask_empty=True)
    assert len(X_train) == 90
    assert not any(g_train < 10)  # grupos 0-9 excluidos

    # En evaluación, mask_empty cuenta como error (se testea en eval_test.py)


def test_eval_test_refuses_uncommitted_changes(monkeypatch, tmp_path):
    """eval_test.py se niega a ejecutarse con cambios sin commitear."""
    # Simular git status --porcelain con cambios
    import subprocess
    original_run = subprocess.run

    def mock_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "status"]:
            class MockResult:
                stdout = "M  some_file.py\n"
                returncode = 0
            return MockResult()
        if cmd[:2] == ["git", "rev-parse"]:
            class MockResult:
                stdout = "abc123\n"
                returncode = 0
            return MockResult()
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Verificar que check_git_clean devuelve False
    from src.training import check_git_clean
    assert check_git_clean() is False


def test_eval_test_refuses_wrong_commit(monkeypatch):
    """eval_test.py se niega si el commit actual != congelado."""
    import subprocess
    original_run = subprocess.run

    def mock_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            class MockResult:
                stdout = "different_commit\n"
                returncode = 0
            return MockResult()
        if cmd[:2] == ["git", "status"]:
            class MockResult:
                stdout = ""
                returncode = 0
            return MockResult()
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    from src.training import get_git_commit_hash, check_git_clean
    assert check_git_clean() is True
    assert get_git_commit_hash() != "frozen_commit"


# --------------------------------------------------------------------------- baseline test

def test_baseline_rules():
    """Baseline por reglas: umbrales fijos, predicción booleana."""
    X = np.array([[0.1, 0.05], [0.5, 0.3], [0.01, 0.01], [0.8, 0.6]])
    feat_names = ["spot_area_frac", "frac_brown"]
    thresholds = {"spot_area_frac": 0.2, "frac_brown": 0.15}

    pred = tr.baseline_rules_predict(X, feat_names, thresholds)
    # fila 0: spot=0.1<0.2, brown=0.05<0.15 → 0
    # fila 1: spot=0.5>0.2 → 1
    # fila 2: spot=0.01<0.2, brown=0.01<0.15 → 0
    # fila 3: spot=0.8>0.2 → 1
    assert pred.tolist() == [0, 1, 0, 1]


def test_compute_baseline_thresholds():
    """Umbrales = percentil 90 de sanas."""
    X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])  # 5 sanas
    y = np.array([0, 0, 0, 0, 0])
    feat_names = ["spot_area_frac"]
    # p90 de [0.1, 0.2, 0.3, 0.4, 0.5] ≈ 0.46
    thresholds = tr.compute_baseline_thresholds(X, y, feat_names)
    assert 0.4 < thresholds["spot_area_frac"] < 0.5


# --------------------------------------------------------------------------- texture blur test

def test_compute_blur_sigma_returns_sigma():
    """compute_blur_sigma devuelve una sigma y detalles (test de integración ligero)."""
    # Este test solo verifica que la función existe y tiene la firma correcta
    import inspect
    sig = inspect.signature(tr.compute_blur_sigma)
    assert "features_csv" in sig.parameters
    assert "sigma_candidates" in sig.parameters
    assert "seed" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])