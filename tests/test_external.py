"""Tests del test externo (T06) con imágenes y datos sintéticos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src import external as ex
from src import training as tr

RANGES = {
    "leaf_area_frac": [0.2, 0.6],
    "border_touch_frac": [0.0, 0.2],
    "mask_solidity": [0.6, 0.96],
}


def row(**kw) -> dict:
    """Fila de calidad válida (dentro de rango) con modificaciones."""
    return {
        "mask_empty": False,
        "leaf_area_frac": 0.4,
        "border_touch_frac": 0.1,
        "mask_solidity": 0.8,
    } | kw


def test_normalize_keeps_proportion_and_long_side() -> None:
    for shape in [(600, 400, 3), (100, 300, 3), (256, 256, 3)]:
        out = ex.normalize_size(np.zeros(shape, np.uint8))
        assert max(out.shape[:2]) == 256
        assert out.shape[0] / out.shape[1] == pytest.approx(shape[0] / shape[1], abs=0.01)


def test_four_channel_image_is_read_as_bgr(tmp_path: Path) -> None:
    bgra = np.zeros((20, 30, 4), np.uint8)
    bgra[..., 2], bgra[..., 3] = 200, 255  # rojo opaco
    cv2.imwrite(str(tmp_path / "a.png"), bgra)
    image = ex.read_bgr(tmp_path / "a.png")
    assert image.shape == (20, 30, 3) and image[0, 0, 2] == 200
    assert ex.read_bgr(tmp_path / "no_existe.png") is None


def test_inconclusive_flags_empty_mask_and_out_of_range() -> None:
    assert not ex.is_inconclusive_range(row(), RANGES)
    assert ex.is_inconclusive_range(row(mask_empty=True), RANGES)
    assert ex.is_inconclusive_range(row(leaf_area_frac=0.1), RANGES)  # por debajo
    assert ex.is_inconclusive_range(row(leaf_area_frac=0.7), RANGES)  # por encima
    assert ex.is_inconclusive_range(row(border_touch_frac=0.5), RANGES)
    assert ex.is_inconclusive_range(row(mask_solidity=0.3), RANGES)


def test_quality_ranges_come_from_percentiles_of_valid_rows() -> None:
    table = pd.DataFrame([row(leaf_area_frac=v) for v in np.linspace(0, 1, 1001)])
    table = pd.concat([table, pd.DataFrame([row(mask_empty=True, leaf_area_frac=99)])])
    lo, hi = ex.quality_ranges(table)["leaf_area_frac"]
    assert (lo, hi) == (pytest.approx(0.005), pytest.approx(0.995))  # ignora la máscara vacía
    rate = ex.reference_flag_rate(table, ex.quality_ranges(table))
    assert rate["n"] == 1002 and rate["mask_empty"] > 0


def small_prediction() -> pd.DataFrame:
    """4 sanas (3 bien, 1 mal), 4 enfermas (3 bien, 1 no concluyente que cuenta como error)."""
    return pd.DataFrame(
        {
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_pred": [0, 0, 0, 1, 1, 1, 1, 0],
            "score": [0.1, 0.2, 0.3, 0.9, 0.8, 0.9, 0.7, 0.0],
            "inconclusive": [False] * 7 + [True],
        }
    )


def test_two_readings_with_coverage() -> None:
    pred = small_prediction()
    everything = ex.reading_metrics(pred, "all", with_ci=False)
    only = ex.reading_metrics(pred, "conclusive", with_ci=False)
    assert everything["balanced_accuracy"] == pytest.approx((3 / 4 + 3 / 4) / 2)
    assert everything["n"] == 8 and everything["coverage"] == pytest.approx(7 / 8)
    assert only["balanced_accuracy"] == pytest.approx((3 / 4 + 1) / 2)
    assert only["n"] == 7 and only["coverage"] == pytest.approx(7 / 8)
    with pytest.raises(ValueError):
        ex.reading_metrics(pred, "otra", with_ci=False)


def test_recall_by_disease() -> None:
    pred = small_prediction().assign(disease_name=["healthy"] * 4 + ["x", "x", "y", "y"])
    out = ex.recall_by_disease(pred, "all").set_index("disease_name")
    assert out.loc["healthy", "recall"] == 0.75 and out.loc["y", "recall"] == 0.5


def test_paired_bootstrap_uses_same_images_for_both_variants() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pred = np.where(rng.random(200) < 0.8, y, 1 - y)
    same = ex.paired_bootstrap_diff(y, pred, pred, n_boot=500)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]  # emparejado: sin ruido
    better = ex.paired_bootstrap_diff(y, y, pred, n_boot=500)
    assert better["diff"] > 0 and better["ci95"][0] > 0


def test_hypothesis_verdicts() -> None:
    assert ex.verdict([0.01, 0.1]) == "apoyada"
    assert ex.verdict([-0.1, -0.01]) == "no apoyada (efecto contrario)"
    assert ex.verdict([-0.05, 0.1]) == "no concluyente"
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    good = {"val_ba": 0.95, "y_true": y, "y_pred": y}
    poor = {"val_ba": 0.95, "y_true": y, "y_pred": np.where(rng.random(300) < 0.7, y, 1 - y)}
    result = ex.evaluate_hypothesis(worse=poor, better=good, n_boot=500)
    assert result["verdict"] == "apoyada" and result["diff_delta"] > 0
    assert ex.evaluate_hypothesis(worse=good, better=poor, n_boot=500)["verdict"].startswith(
        "no apoyada"
    )


def load_eval_external():
    """Carga scripts/eval_external.py como módulo (scripts/ no es un paquete)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_external.py"
    spec = importlib.util.spec_from_file_location("eval_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_blocks_a_second_run_even_with_another_commit(tmp_path, monkeypatch) -> None:
    module = load_eval_external()
    monkeypatch.setattr(tr, "get_git_commit_hash", lambda: "abc")
    monkeypatch.setattr(tr, "check_git_clean", lambda: True)
    record = module.reserve_external("abc", tmp_path)
    assert json.loads(record.read_text())["status"] == "started"
    with pytest.raises(SystemExit):
        module.reserve_external("abc", tmp_path)  # mismo commit
    monkeypatch.setattr(tr, "get_git_commit_hash", lambda: "def")
    with pytest.raises(SystemExit):
        module.reserve_external("def", tmp_path)  # commit distinto: sigue prohibido


def test_guard_requires_exact_commit_and_clean_tree(tmp_path, monkeypatch) -> None:
    module = load_eval_external()
    monkeypatch.setattr(tr, "get_git_commit_hash", lambda: "abc")
    monkeypatch.setattr(tr, "check_git_clean", lambda: False)
    with pytest.raises(SystemExit):
        module.reserve_external("abc", tmp_path)
    monkeypatch.setattr(tr, "check_git_clean", lambda: True)
    with pytest.raises(SystemExit):
        module.reserve_external("otro", tmp_path)


def test_grid_attribution_does_not_read_images(tmp_path, monkeypatch):
    """El crédito acompaña a las rejillas y puede regenerarse sin acceder a PlantDoc."""
    from scripts import plot_external_errors as plot

    def forbidden(*args, **kwargs):
        raise AssertionError("No se deben leer ni procesar imágenes")

    monkeypatch.setattr(plot.ex, "read_bgr", forbidden)
    monkeypatch.setattr(plot.ft, "leaf_mask", forbidden)
    plot.write_attribution(tmp_path)
    text = (tmp_path / "ATTRIBUTION.md").read_text()
    for required in (
        "Singh",
        "2020",
        "Nipun Batra",
        "CC BY 4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        "https://github.com/pratikkayal/PlantDoc-Dataset",
        "Modificaciones",
    ):
        assert required in text
