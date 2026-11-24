"""Tests del análisis exploratorio a posteriori de T07 (``scripts/explore_own_backgrounds.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/explore_own_backgrounds.py"
spec = importlib.util.spec_from_file_location("explore_own_backgrounds", path)
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


def test_icc1_separates_unit_from_background_variation() -> None:
    # Solo varía la unidad: ICC = 1. Solo varía el fondo (igual en todas las unidades): ICC < 0.
    by_unit = pd.DataFrame({"a": [1.0, 5.0, 9.0], "b": [1.0, 5.0, 9.0], "c": [1.0, 5.0, 9.0]})
    by_background = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [5.0, 5.0, 5.0], "c": [9.0] * 3})
    assert explore.icc1(by_unit)["icc1"] == pytest.approx(1.0)
    res = explore.icc1(by_background)
    assert res["icc1"] < 0 and res["sd_between"] == 0 and res["n"] == 3


def test_agreement_counts_identical_rows() -> None:
    out = pd.DataFrame({"a": ["diseased", "healthy"], "b": ["diseased", "diseased"]})
    assert explore.agreement(out) == {"n": 2, "same": 1}
