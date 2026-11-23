"""Tests de las piezas puras de loop/metric.py (sin ejecutar mutmut)."""

from __future__ import annotations

import pytest

from loop import metric


def test_rewrite_imports_only_touches_import_lines() -> None:
    """Los imports de `src` pasan a `pdd`; cadenas y nombres parecidos no cambian."""
    text = (
        "from src import features as ft\n"
        "    from src.training import prepare_xy\n"
        "import src.segmentation\n"
        'path = "src/features.py"\n'
        "from srcx import y\n"
    )
    assert metric.rewrite_imports(text) == (
        "from pdd import features as ft\n"
        "    from pdd.training import prepare_xy\n"
        "import pdd.segmentation\n"
        'path = "src/features.py"\n'
        "from srcx import y\n"
    )


def test_mutation_score_counts_timeout_and_segfault_as_killed() -> None:
    """(killed + timeout + segfault) / (total − skipped); «no tests» cuenta como vivo."""
    statuses = {
        "a": "killed",
        "b": "timeout",
        "c": "segfault",
        "d": "survived",
        "e": "no tests",
        "f": "skipped",
    }
    assert metric.mutation_score(statuses) == pytest.approx(3 / 5)


def test_mutation_score_rejects_empty() -> None:
    """Sin mutantes evaluados no hay fracción que dar."""
    with pytest.raises(ValueError):
        metric.mutation_score({"a": "skipped"})


def test_parse_results_reads_name_and_status() -> None:
    """Lee la salida de `mutmut results --all true`."""
    text = "    pdd.features.x_f__mutmut_1: killed\n    pdd.features.x_f__mutmut_2: no tests\n"
    assert metric.parse_results(text) == {
        "pdd.features.x_f__mutmut_1": "killed",
        "pdd.features.x_f__mutmut_2": "no tests",
    }


def test_survivor_location_maps_to_real_source_line() -> None:
    """El cambio del diff se sitúa en la línea real de src/, dentro de su función."""
    source = (
        "# cabecera\n"
        "def other(x):\n"
        "    return x + 1\n"
        "\n"
        "def target(x):\n"
        "    y = x * 2\n"
        "    return x + 1\n"
    )
    diff = (
        "--- pdd/m.py\n+++ pdd/m.py\n@@ -1,3 +1,3 @@\n def target(x):\n"
        "     y = x * 2\n-    return x + 1\n+    return x - 1"
    )
    removed, added = metric.changed_lines(diff)
    assert (removed, added) == ("    return x + 1", "    return x - 1")

    spans = metric.function_spans(source)
    assert metric.mutmut_function_key("pdd.m.x_target__mutmut_4", "m") == "x_target"
    assert metric.locate(removed, spans["x_target"], source.splitlines()) == 7
