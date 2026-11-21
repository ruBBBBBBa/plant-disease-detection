"""Tests de la inferencia y del protocolo de fotos propias (T07), con datos sintéticos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src import inference as inf
from src import own_eval as oe
from src import segmentation as seg
from src import training as tr

ROOT = Path(__file__).resolve().parents[1]
SEG = json.loads((ROOT / "configs/segmentation.json").read_text())
FEAT = json.loads((ROOT / "configs/features.json").read_text())
OWN = json.loads((ROOT / "configs/own.json").read_text())
GROUP = "color_manchas"
WIDE = {"leaf_area_frac": [0, 1], "border_touch_frac": [0, 1], "mask_solidity": [0, 1]}


def load_script(name: str):
    """Importa un script de ``scripts/`` como módulo."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def leaf_image(h: int = 300, w: int = 400) -> np.ndarray:
    """Hoja verde elíptica con una mancha marrón sobre fondo gris."""
    image = np.full((h, w, 3), 128, np.uint8)
    cv2.ellipse(image, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, (40, 150, 50), -1)
    cv2.circle(image, (w // 2, h // 2), 12, (30, 70, 120), -1)
    return image


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    """SVM lineal estandarizado sobre las columnas de B, ajustado con datos aleatorios."""
    rng = np.random.default_rng(0)
    n = len(tr.FEATURE_GROUPS[GROUP])
    X = rng.normal(size=(80, n)) * rng.uniform(0.1, 10, n)
    y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(random_state=0))])
    return model.fit(X, y)


def test_contributions_sum_to_score_minus_intercept(pipeline) -> None:
    x = np.random.default_rng(1).normal(size=len(tr.FEATURE_GROUPS[GROUP]))
    rows, b = inf.linear_contributions(pipeline, x, tr.FEATURE_GROUPS[GROUP])
    score = pipeline.decision_function(x.reshape(1, -1))[0]
    assert sum(r["contribution"] for r in rows) == pytest.approx(score - b)
    sizes = [abs(r["contribution"]) for r in rows]
    assert sizes == sorted(sizes, reverse=True)


def test_figure_and_json_have_all_fields(pipeline, tmp_path: Path) -> None:
    image = leaf_image()
    analysis = inf.segment_and_extract(image, None, SEG, FEAT, 256, OWN["roi"]["margin"])
    pred = inf.predict_variant(analysis, pipeline, GROUP, WIDE)
    result = inf.result_json(analysis, pred, "B", {"group": GROUP, "model": "svm_lineal"})
    for key in [
        "mode",
        "variant",
        "prediction",
        "score",
        "score_note",
        "intercept",
        "top_contributions",
        "inconclusive",
        "reasons",
        "quality",
        "warnings",
    ]:
        assert key in result
    assert result["prediction"] in {"healthy", "diseased"} and len(result["top_contributions"]) == 5
    text = "\n".join(inf.figure_lines(result))
    assert "no una probabilidad" in text and "Predicción" in text
    assert all(w in text for w in inf.FIXED_WARNINGS)
    assert all(c["feature"] in text for c in result["top_contributions"])
    json.dumps(result)  # serializable
    inf.save_figure(image, analysis, result, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_inconclusive_has_no_prediction(pipeline) -> None:
    analysis = inf.segment_and_extract(leaf_image(), None, SEG, FEAT, 256, 0.1)
    narrow = WIDE | {"leaf_area_frac": [0.99, 1.0]}
    pred = inf.predict_variant(analysis, pipeline, GROUP, narrow)
    assert pred["inconclusive"] and pred["prediction"] is None and pred["score"] is None
    assert pred["top_contributions"] == [] and "leaf_area_frac" in pred["reasons"][0]
    lines = inf.figure_lines(inf.result_json(analysis, pred, "B", {"group": GROUP, "model": "m"}))
    assert any("NO CONCLUYENTE" in line for line in lines)
    assert not any(line.startswith("Predicción") for line in lines)


def test_roi_crop_rescale_and_coordinates() -> None:
    image = np.zeros((800, 1000, 3), np.uint8)
    roi = (100, 200, 300, 150)
    crop, box, inner = inf.crop_roi(image, roi, 0.1, 256)
    assert box == (70, 185, 430, 365)  # 10 % de w (30) y de h (15) por lado
    assert max(crop.shape[:2]) == 256 and crop.shape[:2] == (128, 256)
    s = 256 / 360
    assert inner == (round(30 * s), round(15 * s), round(300 * s), round(150 * s))
    # la ROI pegada al borde se recorta a la imagen
    assert inf.expand_roi((0, 0, 100, 100), 0.1, image.shape) == (0, 0, 110, 110)
    full = inf.mask_to_original(np.full(crop.shape[:2], 255, np.uint8), box, image.shape)
    ys, xs = np.nonzero(full)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == box


def test_roi_grabcut_finds_the_leaf() -> None:
    image = leaf_image(600, 800)
    ref = seg.to_mask(cv2.inRange(image, (35, 145, 45), (45, 155, 55)) > 0)
    ref = seg.fill_holes(ref)
    ys, xs = np.nonzero(ref)
    roi = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    analysis = inf.segment_and_extract(image, roi, SEG, FEAT, 256, 0.1)
    full = inf.mask_to_original(analysis["mask"], analysis["crop_box"], image.shape)
    assert analysis["mode"] == "roi" and seg.iou(full, ref) > 0.9


def test_annotate_saves_several_rois_in_original_coordinates(tmp_path: Path) -> None:
    annotate = load_script("annotate_roi")
    out = tmp_path / "rois.csv"
    scale = inf.display_scale((4000, 3000))
    assert scale == 0.25
    drawn = [(10, 20, 30, 40), (100, 50, 25, 25), (0, 0, 0, 0)]  # el último: ROI vacía, se descarta
    rois = inf.to_original(drawn, scale)
    assert rois == [(40, 80, 120, 160), (400, 200, 100, 100)]
    annotate.write_rois(out, "h01_gris.jpg", rois, ["f1", "f2"])
    annotate.write_rois(out, "h01_blanco.jpg", rois[:1], ["f2"])
    table = pd.read_csv(out)
    assert list(table.columns) == oe.ROI_COLUMNS and len(table) == 3
    gris = table[table["file"] == "h01_gris.jpg"].set_index("leaflet_id")
    assert tuple(gris.loc["f2", ["x", "y", "w", "h"]]) == (400, 200, 100, 100)
    annotate.write_rois(out, "h01_gris.jpg", rois[:1], ["f3"])  # volver a anotar sustituye
    assert set(pd.read_csv(out)["leaflet_id"]) == {"f2", "f3"} and len(pd.read_csv(out)) == 2
    with pytest.raises(ValueError):
        annotate.write_rois(out, "x.jpg", rois, ["f1", "f1"])
    with pytest.raises(ValueError):
        annotate.write_rois(out, "x.jpg", rois[:1], ["hoja1"])


def units(leaf: str, background: str, mode: str, fails: list[bool]) -> list[dict]:
    """Unidades sanas sintéticas: ``True`` = FP (fallo), ``False`` = acierto."""
    return [
        {
            "leaf_id": leaf,
            "background": background,
            "mode": mode,
            "variant": "B",
            "label": "healthy",
            "notes": "",
            "inconclusive": False,
            "prediction": "diseased" if f else "healthy",
        }
        for f in fails
    ]


def test_bootstrap_resamples_compound_leaves() -> None:
    # Una sola hoja con folíolos muy distintos: si se remuestrearan folíolos, el IC tendría anchura.
    one = pd.DataFrame(
        units("h1", "natural", "roi", [True, False] * 5)
        + units("h1", "gris_oscuro", "roi", [False, True, False, False, False])
    )
    diffs = oe.paired_diffs(one, OWN["hypotheses"]["H4"])
    res = oe.paired_leaf_bootstrap(diffs.to_numpy(), 1000, 42)
    assert res["n_leaves"] == 1 and res["ci95"][0] == res["ci95"][1] == pytest.approx(0.5 - 0.2)
    # Cada hoja pesa lo mismo: media de las diferencias por hoja, no de todos los folíolos.
    two = pd.DataFrame(
        units("h1", "natural", "roi", [True] * 10)
        + units("h1", "gris_oscuro", "roi", [False])
        + units("h2", "natural", "roi", [False])
        + units("h2", "gris_oscuro", "roi", [False])
        + units("h3", "natural", "roi", [True])
    )  # h3 sin gris: no entra
    diffs = oe.paired_diffs(two, OWN["hypotheses"]["H4"])
    assert sorted(diffs.index) == ["h1", "h2"] and diffs.mean() == pytest.approx(0.5)


def test_failure_counts_inconclusive_and_h6_compares_modes() -> None:
    rows = units("h1", "natural", "auto", [False]) + units("h1", "natural", "roi", [False, False])
    rows[0] |= {"inconclusive": True, "prediction": ""}  # no concluyente cuenta como fallo
    diffs = oe.paired_diffs(pd.DataFrame(rows), OWN["hypotheses"]["H6"])
    assert diffs.loc["h1"] == pytest.approx(1.0)
    rates = oe.rate_table(pd.DataFrame(rows))
    auto = rates[(rates["mode"] == "auto") & (rates["background"] == "todos")].iloc[0]
    assert auto["fail_rate"] == 1 and np.isnan(auto["fp_rate_conclusive"]) and auto["coverage"] == 0


def test_validate_annotations_reports_problems(tmp_path: Path) -> None:
    for name in ["h01_gris.jpg", "h01_natural.jpg", "h02_gris.jpg"]:
        (tmp_path / name).write_bytes(b"")
    photos = pd.DataFrame(
        [
            {
                "file": "h01_gris.jpg",
                "leaf_id": "h01",
                "background": "gris_oscuro",
                "label": "healthy",
                "notes": "residuo;compuesta",
            },
            {
                "file": "h01_natural.jpg",
                "leaf_id": "h01",
                "background": "natural",
                "label": "healthy",
                "notes": "",
            },
        ]
    )
    rois = pd.DataFrame(
        [{"file": "h01_gris.jpg", "leaflet_id": "f1", "x": 0, "y": 0, "w": 5, "h": 5}]
    )
    problems = oe.validate_annotations(photos, rois, tmp_path, OWN)
    assert any("h02_gris.jpg" in p and "sin anotar" in p for p in problems)
    assert any("h01_natural.jpg" in p and "sin ROI" in p for p in problems)
    assert len(problems) == 2
    ok_rois = pd.concat([rois, rois.assign(file="h01_natural.jpg")])
    assert oe.validate_annotations(photos, ok_rois, tmp_path, OWN) == [
        "h02_gris.jpg: foto sin anotar en own_photos.csv"
    ]
    assert oe.background_combinations(photos).to_dict() == {"gris_oscuro+natural": 1}


def test_eval_own_guard_blocks_second_run(tmp_path: Path, monkeypatch) -> None:
    eval_own = load_script("eval_own")
    monkeypatch.setattr(tr, "get_git_commit_hash", lambda: "abc")
    monkeypatch.setattr(tr, "check_git_clean", lambda: True)
    record = eval_own.reserve_own("abc", tmp_path)
    assert record.name == "execution_abc.json"
    with pytest.raises(SystemExit):
        eval_own.reserve_own("abc", tmp_path)
    with pytest.raises(SystemExit):  # tampoco con otro commit
        monkeypatch.setattr(tr, "get_git_commit_hash", lambda: "def")
        eval_own.reserve_own("def", tmp_path)
