"""Tests de T08 (fase A): mapeo de Taiwán, casi-duplicados frente a PlantVillage y auditoría."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import data_audit as audit
from src import manifest as mf

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Importa un script de ``scripts/`` como módulo."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gradient(seed: int) -> np.ndarray:
    """Imagen sintética con estructura (el dHash de una imagen lisa no discrimina)."""
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 256, (8, 9, 3), dtype=np.uint8), (256, 256))


def test_taiwan_mapping_marks_unseen_diseases() -> None:
    mapping = mf.load_label_mapping(REPO_ROOT / "configs/label_mapping.csv")
    tw = mapping[mapping["source"] == "taiwan"].set_index("folder")
    assert len(tw) == 6
    assert tw.loc["health", "binary_label"] == "healthy"
    assert (tw.drop(index="health")["binary_label"] == "diseased").all()
    pv = set(mapping.loc[mapping["source"] == "plantvillage", "disease_name"])
    unseen = {d for d in tw["disease_name"] if d not in pv}
    assert unseen == {"black_mold", "gray_spot", "powdery_mildew"}


def test_near_duplicate_of_plantvillage_is_excluded() -> None:
    prep = load_script("prepare_taiwan")
    pv_image = gradient(0)
    pv = pd.DataFrame(
        {
            "image_id": ["pv_a", "pv_b"],
            "image_path": ["a.jpg", "b.jpg"],
            "split": ["train", "val"],
            "disease_name": ["healthy", "bacterial_spot"],
            "dhash": [mf.dhash(pv_image), mf.dhash(gradient(1))],
        }
    )
    # Misma imagen reducida a 227 px (como en Taiwán) y una imagen distinta.
    small = cv2.resize(pv_image, (227, 227), interpolation=cv2.INTER_AREA)
    inventory = pd.DataFrame(
        {
            "path": ["tw/health/x.jpg", "tw/Gray spot/y.jpg"],
            "class_name": ["health", "Gray spot"],
            "original_split": ["train", "test"],
            "readable": [True, True],
            "sha1": ["s1", "s2"],
            "dhash": [mf.dhash(small), mf.dhash(gradient(2))],
        }
    )
    mapping = mf.load_label_mapping(REPO_ROOT / "configs/label_mapping.csv")
    df, near = prep.build_manifest(inventory, mapping, pv, threshold=2)
    assert df["exclusion_reason"].tolist() == ["near_duplicate_of_pv:pv_a", ""]
    assert df["split"].tolist() == ["excluded", "external_test"]
    assert df["seen_in_training"].tolist() == [True, False]
    assert near["pv_id"].tolist() == ["pv_a"]


def test_taiwan_audit_reads_only_preprocessed_data(tmp_path: Path) -> None:
    for part in (
        "Preprocessed data/Train/health",
        "Preprocessed data/Test/Gray spot",
        "data augmentation/Train/health",
    ):
        (tmp_path / part).mkdir(parents=True)
    dirs = audit.find_class_dirs(tmp_path, "taiwan", "tomato")
    assert [(d.parent.name, d.name, s) for d, s in dirs] == [
        ("Train", "health", "train"),
        ("Test", "Gray spot", "test"),
    ]


def test_annotate_conditions_keys_order_and_csv(tmp_path: Path) -> None:
    tool = load_script("annotate_conditions")
    state = tool.apply_key({}, "3")
    state = tool.apply_key(state, "x")  # tecla sin efecto
    assert tool.apply_key(state, "s") == {"leaves": "multiple", "background": "simple"}
    files = [f"f{i}.jpg" for i in range(20)]
    order = tool.annotation_order(files, 42)
    assert (
        sorted(order) == sorted(files)
        and order != files
        and order == tool.annotation_order(files, 42)
    )
    out = tmp_path / "conditions.csv"
    assert tool.apply_key({}, "1")["leaves"] == "leaflet"
    assert tool.apply_key({}, "2")["leaves"] == "compound"
    tool.write_annotation(out, "b.jpg", "leaflet", "simple")
    tool.write_annotation(out, "a.jpg", "multiple", "complex")
    table = tool.write_annotation(out, "b.jpg", "compound", "complex")  # corrección
    assert table.values.tolist() == [
        ["a.jpg", "multiple", "complex"],
        ["b.jpg", "compound", "complex"],
    ]
    try:
        tool.write_annotation(out, "c.jpg", "one", "simple")
    except ValueError:
        pass
    else:
        raise AssertionError("debería rechazar etiquetas no válidas")


def test_gray_world_equalizes_channel_means_in_uint8() -> None:
    from src import features as ft

    rng = np.random.default_rng(0)
    image = rng.integers(0, 120, (64, 64, 3)).astype(np.uint8)
    image[:, :, 2] = np.clip(image[:, :, 2].astype(int) + 100, 0, 255)  # dominante roja (BGR)
    out = ft.gray_world(image)
    means = out.reshape(-1, 3).mean(axis=0)
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert np.ptp(means) < 1.0  # medias iguales salvo redondeo y recorte
    assert np.ptp(image.reshape(-1, 3).mean(axis=0)) > 50
    black = np.zeros((4, 4, 3), np.uint8)  # canal con media 0: ganancia 1, sin dividir por 0
    assert (ft.gray_world(black) == 0).all()


def test_b_prime_shares_mask_and_quality_with_b() -> None:
    import json

    from src import features as ft

    cfg = json.loads((REPO_ROOT / "configs/features.json").read_text())
    seg_cfg = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    image = np.full((200, 200, 3), (60, 60, 140), np.uint8)  # fondo con dominante
    cv2.ellipse(image, (100, 100), (60, 40), 0, 0, 360, (40, 150, 60), -1)  # hoja verde
    mask = ft.leaf_mask(image, seg_cfg)
    b = ft.extract_features(image, mask, cfg)
    b_prime = ft.extract_features(image, mask, cfg, feature_image=ft.gray_world(image))
    shared = ["mask_empty", "leaf_area_frac", "border_touch_frac", "mask_solidity"]
    assert {k: b[k] for k in shared} == {k: b_prime[k] for k in shared}
    assert b["b_mean"] != b_prime["b_mean"]  # solo cambia el color que ven las características
    assert ft.extract_features(image, mask, cfg, feature_image=image) == b


# --------------------------------------------------------------------------- fase B (sintético)


def test_unpaired_bootstrap_resamples_each_group_with_its_size() -> None:
    from src import external as ex

    better, worse = np.r_[np.ones(90), np.zeros(10)], np.r_[np.ones(30), np.zeros(30)]
    r = ex.unpaired_bootstrap_diff(better, worse, n_boot=2000, seed=0)
    assert r["diff"] == 0.9 - 0.5 and r["verdict"] == "apoyada"
    assert r["ci95"][0] < 0.4 < r["ci95"][1]
    assert ex.unpaired_bootstrap_diff(worse, better, 2000, 0)["verdict"].startswith("no apoyada")
    same = ex.unpaired_bootstrap_diff(worse, worse, 2000, 0)
    assert same["verdict"] == "no concluyente"


def test_size_rule_overrides_verdict_below_twenty() -> None:
    from src import external as ex

    r = {"diff": 0.3, "ci95": [0.1, 0.5], "verdict": "apoyada"}
    assert ex.size_rule(r, {"foco": 20, "resto": 300})["verdict"] == "apoyada"
    small = ex.size_rule(r, {"foco": 19, "resto": 300})
    assert small["verdict"] == "no concluyente por tamaño" and small["small_cells"] == {"foco": 19}


def test_extract_image_gray_world_keeps_quality_and_changes_color(tmp_path: Path) -> None:
    import json

    from src import external as ex

    image = np.full((227, 227, 3), (60, 60, 140), np.uint8)
    cv2.ellipse(image, (113, 113), (70, 45), 0, 0, 360, (40, 150, 60), -1)
    path = tmp_path / "leaf.png"
    cv2.imwrite(str(path), image)
    cfg = json.loads((REPO_ROOT / "configs/features.json").read_text())
    seg = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    plain, _ = ex.extract_image(path, seg, cfg, 3.0, 256)
    gray, _ = ex.extract_image(path, seg, cfg, 3.0, 256, gray_world=True)
    for k in ["mask_empty", *ex.QUALITY_COLUMNS]:
        assert plain[k] == gray[k]
    assert plain["b_mean"] != gray["b_mean"]


def test_eval_taiwan_refuses_without_full_annotations(tmp_path: Path) -> None:
    import pytest

    module = load_script("eval_external")
    rows = pd.DataFrame({"image_path": ["a.jpg", "b.jpg"]})
    with pytest.raises(SystemExit):
        module.check_annotations(rows, tmp_path / "missing.csv")
    path = tmp_path / "ann.csv"
    pd.DataFrame({"file": ["a.jpg"], "leaves": ["leaflet"], "background": ["simple"]}).to_csv(
        path, index=False
    )
    with pytest.raises(SystemExit):
        module.check_annotations(rows, path)
    module.check_annotations(rows.iloc[:1], path)  # completa: no se queja


def test_report_taiwan_end_to_end_on_synthetic_predictions(tmp_path: Path) -> None:
    import json

    rng = np.random.default_rng(0)
    n = 240
    paths = [f"img{i:03d}.jpg" for i in range(n)]
    y = (np.arange(n) % 4 != 0).astype(int)  # 60 sanas, 180 enfermas
    leaves = np.where(np.arange(n) < 120, "leaflet", rng.choice(["compound", "multiple"], n))
    background = np.where((y == 1) & (np.arange(n) < 160), "simple", "complex")
    disease = np.where(y == 0, "healthy", rng.choice(["late_blight", "gray_spot"], n))
    pd.DataFrame({"file": paths, "leaves": leaves, "background": background}).to_csv(
        tmp_path / "ann.csv", index=False
    )
    pd.DataFrame({"image_path": paths, "seen_in_training": disease != "gray_spot"}).to_csv(
        tmp_path / "meta.csv", index=False
    )
    (tmp_path / "predictions").mkdir()
    inconclusive = rng.random(n) < 0.1
    for name in ("A", "B", "B_prime"):
        pred = np.where(rng.random(n) < 0.8, y, 1 - y)
        pred = np.where(inconclusive, 1 - y, pred)
        pd.DataFrame(
            {
                "image_id": paths,
                "image_path": paths,
                "disease_name": disease,
                "binary_label": np.where(y == 1, "diseased", "healthy"),
                "y_true": y,
                "y_pred": pred,
                "score": rng.random(n),
                "inconclusive": inconclusive,
            }
        ).to_csv(tmp_path / "predictions" / f"pred__{name}.csv.gz", index=False)
    m = {"coverage": 0.8, "balanced_accuracy": 0.6}
    plantdoc = {"variants": {v: {"all": m, "conclusive": m} for v in ("A", "B")}}
    (tmp_path / "plantdoc.json").write_text(json.dumps(plantdoc))
    config = json.loads((REPO_ROOT / "configs/taiwan.json").read_text())
    config["bootstrap"] |= {"n_boot": 500, "n_boot_descriptive": 20}  # rápido
    (tmp_path / "taiwan.json").write_text(json.dumps(config))
    load_script("report_taiwan").main(
        [
            "--out-dir", str(tmp_path),
            "--config", str(tmp_path / "taiwan.json"),
            "--annotations", str(tmp_path / "ann.csv"),
            "--metadata", str(tmp_path / "meta.csv"),
            "--plantdoc", str(tmp_path / "plantdoc.json"),
        ]
    )  # fmt: skip
    result = json.loads((tmp_path / "metrics.json").read_text())
    h = result["hypotheses"]
    assert h["H7"]["n_focus"] + h["H7"]["n_rest"] == int(y.sum())  # solo enfermas
    assert h["H7"]["n_focus"] == int(((leaves == "leaflet") & (background == "simple")).sum())
    assert h["H8"]["verdict"] in {"apoyada", "no apoyada (efecto contrario)", "no concluyente"}
    assert "H9" in h and "verdict" in h["H9"]
    assert "## Hipótesis" in (tmp_path / "tables.md").read_text()


def test_condition_readings_without_healthy_and_without_conclusive() -> None:
    """Sin sanas BA/recall sano son no medibles; NC no entra en la matriz concluyente."""
    report = load_script("report_taiwan")
    cell = pd.DataFrame(
        {
            "leaves": ["leaflet"] * 3,
            "background": ["simple"] * 3,
            "y_true": [1, 1, 1],
            "y_pred": [1, 0, 0],
            "score": [0.8, 0.2, -0.5],
            "inconclusive": [False, False, True],
        }
    )
    all_rows, conclusive = report.condition_rows(cell, 20, 42)
    for row in (all_rows, conclusive):
        assert np.isnan(row["balanced_accuracy"]) and np.isnan(row["recall_healthy"])
        assert row["n_healthy"] == row["n_healthy_total"] == 0
        assert np.isclose(row["coverage"], 2 / 3)
        assert report.fmt(row["balanced_accuracy"]) == "no medible"
    assert all_rows["confusion_matrix"] == [[0, 0], [2, 1]]
    assert conclusive["confusion_matrix"] == [[0, 0], [1, 1]]
    assert conclusive["n_diseased"] == 2 and conclusive["recall_diseased"] == 0.5
    cell["inconclusive"] = True
    empty = report.condition_rows(cell, 20, 42)[1]
    assert empty["n"] == 0 and empty["coverage"] == 0
    assert empty["confusion_matrix"] == [[0, 0], [0, 0]]
    assert np.isnan(empty["recall_diseased"])
