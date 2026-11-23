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
    state = tool.apply_key({}, "2")
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
    tool.write_annotation(out, "b.jpg", "single", "simple")
    tool.write_annotation(out, "a.jpg", "multiple", "complex")
    table = tool.write_annotation(out, "b.jpg", "single", "complex")  # corrección
    assert table.values.tolist() == [
        ["a.jpg", "multiple", "complex"],
        ["b.jpg", "single", "complex"],
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
