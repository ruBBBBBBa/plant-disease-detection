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
