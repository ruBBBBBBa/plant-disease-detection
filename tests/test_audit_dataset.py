"""Tests de la auditoría (T01) con un mini dataset sintético; nunca usan los datos reales."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import audit_dataset
from src import data_audit as audit


@pytest.fixture
def fake_plantvillage(tmp_path: Path) -> Path:
    """Mini PlantVillage: 2 clases de tomate + 1 de otra especie, un corrupto y un duplicado.

    - ``Tomato___healthy``: 2 imágenes verdes 32×32 y un fichero corrupto.
    - ``Tomato___Late_blight``: 1 imagen marrón 40×20 y una copia exacta de una sana
      (duplicado entre clases).
    - ``Potato___healthy``: no debe auditarse con ``--species tomato``.
    """
    root = tmp_path / "plantvillage"
    color = root / "raw" / "color"
    seg = root / "raw" / "segmented"
    healthy = color / "Tomato___healthy"
    blight = color / "Tomato___Late_blight"
    for d in (healthy, blight, color / "Potato___healthy", seg / "Tomato___healthy"):
        d.mkdir(parents=True)

    green = np.zeros((32, 32, 3), dtype=np.uint8)
    green[:, :, 1] = 200
    cv2.imwrite(str(healthy / "a___RS_HL 0001.JPG"), green)
    cv2.imwrite(str(healthy / "b___RS_HL 0002.JPG"), np.full((32, 32, 3), 90, dtype=np.uint8))
    (healthy / "c___RS_HL 0003.JPG").write_bytes(b"esto no es un JPEG")
    cv2.imwrite(str(blight / "d___GHLB 0001.jpg"), np.full((20, 40, 3), (30, 60, 120), np.uint8))
    shutil.copy(healthy / "a___RS_HL 0001.JPG", blight / "e___GHLB 0002.jpg")
    cv2.imwrite(str(color / "Potato___healthy" / "p.JPG"), green)

    cv2.imwrite(str(seg / "Tomato___healthy" / "a___RS_HL 0001_final_masked.jpg"), green)
    cv2.imwrite(str(seg / "Tomato___healthy" / "z___RS_HL 0009_final_masked.jpg"), green)
    return root


def test_inventory_counts_unreadable_and_shapes(fake_plantvillage: Path) -> None:
    """Se auditan solo las carpetas de tomate, con recuentos, ilegibles y dimensiones."""
    dirs = audit.find_class_dirs(fake_plantvillage, "plantvillage", "TOMATO")
    inv = audit.build_inventory(dirs, fake_plantvillage)

    assert sorted(inv["class_name"].unique()) == ["Tomato___Late_blight", "Tomato___healthy"]
    assert len(inv) == 5
    assert (~inv["readable"]).sum() == 1
    blight = inv[inv["path"].str.endswith("d___GHLB 0001.jpg")].iloc[0]
    assert (blight["height"], blight["width"], blight["channels"]) == (20, 40, 3)
    assert not inv["path"].str.startswith("/").any()  # rutas relativas


def test_exact_duplicates_cross_class(fake_plantvillage: Path) -> None:
    """La copia exacta de una imagen sana en otra clase se detecta como duplicado entre clases."""
    dirs = audit.find_class_dirs(fake_plantvillage, "plantvillage", "tomato")
    dups = audit.exact_duplicates(audit.build_inventory(dirs, fake_plantvillage))
    assert len(dups) == 1
    assert dups.iloc[0]["files"] == 2
    assert dups.iloc[0]["scope"] == "cross_class"


def test_cross_source_duplicates(fake_plantvillage: Path) -> None:
    """Dos inventarios que comparten un SHA-1 producen un par entre fuentes por fichero."""
    dirs = audit.find_class_dirs(fake_plantvillage, "plantvillage", "tomato")
    inv = audit.build_inventory(dirs, fake_plantvillage)
    other = inv[inv["class_name"] == "Tomato___Late_blight"]
    # La imagen sana "a" y su copia "e" tienen el mismo SHA-1 que "e" en el otro inventario.
    assert len(audit.cross_source_duplicates(inv[inv["path"].str.contains("a___")], other)) == 1


def test_pair_color_segmented(fake_plantvillage: Path) -> None:
    """Empareja por nombre base ignorando ``_final_masked`` y la extensión."""
    pairs = audit.pair_color_segmented(
        fake_plantvillage / "raw" / "color", fake_plantvillage / "raw" / "segmented"
    ).set_index("class_name")
    assert pairs.loc["Tomato___healthy"].tolist() == [1, 2, 1]
    assert pairs.loc["Tomato___Late_blight"].tolist() == [0, 2, 0]


def test_leaf_lookup_and_resolution() -> None:
    """La clave reproduce ``plant_village.py`` y los candidatos ambiguos se resuelven por clase."""
    assert audit.leaf_lookup_key("uuid___RS_HL 0105.JPG") == "rs_hl 0105"
    assert audit.leaf_lookup_key("uuid___RS_HL 0105_final_masked.jpg") == "rs_hl 0105"
    assert audit.leaf_lookup_key("uuid___Leaf 7 copy 2.jpg") == "leaf 7"
    leaf_map = {"rs_hl 0105": ["Tomato___healthy:::3.0", "Potato___healthy:::9.0"]}
    assert audit.resolve_leaf_id("x___RS_HL 0105.JPG", "Tomato___healthy", leaf_map) == (
        "Tomato___healthy:::3.0"
    )
    assert audit.resolve_leaf_id("x___RS_HL 0105.JPG", "Apple___healthy", leaf_map) == ""
    assert audit.resolve_leaf_id("x___RS_HL 9999.JPG", "Tomato___healthy", leaf_map) == ""


def test_svm_split_leakage_detects_shared_leaf(tmp_path: Path) -> None:
    """Una hoja con fotos en train y en test cuenta como fuga."""
    leaf_map = {"rs_hl 0001": ["Tomato___healthy:::1.0"], "rs_hl 0002": ["Tomato___healthy:::1.0"]}
    train, test = tmp_path / "train.txt", tmp_path / "test.txt"
    train.write_text("raw/color/Tomato___healthy/a___RS_HL 0001.JPG\tSVM/train/0/a.JPG\n")
    test.write_text(
        "raw/color/Tomato___healthy/b___RS_HL 0002.JPG\tSVM/test/0/b.JPG\n"
        "raw/color/Apple___healthy/c___X 1.JPG\tSVM/test/1/c.JPG\n"
    )
    result = audit.svm_split_leakage({"train": train, "test": test}, leaf_map, "tomato")
    assert result["train_images"] == 1 and result["test_images"] == 1
    assert result["leaves_in_both"] == 1


def test_sample_grid_is_reproducible(fake_plantvillage: Path) -> None:
    """Misma semilla → misma rejilla; los ficheros corruptos se saltan."""
    paths = list((fake_plantvillage / "raw" / "color" / "Tomato___healthy").iterdir())
    grid = audit.sample_grid(paths, n=3, seed=42, thumb=16, cols=2)
    assert grid is not None and grid.shape == (16, 32, 3) and grid.dtype == np.uint8
    np.testing.assert_array_equal(grid, audit.sample_grid(paths, n=3, seed=42, thumb=16, cols=2))


def test_cli_end_to_end(fake_plantvillage: Path, tmp_path: Path) -> None:
    """El script genera inventario, resumen y rejillas sin tocar la carpeta del dataset."""
    (fake_plantvillage / "leaf-map.json").write_text(
        json.dumps({"rs_hl 0001": ["Tomato___healthy:::1.0"]})
    )
    before = sorted(p.relative_to(fake_plantvillage) for p in fake_plantvillage.rglob("*"))
    out = tmp_path / "out"
    args = ["--source", "plantvillage", "--root", str(fake_plantvillage), "--out", str(out)]
    assert audit_dataset.main(args) == 0

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Ilegibles para OpenCV: **1**" in summary
    assert "Grupos entre clases distintas: 1" in summary
    assert "Agrupación por hoja" in summary
    assert (out / "inventory.csv").is_file()
    assert (out / "samples_tomato_healthy.jpg").is_file()
    assert sorted(p.relative_to(fake_plantvillage) for p in fake_plantvillage.rglob("*")) == before


def test_cli_missing_root_fails_with_message(tmp_path: Path) -> None:
    """Si falta el dataset, el script termina con un mensaje que remite a data/README.md."""
    with pytest.raises(SystemExit, match="data/README.md"):
        audit_dataset.main(["--source", "plantdoc", "--root", str(tmp_path / "nope")])
