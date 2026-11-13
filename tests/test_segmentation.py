"""Tests de la segmentación (T03) con imágenes sintéticas; nunca usan los datos reales."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_segmentation
from src import segmentation as seg

SIZE = 128


def synthetic_leaf() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elipse verde con una mancha marrón interior sobre fondo gris (algo de ruido).

    Devuelve (imagen BGR, máscara de la hoja, máscara de la mancha), máscaras 0/255.
    """
    rng = np.random.default_rng(0)
    image = np.full((SIZE, SIZE, 3), 150, np.uint8)
    leaf = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(leaf, (64, 64), (40, 28), 20, 0, 360, 255, -1)
    spot = np.zeros_like(leaf)
    cv2.circle(spot, (60, 62), 9, 255, -1)
    image[leaf > 0] = (40, 160, 50)  # verde (BGR)
    image[spot > 0] = (30, 70, 120)  # marrón (BGR)
    noise = rng.integers(-6, 7, image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image, leaf, spot


@pytest.mark.parametrize("name", list(seg.METHODS))
def test_methods_return_binary_mask_of_right_size(name: str) -> None:
    """Cada método devuelve uint8 0/255 del mismo alto y ancho y encuentra la hoja."""
    image, leaf, _ = synthetic_leaf()
    mask = seg.METHODS[name](image)
    assert mask.dtype == np.uint8 and mask.shape == image.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    assert seg.iou(seg.postprocess(mask), leaf) > 0.8


def test_green_threshold_misses_brown_spot_and_fill_holes_recovers_it() -> None:
    """El umbral de verde deja la mancha como agujero; el relleno de huecos la recupera."""
    image, leaf, spot = synthetic_leaf()
    green = seg.segment_hsv_green(image)
    assert (green[spot > 0] == 0).mean() > 0.9
    filled = seg.fill_holes(green)
    assert (filled[spot > 0] == 255).all()
    assert seg.iou(filled, leaf) > 0.95


def test_kmeans_keeps_brown_spot() -> None:
    """K-means (regla del borde) no descarta la mancha marrón por no ser verde."""
    image, _, spot = synthetic_leaf()
    mask = seg.segment_kmeans(image, k=3)
    assert (mask[spot > 0] == 255).mean() > 0.9


def test_largest_component_drops_small_blobs() -> None:
    """Se conserva la mayor componente conexa y se descartan las pequeñas."""
    mask = np.zeros((50, 50), np.uint8)
    mask[10:30, 10:30] = 255
    mask[40:43, 40:43] = 255
    kept = seg.largest_component(mask)
    assert kept[20, 20] == 255 and kept[41, 41] == 0


def test_iou_and_dice_identical_and_disjoint() -> None:
    """IoU y Dice valen 1 con máscaras idénticas y 0 con disjuntas."""
    a = np.zeros((10, 10), np.uint8)
    a[:5] = 255
    b = np.zeros_like(a)
    b[5:] = 255
    assert seg.iou(a, a) == 1.0 and seg.dice(a, a) == 1.0
    assert seg.iou(a, b) == 0.0 and seg.dice(a, b) == 0.0


def test_reference_mask_on_black_background() -> None:
    """La referencia separa la hoja del fondo negro con ruido JPEG y rellena huecos oscuros."""
    _, leaf, spot = synthetic_leaf()
    segmented = np.zeros((SIZE, SIZE, 3), np.uint8)
    segmented[leaf > 0] = (40, 160, 50)
    segmented[spot > 0] = (5, 5, 5)  # lesión casi negra dentro de la hoja
    segmented[0:3, 0:3] = 8  # ruido de compresión en el fondo
    ref = seg.reference_mask(segmented, threshold=20)
    assert set(np.unique(ref)) <= {0, 255}
    assert seg.iou(ref, leaf) == 1.0


def test_non_green_coverage_known_case() -> None:
    """Cobertura no verde = fracción de la mancha marrón (referencia) que incluye la máscara."""
    image, leaf, spot = synthetic_leaf()
    non_green = seg.non_green_mask(image, leaf)
    assert seg.iou(non_green, spot) > 0.9  # lo no verde de la hoja es la mancha
    half = spot.copy()
    half[:, 61:] = 0  # máscara que solo incluye la mitad izquierda de la mancha
    expected = ((half > 0) & (non_green > 0)).sum() / (non_green > 0).sum()
    assert seg.non_green_coverage(half, non_green) == pytest.approx(expected)
    assert 0.3 < expected < 0.7
    assert np.isnan(seg.non_green_coverage(leaf, np.zeros_like(leaf)))


def test_segmented_path_without_uuid_prefix(tmp_path: Path) -> None:
    """Empareja color → segmented, también si el segmentado no lleva prefijo UUID."""
    color = tmp_path / "raw" / "color" / "Tomato___X" / "uuid___Com.G_SpM_FL 1378.JPG"
    seg_dir = tmp_path / "raw" / "segmented" / "Tomato___X"
    seg_dir.mkdir(parents=True)
    (seg_dir / "Com.G_SpM_FL 1378_final_masked.jpg").touch()
    assert seg.segmented_path(color) == seg_dir / "Com.G_SpM_FL 1378_final_masked.jpg"


def test_test_split_requires_frozen_choice(tmp_path: Path) -> None:
    """``--split test`` se niega a ejecutarse si aún no hay método elegido con val."""
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"image_path": []}).to_csv(metadata, index=False)
    config = tmp_path / "config.json"
    config.write_text('{"default_method": null}')
    args = ["--split", "test", "--metadata", str(metadata), "--config", str(config)]
    with pytest.raises(SystemExit, match="congelado"):
        evaluate_segmentation.main([*args, "--out", str(tmp_path / "out")])


def test_rim_fraction() -> None:
    """Un anillo en el borde da fracción 1; una mancha central da 0."""
    ref = np.zeros((60, 60), np.uint8)
    cv2.circle(ref, (30, 30), 20, 255, -1)
    ring = cv2.subtract(ref, cv2.erode(ref, seg.ellipse_kernel(3)))
    center = np.zeros_like(ref)
    cv2.circle(center, (30, 30), 4, 255, -1)
    assert seg.rim_fraction(ring, ref) == 1.0
    assert seg.rim_fraction(center, ref) == 0.0


@pytest.mark.parametrize("script_name", ["evaluate_segmentation", "tune_segmentation"])
@pytest.mark.parametrize("missing", ["color", "segmented"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_cli_reports_unreadable_images(
    script_name: str, missing: str, corrupt: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluación y ajuste identifican imágenes ausentes/corruptas antes de generar salidas."""
    import json

    from scripts import tune_segmentation

    script = evaluate_segmentation if script_name == "evaluate_segmentation" else tune_segmentation
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    color = tmp_path / "raw/color/Tomato___healthy/leaf.JPG"
    reference = tmp_path / "raw/segmented/Tomato___healthy/leaf_final_masked.jpg"
    bad_path = color if missing == "color" else reference
    for path in (color, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path == bad_path:
            if corrupt:
                path.write_text("imagen ilegible")
        else:
            assert cv2.imwrite(str(path), synthetic_leaf()[0])
    split = "val" if script is evaluate_segmentation else "train"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_path": color.relative_to(tmp_path).as_posix(),
                "source": "plantvillage",
                "split": split,
                "in_subset": True,
                "disease_name": "healthy",
            }
        ]
    ).to_csv(metadata, index=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"postprocess_kernel": 3, "reference_threshold": 22}))
    out = tmp_path / "out"
    args = ["--metadata", str(metadata), "--config", str(config), "--out", str(out)]
    if script is evaluate_segmentation:
        args += ["--split", "val"]
    with pytest.raises(SystemExit) as error:
        script.main(args)
    assert str(bad_path) in str(error.value)
    assert "data/README.md" in str(error.value)
    assert not out.exists()
