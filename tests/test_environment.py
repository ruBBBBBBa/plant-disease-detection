"""Comprueba que el entorno tiene las librerías previstas y que OpenCV incluye contrib."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("module", ["cv2", "numpy", "pandas", "skimage", "sklearn", "matplotlib"])
def test_required_libraries_import(module: str) -> None:
    """Las dependencias de requirements.txt se pueden importar."""
    importlib.import_module(module)


def test_opencv_contrib_and_sift_available() -> None:
    """opencv-contrib está instalado (módulo ximgproc) y SIFT funciona en una imagen sintética."""
    import cv2

    assert hasattr(cv2, "ximgproc"), "Instala opencv-contrib-python (no opencv-python)"
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), 255, -1)
    keypoints = cv2.SIFT_create().detect(image, None)
    assert isinstance(keypoints, (list, tuple))
