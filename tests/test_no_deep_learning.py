"""Guarda del proyecto: prohíbe redes neuronales y modelos preentrenados.

La restricción no negociable de la asignatura (ver PROJECT_CONTEXT.md, sección 1) es no usar
deep learning en ninguna etapa del sistema. Este test analiza el código del repositorio con
``ast`` (no con búsquedas de texto, para no dar falsos positivos en comentarios) y los ficheros
``requirements*.txt``, y falla si encuentra una dependencia o un uso prohibido.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Módulos de Python cuyo import está prohibido (se compara el primer componente del import).
FORBIDDEN_MODULES = {
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "tf_keras",
    "jax",
    "flax",
    "transformers",
    "timm",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "openvino",
    "tflite_runtime",
    "rembg",
    "segment_anything",
    "mediapipe",
    "paddle",
    "mxnet",
    "fastai",
    "lightning",
    "pytorch_lightning",
    "detectron2",
    "mmcv",
    "mmdet",
}

# Nombres de paquete pip prohibidos (normalizados: minúsculas y "-" en vez de "_").
FORBIDDEN_PACKAGES = {name.replace("_", "-") for name in FORBIDDEN_MODULES} | {
    "tensorflow-cpu",
    "tensorflow-gpu",
    "onnxruntime-gpu",
    "paddlepaddle",
}

# Usos prohibidos dentro de librerías permitidas.
FORBIDDEN_SUBMODULES = {"sklearn.neural_network"}
FORBIDDEN_NAMES = {"MLPClassifier", "MLPRegressor", "BernoulliRBM"}
OPENCV_ALIASES = {"cv2", "cv"}

SKIP_DIRS = {".git", ".venv", "venv", "data", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _iter_files(pattern: str) -> list[Path]:
    """Devuelve los ficheros del repo que cumplen ``pattern``, ignorando entornos y datos."""
    return [
        path
        for path in REPO_ROOT.rglob(pattern)
        if not any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts)
    ]


def _python_sources() -> list[tuple[Path, str]]:
    """Código Python del repo: ficheros ``.py`` y celdas de código de notebooks."""
    sources = [(path, path.read_text(encoding="utf-8")) for path in _iter_files("*.py")]
    for notebook in _iter_files("*.ipynb"):
        cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
        code_lines = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            # Las líneas mágicas de Jupyter (%, !) no son Python válido.
            code_lines += [ln for ln in text.splitlines() if not ln.lstrip().startswith(("%", "!"))]
        sources.append((notebook, "\n".join(code_lines)))
    return sources


def find_forbidden_usages(code: str) -> list[str]:
    """Analiza ``code`` y devuelve una descripción de cada uso prohibido encontrado."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    problems.append(f"import {alias.name} (línea {node.lineno})")
                if alias.name in FORBIDDEN_SUBMODULES:
                    problems.append(f"import {alias.name} (línea {node.lineno})")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            imported = {alias.name for alias in node.names}
            if root in FORBIDDEN_MODULES or node.module in FORBIDDEN_SUBMODULES:
                problems.append(f"from {node.module} import ... (línea {node.lineno})")
            elif root in OPENCV_ALIASES and "dnn" in imported:
                problems.append(f"from {node.module} import dnn (línea {node.lineno})")
            elif node.module == "sklearn" and "neural_network" in imported:
                problems.append(f"from sklearn import neural_network (línea {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if (
                node.attr == "dnn"
                and isinstance(node.value, ast.Name)
                and node.value.id in OPENCV_ALIASES
            ):
                problems.append(f"{node.value.id}.dnn (línea {node.lineno})")
            if node.attr in FORBIDDEN_NAMES:
                problems.append(f"{node.attr} (línea {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{node.id} (línea {node.lineno})")
    return problems


def _requirement_name(line: str) -> str | None:
    """Extrae el nombre normalizado de paquete de una línea de requirements (o None)."""
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    name = re.split(r"[\s<>=!~\[;@]", line, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def test_detector_catches_forbidden_code() -> None:
    """Comprueba que la guarda detecta los casos que debe detectar (y no los permitidos)."""
    bad = (
        "import torch\n"
        "from tensorflow import keras\n"
        "import cv2\n"
        "net = cv2.dnn.readNet('model.onnx')\n"
        "from sklearn.neural_network import MLPClassifier\n"
    )
    assert len(find_forbidden_usages(bad)) >= 4
    good = "import cv2\nfrom sklearn.svm import SVC\n# cv2.dnn no se usa\nhsv = cv2.cvtColor\n"
    assert find_forbidden_usages(good) == []


def test_no_forbidden_imports_in_code() -> None:
    """Ningún fichero de código del repo usa redes neuronales ni modelos preentrenados."""
    problems = []
    for path, code in _python_sources():
        for problem in find_forbidden_usages(code):
            problems.append(f"{path.relative_to(REPO_ROOT)}: {problem}")
    assert not problems, "Uso prohibido de deep learning:\n" + "\n".join(problems)


def test_no_forbidden_requirements() -> None:
    """Ningún requirements*.txt incluye paquetes de deep learning."""
    problems = []
    for req_file in _iter_files("requirements*.txt"):
        for line in req_file.read_text(encoding="utf-8").splitlines():
            name = _requirement_name(line)
            if name in FORBIDDEN_PACKAGES:
                problems.append(f"{req_file.name}: {name}")
    assert not problems, "Dependencias prohibidas:\n" + "\n".join(problems)
