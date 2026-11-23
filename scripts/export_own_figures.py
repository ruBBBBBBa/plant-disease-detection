"""Copias reducidas y versionables de las figuras del test con fotos propias (T07, D046–D047).

Lee las rejillas de ``plot_own.py`` y las figuras de ejemplo de ``predict.py`` (en
``data/processed/own_v1/``, fuera de git), las reduce (lado mayor ``--max-side``, JPEG) y las
escribe en ``experiments/own/v1/figures/``. Las fotos originales no se copian. Se para si alguna
copia supera el límite de 2 MB del repositorio.

Ejemplo:
    python scripts/export_own_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2 * 1024 * 1024  # bytes; CI rechaza ficheros mayores


def shrink(image, max_side: int):
    """Reduce la imagen para que su lado mayor sea ``max_side`` (nunca amplía; INTER_AREA)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> int:
    """Reduce y copia cada figura; error si falta alguna o si una copia pasa de 2 MB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "data/processed/own_v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments/own/v1/figures")
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args(argv)
    sources = sorted(args.src.glob("grid_*.jpg")) + sorted(args.src.glob("examples/*.png"))
    if not sources:
        raise SystemExit(f"Error: no hay figuras en {args.src}; ejecuta plot_own.py y predict.py")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in sources:
        image = cv2.imread(str(path))
        target = args.out / f"{path.stem}.jpg"
        cv2.imwrite(
            str(target), shrink(image, args.max_side), [cv2.IMWRITE_JPEG_QUALITY, args.quality]
        )
        size = target.stat().st_size
        if size > LIMIT:
            raise SystemExit(f"Error: {target} ocupa {size} B (> 2 MB); baja --max-side")
        print(f"{target.relative_to(REPO_ROOT)}  {size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
