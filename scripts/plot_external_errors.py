"""Rejillas de errores de PlantDoc con la máscara de la hoja superpuesta (T06).

Lee las predicciones guardadas (no predice). Para cada variante pedida dibuja falsos negativos
y falsos positivos entre las imágenes concluyentes, y una muestra con semilla de las no
concluyentes. Escribe ``grid_<variante>_<tipo>_<n>.jpg`` y ``error_grid_index.csv`` (número de
celda → imagen) en ``experiments/external/plantdoc_v1/``.

Ejemplo:
    python scripts/plot_external_errors.py --variants A B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402
from src import features as ft  # noqa: E402

CELL, COLS, PER_GRID = 256, 4, 16


def write_attribution(out_dir: Path) -> None:
    """Escribe el crédito junto a las rejillas sin leer ni procesar imágenes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ATTRIBUTION.md").write_text(
        "# Atribución de las fotografías de las rejillas\n\n"
        "PlantDoc: A Dataset for Visual Plant Disease Detection — Singh et al. (2020).\n"
        "Autores: Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, "
        "Sudhakar Kumawat y Nipun Batra.\n\n"
        "Fuente: [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset).\n"
        "Publicación: [Singh et al., 2020](https://doi.org/10.1145/3371158.3371196).\n"
        "Licencia: [Creative Commons Attribution 4.0 International (CC BY 4.0)]"
        "(https://creativecommons.org/licenses/by/4.0/).\n\n"
        "Modificaciones: reescalado conservando la proporción (lado mayor de 256 px), "
        "normalización de canales a BGR, superposición de contornos de máscara y etiquetas, "
        "y composición sobre un lienzo negro en rejillas de errores/no concluyentes. "
        "Las fotografías originales no se han modificado. "
        "La atribución no implica el respaldo de los autores al análisis.\n",
        encoding="utf-8",
    )


def cell(path: Path, seg_config: dict, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Miniatura normalizada con el contorno de la máscara (verde) y una etiqueta."""
    image = ex.read_bgr(path)
    canvas = np.zeros((CELL, CELL, 3), np.uint8)
    if image is not None:
        image = ex.normalize_size(image, CELL)
        mask = ft.leaf_mask(image, seg_config)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        canvas[: image.shape[0], : image.shape[1]] = image
    cv2.rectangle(canvas, (0, 0), (CELL, 18), (0, 0, 0), -1)
    cv2.putText(canvas, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    """Dibuja las rejillas de FN, FP y no concluyentes de las variantes pedidas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["A", "B"])
    parser.add_argument("--sample", type=int, default=48, help="no concluyentes a dibujar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "experiments/external/plantdoc_v1"
    )
    args = parser.parse_args(argv)
    write_attribution(args.out_dir)
    config = json.loads((REPO_ROOT / "configs/external.json").read_text())
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    index_rows = []
    for name in args.variants:
        v = config["variants"][name]
        pred = pd.read_csv(
            args.out_dir / "predictions" / f"pred__{v['group']}__{v['model']}.csv.gz"
        )
        done = pred[~pred["inconclusive"]]
        sets = {
            "FN": done[(done["y_true"] == 1) & (done["y_pred"] == 0)],
            "FP": done[(done["y_true"] == 0) & (done["y_pred"] == 1)],
            "NC": pred[pred["inconclusive"]].sample(args.sample, random_state=args.seed),
        }
        if name != args.variants[0]:
            del sets["NC"]  # «no concluyente» no depende de la variante
        for kind, rows in sets.items():
            rows = rows.sort_values("image_path")
            for g, start in enumerate(range(0, len(rows), PER_GRID), 1):
                chunk = rows.iloc[start : start + PER_GRID]
                cells = []
                for k, r in enumerate(chunk.itertuples(), 1):
                    label = f"#{k} {r.disease_name[:18]}"
                    cells.append(cell(REPO_ROOT / r.image_path, seg_config, label, (0, 255, 255)))
                    index_rows.append((name, kind, g, k, r.image_id, r.image_path, r.disease_name))
                while len(cells) % COLS:
                    cells.append(np.zeros((CELL, CELL, 3), np.uint8))
                rows_img = [np.hstack(cells[i : i + COLS]) for i in range(0, len(cells), COLS)]
                cv2.imwrite(
                    str(args.out_dir / f"grid_{name}_{kind}_{g}.jpg"),
                    np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
            print(name, kind, len(rows))
    cols = ["variant", "kind", "grid", "cell", "image_id", "image_path", "disease_name"]
    pd.DataFrame(index_rows, columns=cols).to_csv(
        args.out_dir / "error_grid_index.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
