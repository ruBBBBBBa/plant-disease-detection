"""Rejillas de aciertos y errores de Taiwán por condición ``leaves × background`` (T08).

Lee las predicciones guardadas (no predice). Para cada condición dibuja hasta ``--per-kind``
errores (FN, FP, no concluyentes) y otros tantos aciertos, elegidos con semilla, con el contorno
de la máscara superpuesto. Escribe ``grid_<variante>_<hojas>_<fondo>.jpg``,
``grid_index.csv`` (celda → imagen) y ``ATTRIBUTION.md`` en ``experiments/external/taiwan_v1/``.

Ejemplo:
    python scripts/plot_taiwan_grids.py --variant B
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_external_errors import CELL, COLS, cell  # noqa: E402

ATTRIBUTION = (
    "# Atribución de las fotografías de las rejillas\n\n"
    "Huang, M.-L. y Chang, Y.-H. (2020). *Dataset of Tomato Leaves*. Mendeley Data, V1. "
    "DOI: [10.17632/ngdgg79rzb.1](https://doi.org/10.17632/ngdgg79rzb.1).\n"
    "Licencia: [Creative Commons Attribution 4.0 International (CC BY 4.0)]"
    "(https://creativecommons.org/licenses/by/4.0/).\n\n"
    "Modificaciones: reescalado conservando la proporción (lado mayor de 256 px), superposición "
    "de contornos de máscara y etiquetas, y composición en rejillas por condición. Las "
    "fotografías originales no se han modificado. La atribución no implica el respaldo de los "
    "autores al análisis.\n"
)


def outcome(row) -> str:
    """OK, FN, FP o NC (no concluyente) de una predicción."""
    if row.inconclusive:
        return "NC"
    if row.y_pred == row.y_true:
        return "OK"
    return "FN" if row.y_true == 1 else "FP"


def main(argv: list[str] | None = None) -> int:
    """Una rejilla por condición con errores y aciertos de la variante pedida."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--per-kind", type=int, default=8, help="errores y aciertos por rejilla")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "experiments/external/taiwan_v1"
    )
    parser.add_argument(
        "--annotations", type=Path, default=REPO_ROOT / "data/annotations/taiwan_conditions.csv"
    )
    args = parser.parse_args(argv)
    (args.out_dir / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    pred = pd.read_csv(args.out_dir / "predictions" / f"pred__{args.variant}.csv.gz")
    cond = pd.read_csv(args.annotations, keep_default_na=False)
    pred = pred.merge(cond.rename(columns={"file": "image_path"}), on="image_path")
    pred["outcome"] = [outcome(r) for r in pred.itertuples()]
    index_rows = []
    for (leaves, background), rows in pred.groupby(["leaves", "background"]):
        errors = rows[rows["outcome"] != "OK"]
        hits = rows[rows["outcome"] == "OK"]
        chosen = pd.concat(
            [
                errors.sample(min(args.per_kind, len(errors)), random_state=args.seed),
                hits.sample(min(args.per_kind, len(hits)), random_state=args.seed),
            ]
        )
        cells = []
        for k, r in enumerate(chosen.itertuples(), 1):
            color = (0, 255, 0) if r.outcome == "OK" else (0, 0, 255)
            text = f"#{k} {r.outcome} {r.disease_name[:14]}"
            cells.append(cell(REPO_ROOT / r.image_path, seg_config, text, color))
            index_rows.append(
                (args.variant, leaves, background, k, r.outcome, r.image_id, r.image_path)
            )
        while len(cells) % COLS:
            cells.append(np.zeros((CELL, CELL, 3), np.uint8))
        grid = np.vstack([np.hstack(cells[i : i + COLS]) for i in range(0, len(cells), COLS)])
        target = args.out_dir / f"grid_{args.variant}_{leaves}_{background}.jpg"
        cv2.imwrite(str(target), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(target.name, f"{len(errors)} errores, {len(hits)} aciertos")
    cols = ["variant", "leaves", "background", "cell", "outcome", "image_id", "image_path"]
    pd.DataFrame(index_rows, columns=cols).to_csv(args.out_dir / "grid_index.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
