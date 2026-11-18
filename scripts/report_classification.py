"""Regenera métricas y rejillas desde predicciones guardadas, sin entrenar ni predecir (T05)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import features as ft  # noqa: E402
from src import segmentation as seg  # noqa: E402
from src import training as tr  # noqa: E402
from src.data_audit import md_table  # noqa: E402


def summarize(path: Path) -> dict:
    """Métricas e IC agrupados reconstruidos del CSV, incluyendo errores mask_empty."""
    df = pd.read_csv(path)
    y, pred, score = (df[c].to_numpy() for c in ("y_true", "y_pred", "score"))
    metrics = tr.compute_metrics(y, pred, score)
    functions = {
        "balanced_accuracy": lambda a, b, c: balanced_accuracy_score(a, b),
        "recall_diseased": lambda a, b, c: recall_score(a, b, zero_division=0),
        "roc_auc": lambda a, b, c: roc_auc_score(a, c),
    }
    for name, fn in functions.items():
        metrics[name + "_ci95"] = tr.bootstrap_ci_grouped(
            y, pred, score, df.leaf_group_id.to_numpy(), fn
        )
    metrics.update(n_samples=len(df), n_mask_empty=int(df.mask_empty.sum()))
    return metrics


def error_grid(path: Path, output: Path) -> None:
    """Rejilla de errores guardados: contornos hoja y regiones candidatas, sin inferencia."""
    df = pd.read_csv(path)
    errors = df[df.y_true != df.y_pred]
    meta = pd.read_csv(REPO_ROOT / "data/metadata.csv").set_index("image_id")
    cfg = json.loads((REPO_ROOT / "configs/features.json").read_text())
    scfg = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    tiles = []
    for row in errors.head(24).itertuples():
        image = seg.read_image(REPO_ROOT / meta.loc[row.image_id, "image_path"])
        mask = ft.leaf_mask(image, scfg)
        candidate = ft.spot_mask(image, mask, cfg)
        for m, color in ((mask, (0, 255, 0)), (candidate, (255, 0, 255))):
            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1)
        tile = cv2.resize(image, (160, 160))
        cv2.putText(
            tile,
            f"{'FN' if row.y_true else 'FP'} {row.disease_name[:14]}",
            (2, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    if not tiles:
        tiles = [np.full((160, 160, 3), 255, np.uint8)]
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 4)
    cv2.imwrite(
        str(output), np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    )


def main(argv: list[str] | None = None) -> int:
    """Informes derivados exclusivamente de los CSV de val/test ya guardados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(args.exp_dir.glob("val__*.csv.gz")) + sorted(
        (args.exp_dir / "test_evaluation").glob("test_*__*.csv.gz")
    )
    records, complete = [], {}
    for path in paths:
        split, group, model = path.name.removesuffix(".csv.gz").split("__")
        metrics = summarize(path)
        complete[path.name] = metrics
        records.append({"split": split, "group": group, "model": model, **metrics})
    table = pd.DataFrame(records)
    table.to_csv(args.exp_dir / "metrics.csv", index=False)
    (args.exp_dir / "metrics.json").write_text(json.dumps(complete, indent=2) + "\n")
    cfg = json.loads((REPO_ROOT / "configs/classification.json").read_text())
    chosen = cfg["chosen_group"], cfg["chosen_model"]
    lines = [
        "# Clasificación: métricas reconstruidas de predicciones",
        "",
        "Scores no calibrados. IC95: bootstrap por hoja con multiplicidad.",
        "",
        md_table(table),
        "",
        "## Análisis del sesgo",
        "",
        "Mismo modelo entre grupos: diferencias descriptivas, no causales.",
        "",
        md_table(table[["split", "group", "model", "balanced_accuracy", "roc_auc"]]),
    ]
    for path in paths:
        split, group, model = path.name.removesuffix(".csv.gz").split("__")
        if (group, model) != chosen:
            continue
        df = pd.read_csv(path)
        disease = (
            df.assign(correct=df.y_true == df.y_pred)
            .groupby("disease_name")
            .agg(support=("correct", "size"), recall=("correct", "mean"))
        )
        lines += ["", f"## {split}: elegido", "", md_table(disease.reset_index())]
        error_grid(path, args.exp_dir / f"errors_{split}.jpg")
    lines += [
        "",
        "Rejillas: verde = hoja; magenta = regiones candidatas, no localización del clasificador. "
        "FN/FP guardados; hasta 24 errores por split.",
    ]
    (args.exp_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
