"""Rejillas y contribuciones medias del test con fotos propias (T07, fase B), solo descriptivo.

Vuelve a segmentar cada foto con el código congelado (es determinista) y comprueba que la
puntuación de B coincide con la guardada en ``predictions.csv.gz``; si no coincide, se para.
No ajusta nada.

- ``contributions_B_roi.csv`` (versionable): contribución media de cada característica de B
  (coef. × valor estandarizado) sobre los folíolos sanos concluyentes, para explicar los FP.
- Rejillas con miniaturas de las fotos propias en ``--grid-dir`` (por defecto fuera de git:
  el alumno decide si se versionan).

Ejemplo:
    python scripts/plot_own.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402
from src import inference as inf  # noqa: E402

COLORS = {"ok": (0, 200, 0), "fp": (0, 0, 255), "nc": (160, 160, 160)}  # BGR


def outcome(row: pd.Series) -> str:
    """``nc`` (no concluyente), ``fp`` (predicha enferma) u ``ok``."""
    if row["inconclusive"]:
        return "nc"
    return "fp" if row["prediction"] == "diseased" else "ok"


def tile(image: np.ndarray, text: str, size: int = 240) -> np.ndarray:
    """Miniatura cuadrada con un rótulo."""
    scale = size / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((size + 22, size, 3), 255, np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    cv2.putText(canvas, text, (3, size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return canvas


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    """Pega miniaturas del mismo tamaño en filas de ``cols``."""
    blank = np.full_like(tiles[0], 255)
    tiles = tiles + [blank] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def main(argv: list[str] | None = None) -> int:
    """Recalcula, verifica y dibuja."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data/raw/own")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/own/v1")
    parser.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "data/processed/own_v1")
    args = parser.parse_args(argv)
    cfg = inf.load_configs()
    pred = pd.read_csv(args.dir / "predictions.csv.gz", keep_default_na=False)
    pred["inconclusive"] = pred["inconclusive"].astype(str) == "True"
    pred = pred[pred["variant"] == "B"]
    variant = cfg["external"]["variants"]["B"]
    pipeline = inf.load_pipeline(variant)
    ranges = cfg["external"]["inconclusive"]["ranges"]
    long_side, margin = cfg["external"]["normalization"]["long_side"], cfg["own"]["roi"]["margin"]

    photo_tiles, roi_tiles, contributions = {}, [], []
    for file, rows in pred.groupby("file", sort=True):
        image = ex.read_bgr(args.images_dir / file)
        shown = image.copy()
        thick = max(3, round(max(image.shape[:2]) / 200))
        for row in rows.to_dict("records"):
            roi = inf.parse_roi(row["roi"]) if row["roi"] else None
            analysis = inf.segment_and_extract(
                image, roi, cfg["seg"], cfg["feat"], long_side, margin
            )
            res = inf.predict_variant(analysis, pipeline, variant["group"], ranges, top=10**3)
            stored = float(row["score"]) if row["score"] != "" else None
            if res["inconclusive"] != row["inconclusive"] or (
                stored is not None and not np.isclose(res["score"], stored)
            ):
                raise SystemExit(f"Error: {file} {row['leaflet_id']} no reproduce la predicción")
            label = outcome(row)
            if roi is None:
                auto = inf.overlay(analysis["image"], analysis["mask"], analysis["spots"])
                auto_text = f"auto: {label}"
                continue
            x, y, w, h = roi
            cv2.rectangle(shown, (x, y), (x + w, y + h), COLORS[label], thick)
            crop = cv2.cvtColor(
                inf.overlay(analysis["image"], analysis["mask"], analysis["spots"]),
                cv2.COLOR_RGB2BGR,
            )
            score = f"{res['score']:+.1f}" if res["score"] is not None else ""
            roi_tiles.append(tile(crop, f"{file[:-4]} {row['leaflet_id']} {label} {score}"))
            if row["label"] == "healthy" and not res["inconclusive"]:
                contributions += [
                    {"file": file, "leaflet_id": row["leaflet_id"]} | c
                    for c in res["top_contributions"]
                ]
        photo_tiles[file] = np.hstack(
            [
                tile(shown, f"{file[:-4]} ROI (verde ok, rojo FP, gris NC)"),
                tile(cv2.cvtColor(auto, cv2.COLOR_RGB2BGR), auto_text),
            ]
        )

    table = pd.DataFrame(contributions)
    mean = (
        table.groupby("feature")
        .agg(mean_contribution=("contribution", "mean"), mean_z=("z", "mean"), n=("z", "size"))
        .sort_values("mean_contribution", key=abs, ascending=False)
    )
    mean.to_csv(args.dir / "contributions_B_roi.csv", float_format="%.4f")
    args.grid_dir.mkdir(parents=True, exist_ok=True)
    for bg in cfg["own"]["backgrounds"]:
        files = sorted(set(pred.loc[pred["background"] == bg, "file"]))
        cv2.imwrite(
            str(args.grid_dir / f"grid_photos_{bg}.jpg"), grid([photo_tiles[f] for f in files], 2)
        )
    cv2.imwrite(str(args.grid_dir / "grid_rois.jpg"), grid(roi_tiles, 10))
    print(mean.head(10).round(3).to_string())
    print(f"Rejillas en {args.grid_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
