"""Sanidad del modo ROI (T07) y elección del margen de recorte, solo con PlantVillage.

Para cada imagen de ``val`` (subconjunto) se simula la ROI que dibujaría el usuario: la caja
**ajustada** que envuelve la máscara de referencia (``segmented``), igual que la regla de dibujo
«rectángulo pegado al folíolo, sin margen». En PlantVillage la hoja llena casi todo el encuadre,
así que, para que el recorte no choque con el borde, la imagen se amplía **solo en esta
simulación** replicando su borde (``cv2.BORDER_REPLICATE``); la referencia se amplía con fondo.
Para cada margen de recorte de una lista fija se ejecuta el modo ROI de verdad (recorte +
normalización + GrabCut + postprocesado) y se mide:

- el IoU frente a la referencia, en coordenadas de la imagen original, comparado con K-means (T03);
- la ``leaf_area_frac`` del recorte, que es lo que mira la regla de «no concluyente».

Limitaciones: el fondo replicado es una aproximación (crea rayas si hay sombra u hoja en el
borde) y el IoU de esta simulación **no es el rendimiento real** del modo ROI con fotos propias.

Regla de elección (decisión del alumno, fijada antes de procesar fotos propias): el **menor**
margen cuya mediana de ``leaf_area_frac`` en el recorte cae dentro de [p25, p75] de
``leaf_area_frac`` de train+val de PlantVillage. Se escribe en ``configs/own.json`` (``roi``).
No se abren fotos propias.

Ejemplo:
    python scripts/check_roi_mode.py
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
from src import features as ft  # noqa: E402
from src import inference as inf  # noqa: E402
from src import segmentation as seg  # noqa: E402
from src import training as tr  # noqa: E402

MARGINS = [0.10, 0.20, 0.30, 0.40, 0.50]


def bounding_roi(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Caja ajustada ``(x, y, w, h)`` que envuelve la máscara (la ROI simulada)."""
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    return x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1


def pad_for_simulation(image: np.ndarray, ref: np.ndarray, pad: int):
    """Amplía la imagen replicando el borde y la referencia con fondo (solo para simular)."""
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    ref = cv2.copyMakeBorder(ref, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return image, ref


def choose_margin(medians: dict[float, float], low: float, high: float) -> float | None:
    """Menor margen cuya mediana de ``leaf_area_frac`` está en [low, high]; None si ninguno."""
    ok = [m for m in sorted(medians) if low <= medians[m] <= high]
    return ok[0] if ok else None


def main(argv: list[str] | None = None) -> int:
    """Evalúa el modo ROI en val, elige el margen y lo guarda en ``configs/own.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument(
        "--features", type=Path, default=REPO_ROOT / "data/processed/features_v1_subset.csv"
    )
    parser.add_argument("--own-config", type=Path, default=REPO_ROOT / "configs/own.json")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "experiments/segmentation/roi_sanity_val"
    )
    args = parser.parse_args(argv)
    for path in (args.metadata, args.features):
        if not path.is_file():
            raise SystemExit(f"Error: falta {path} (ver data/README.md y T02/T04)")
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    long_side = json.loads((REPO_ROOT / "configs/external.json").read_text())["normalization"][
        "long_side"
    ]
    images = seg.select_images(pd.read_csv(args.metadata, keep_default_na=False), "val", None, 42)
    print(f"val: {len(images)} imágenes, márgenes {MARGINS}")

    rows = []
    for item in images.itertuples(index=False):
        color_path = REPO_ROOT / item.image_path
        image = seg.read_image(color_path)
        ref = seg.reference_mask(
            seg.read_image(seg.segmented_path(color_path)), seg_config["reference_threshold"]
        )
        base = {"image_path": item.image_path, "binary_label": item.binary_label}
        rows.append(
            base
            | {
                "method": "kmeans",
                "margin": np.nan,
                "iou": seg.iou(ft.leaf_mask(image, seg_config), ref),
            }
        )
        pad = int(np.ceil(max(MARGINS) * max(image.shape[:2]))) + 1
        padded, padded_ref = pad_for_simulation(image, ref, pad)
        x, y, w, h = bounding_roi(ref)
        roi = (x + pad, y + pad, w, h)
        for margin in MARGINS:
            crop, box, inner = inf.crop_roi(padded, roi, margin, long_side)
            mask = inf.segment_roi(crop, inner, seg_config)
            full = inf.mask_to_original(mask, box, padded.shape)
            clipped = box != inf.expand_roi(roi, margin, (10**9, 10**9))
            # IoU en la imagen ampliada: lo que entre del fondo replicado cuenta como error.
            rows.append(
                base
                | {
                    "method": "roi",
                    "margin": margin,
                    "iou": seg.iou(full, padded_ref),
                    "clipped": clipped,
                }
                | ft.quality_features(mask)
            )
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        args.out / "per_image.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )

    reference = pd.read_csv(args.features)
    reference = reference[reference["split"].isin(["train", "val"]) & ~reference["mask_empty"]]
    p25, p75 = (float(v) for v in np.percentile(reference["leaf_area_frac"], [25, 75]))
    roi_rows = table[table["method"] == "roi"]
    medians = roi_rows.groupby("margin")["leaf_area_frac"].median().to_dict()
    chosen = choose_margin(medians, p25, p75)
    if chosen is None:
        raise SystemExit(
            f"Error: ningún margen deja la mediana en [{p25:.3f}, {p75:.3f}]: {medians}"
        )

    ranges = json.loads((REPO_ROOT / "configs/external.json").read_text())["inconclusive"]["ranges"]
    flagged = np.array(
        [
            bool(inf.inconclusive_reasons(r | {"mask_empty": False}, ranges))
            for r in roi_rows.to_dict("records")
        ]
    )
    roi_rows = roi_rows.assign(inconclusive=flagged)
    by_margin = roi_rows.groupby("margin").agg(
        median_leaf_area_frac=("leaf_area_frac", "median"),
        inconclusive_rate=("inconclusive", "mean"),
        clipped_rate=("clipped", "mean"),
        median_iou=("iou", "median"),
    )
    iou = table.groupby(["method", "margin", "binary_label"], dropna=False)["iou"].agg(
        ["median", "mean", "count"]
    )
    summary = {
        "split": "val",
        "n_images": len(images),
        "simulated_roi": "caja ajustada de la referencia; imagen ampliada con BORDER_REPLICATE",
        "limitations": [
            "El fondo replicado es una aproximación: puede crear rayas si hay sombra u hoja "
            "en el borde.",
            "El IoU de esta simulación no es el rendimiento real del modo ROI con fotos propias.",
        ],
        "reference_leaf_area_frac": {
            "split": "train+val",
            "p25": p25,
            "p75": p75,
            "features_sha1": tr.sha1_file(args.features),
        },
        "by_margin": {str(k): v for k, v in by_margin.to_dict("index").items()},
        "iou": [
            {"method": m, "margin": None if pd.isna(g) else g, "binary_label": c} | v
            for (m, g, c), v in iou.to_dict("index").items()
        ],
        "chosen_margin": chosen,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    own = json.loads(args.own_config.read_text())
    own["roi"] = {
        "margin": chosen,
        "candidate_margins": MARGINS,
        "rule": "menor margen con mediana de leaf_area_frac del recorte en [p25, p75] de "
        "train+val de PlantVillage (ROI simulada en val: caja ajustada, fondo replicado)",
        "drawing_rule": "rectángulo pegado al folíolo, sin margen: el margen lo añade el recorte",
        "reference_p25_p75": [p25, p75],
        "chosen_with": "experiments/segmentation/roi_sanity_val/summary.json",
    }
    args.own_config.write_text(json.dumps(own, indent=2, ensure_ascii=False) + "\n")
    print(by_margin.to_string())
    print(iou.to_string())
    print(f"[p25, p75] = [{p25:.3f}, {p75:.3f}] → margen elegido: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
