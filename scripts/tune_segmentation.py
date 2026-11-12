"""Ajusta los parámetros de segmentación **solo con train** y escribe ``configs/segmentation.json``.

Protocolo de T03: aquí no se toca ``val`` ni ``test``. Sobre una muestra fija de ``train``
(``--sample-per-class`` imágenes por enfermedad, semilla declarada):

1. Umbral de la referencia: valle del histograma de ``max(B,G,R)`` de las imágenes ``segmented``.
2. Canal de Otsu (M2) entre S, a*, L* y ExG; se guardan histogramas hoja/fondo por canal.
3. Rejillas pequeñas para M1 (rangos HSV), M3 (k, regla del borde, Lab o ab), M4 (iteraciones),
   M5 (erosión/dilatación) y el núcleo del postprocesado.

**Criterio de ajuste** (por imagen y luego media): ``score = (IoU + cobertura no verde) / 2``,
o solo IoU si la imagen no tiene píxeles no verdes. Con solo IoU ganan los métodos «de verde»,
porque la mayoría de los píxeles de hoja son verdes, aunque pierdan las lesiones.

La elección del método por defecto **no** se hace aquí: se hace con ``val``
(``scripts/evaluate_segmentation.py``) y se escribe en ``default_method``.

Ejemplo:
    python scripts/tune_segmentation.py --sample-per-class 30 --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402
from src import segmentation as seg  # noqa: E402

CHANNELS = ["S", "a", "L", "ExG"]
NON_GREEN = {"h_min": 30, "h_max": 90, "s_min": 40}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Ajuste de la segmentación con train (T03).")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--sample-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/segmentation.json")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "experiments/segmentation/tuning_train"
    )
    return parser.parse_args(argv)


def reference_threshold(segmented: list[np.ndarray], low: int = 5, high: int = 60) -> int:
    """Valle del histograma de ``max(B,G,R)`` entre ``low`` y ``high`` (suavizado con media 5)."""
    hist = np.zeros(256)
    for image in segmented:
        hist += np.bincount(image.max(axis=2).ravel(), minlength=256)
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    return int(low + np.argmin(smooth[low : high + 1]))


def evaluate(samples: list[dict], method, post_kernel: int | None = None, **params) -> dict:
    """IoU media, cobertura no verde media y ``score`` medio de ``method`` sobre ``samples``."""
    ious, covs, scores = [], [], []
    for sample in samples:
        mask = method(sample["image"], **params)
        if post_kernel is not None:
            mask = seg.postprocess(mask, post_kernel)
        iou = seg.iou(mask, sample["ref"])
        cov = seg.non_green_coverage(mask, sample["non_green"])
        ious.append(iou)
        covs.append(cov)
        scores.append(iou if np.isnan(cov) else (iou + cov) / 2)
    return {
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_non_green_cov": round(float(np.nanmean(covs)), 4),
        "score": round(float(np.mean(scores)), 4),
    }


def grid_search(samples, method, grid: dict[str, list], rows: list, name: str) -> dict:
    """Prueba todas las combinaciones de ``grid`` y devuelve la de mayor ``score`` medio."""
    best, best_score = None, -1.0
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values, strict=True))
        result = evaluate(samples, method, **params)
        rows.append({"method": name, "params": json.dumps(params), **result})
        if result["score"] > best_score:
            best, best_score = params, result["score"]
    return best


def channel_histograms(samples: list[dict], out: Path) -> None:
    """Figura con el histograma de cada canal candidato, hoja frente a fondo (según referencia)."""
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4 * len(CHANNELS), 3))
    for ax, name in zip(axes, CHANNELS, strict=True):
        leaf, back, lesion = np.zeros(256), np.zeros(256), np.zeros(256)
        for sample in samples:
            values = seg.channel(sample["image"], name)
            leaf += np.bincount(values[sample["ref"] > 0], minlength=256)
            back += np.bincount(values[sample["ref"] == 0], minlength=256)
            lesion += np.bincount(values[sample["non_green"] > 0], minlength=256)
        ax.plot(leaf / leaf.sum(), label="hoja")
        ax.plot(back / back.sum(), label="fondo")
        ax.plot(lesion / lesion.sum(), label="hoja no verde", linestyle="--")
        ax.set_title(f"Canal {name}")
        ax.set_xlabel("valor (0–255)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def reference_examples(samples, segmented, kmeans_params: dict, kernel: int, out: Path) -> None:
    """Rejilla de posibles defectos de la referencia en train: color | segmented | referencia.

    Para elegir qué mostrar se compara la referencia con K-means + postprocesado (solo como
    selector): 4 casos donde la referencia es mucho **menor** (recortes, p. ej. necrosis oscura)
    y 4 donde es **mayor** (sombra incluida u otro exceso). La revisión es visual.
    """
    diffs = []
    for sample in samples:
        mask = seg.postprocess(seg.segment_kmeans(sample["image"], **kmeans_params), kernel)
        diffs.append(int((sample["ref"] > 0).sum()) - int((mask > 0).sum()))
    order = np.argsort(diffs)
    picks = list(order[:4]) + list(order[::-1][:4])
    rows = []
    for i in picks:
        ref = cv2.cvtColor(samples[i]["ref"], cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([samples[i]["image"], segmented[i], ref]))
    grid = np.vstack(rows)
    grid = cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "reference_examples.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el ajuste completo con train y escribe la configuración."""
    args = parse_args(argv)
    if not args.metadata.is_file():
        sys.exit("Error: falta data/metadata.csv. Genéralo con python scripts/prepare_dataset.py.")
    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    rows = seg.select_images(metadata, "train", args.sample_per_class, args.seed)
    images = [cv2.imread(str(REPO_ROOT / p)) for p in rows["image_path"]]
    segmented = [cv2.imread(str(seg.segmented_path(REPO_ROOT / p))) for p in rows["image_path"]]
    ref_threshold = reference_threshold(segmented)
    samples = []
    for image, segmented_image in zip(images, segmented, strict=True):
        ref = seg.reference_mask(segmented_image, ref_threshold)
        samples.append(
            {"image": image, "ref": ref, "non_green": seg.non_green_mask(image, ref, **NON_GREEN)}
        )
    print(f"train: {len(rows)} imágenes; umbral de referencia = {ref_threshold}")

    args.out.mkdir(parents=True, exist_ok=True)
    channel_histograms(samples, args.out / "channel_histograms.png")
    # La elección hecha con val (si ya existe) no la decide este script: se conserva.
    previous = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    results: list[dict] = []
    otsu = grid_search(samples, seg.segment_otsu, {"channel_name": CHANNELS}, results, "otsu")
    channel_name = otsu["channel_name"]
    hsv_grid = {
        "h_min": [20, 25, 30, 35],
        "h_max": [85, 95],
        "s_min": [30, 50, 70],
        "v_min": [20, 40],
    }
    kmeans_grid = {
        "k": [2, 3],
        "border_share": [0.1, 0.2, 0.3, 0.4, 0.5],
        "features": ["Lab", "ab"],
        "seed": [args.seed],
    }
    params = {
        "hsv_green": grid_search(samples, seg.segment_hsv_green, hsv_grid, results, "hsv_green"),
        "otsu": otsu,
        "kmeans": grid_search(samples, seg.segment_kmeans, kmeans_grid, results, "kmeans"),
        "grabcut": grid_search(
            samples,
            seg.segment_grabcut,
            {"channel_name": [channel_name], "iterations": [3, 5]},
            results,
            "grabcut",
        ),
        "watershed": grid_search(
            samples,
            seg.segment_watershed,
            {"channel_name": [channel_name], "erode": [5, 9, 15], "dilate": [5, 9, 15]},
            results,
            "watershed",
        ),
    }
    # Núcleo del postprocesado: el de mayor score, promediando todos los métodos.
    kernel_scores = {}
    for kernel in (3, 5, 7):
        per_method = [
            evaluate(samples, seg.METHODS[name], post_kernel=kernel, **p)
            for name, p in params.items()
        ]
        summary = {
            key: round(float(np.mean([r[key] for r in per_method])), 4) for key in per_method[0]
        }
        kernel_scores[kernel] = summary["score"]
        results.append(
            {"method": "postprocess", "params": json.dumps({"kernel": kernel}), **summary}
        )
    kernel = max(kernel_scores, key=kernel_scores.get)

    config = {
        "_comment": "Parámetros ajustados con train (scripts/tune_segmentation.py). "
        "default_* se eligen con val y se conservan al volver a ajustar.",
        "tuned_on": {
            "split": "train",
            "sample_per_class": args.sample_per_class,
            "seed": args.seed,
        },
        "tuning_criterion": "media por imagen de (IoU + cobertura no verde) / 2",
        "reference_threshold": ref_threshold,
        "non_green": NON_GREEN,
        "postprocess_kernel": kernel,
        "methods": params,
        "default_method": previous.get("default_method"),
        "default_postprocess": previous.get("default_postprocess"),
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    reference_examples(samples, segmented, params["kmeans"], kernel, args.out)
    args.config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame(results)
    table.to_csv(args.out / "tuning.csv", index=False)
    best = table.loc[table.groupby("method")["score"].idxmax()]
    (args.out / "summary.md").write_text(
        "# Ajuste de segmentación con train\n\n"
        f"`python scripts/tune_segmentation.py --sample-per-class {args.sample_per_class} "
        f"--seed {args.seed}` — {len(rows)} imágenes de `train` "
        f"({args.sample_per_class} por enfermedad).\n\n"
        f"- Umbral de la referencia (valle del histograma de max(B,G,R)): **{ref_threshold}**.\n"
        f"- Canal de Otsu elegido: **{channel_name}** (histogramas en `channel_histograms.png`).\n"
        f"- Núcleo del postprocesado: **{kernel}**.\n"
        "- Criterio: media por imagen de (IoU + cobertura no verde) / 2 (solo IoU si no hay "
        "píxeles no verdes). Métodos sin postprocesado, salvo las filas `postprocess` "
        "(media de los 5 métodos con postprocesado).\n\n"
        "## Mejor combinación por método\n\n"
        + audit.md_table(best)
        + "\n\n## Todas las combinaciones\n\n"
        + audit.md_table(table)
        + "\n",
        encoding="utf-8",
    )
    print(f"Configuración escrita en {args.config.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
