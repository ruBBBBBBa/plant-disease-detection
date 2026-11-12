"""Evalúa los métodos de segmentación frente a la referencia ``segmented`` (T03).

Protocolo: parámetros fijados con ``train`` (``configs/segmentation.json``, generado por
``scripts/tune_segmentation.py``); método por defecto elegido con ``val``; ``test`` una sola vez
al final con todo congelado. Para impedir mirar ``test`` antes de tiempo, ``--split test`` exige
que la configuración ya tenga ``default_method``.

Salidas en ``--out``: ``per_image.csv``, ``summary.md`` y ``overlay_<método>.jpg`` (contorno de la
referencia en amarillo y del método en rojo; casos típicos cerca de la mediana y los 8 peores).

Ejemplo:
    python scripts/evaluate_segmentation.py --split val --out experiments/segmentation/val
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402
from src import segmentation as seg  # noqa: E402

METRICS = ["iou", "dice", "non_green_cov"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluación de la segmentación de la hoja (T03).")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--methods", nargs="+", default=list(seg.METHODS), choices=list(seg.METHODS)
    )
    parser.add_argument(
        "--postprocess",
        choices=["both", "on", "off"],
        default="both",
        help="Evaluar con postprocesado, sin él o ambos (por defecto).",
    )
    parser.add_argument(
        "--sample-per-class", type=int, help="Solo con --split train: muestra fija por enfermedad."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/segmentation.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample_per_class and args.split != "train":
        parser.error("--sample-per-class solo se permite con train (val y test se usan enteros)")
    return args


def cpu_name() -> str:
    """Modelo de CPU (Linux: /proc/cpuinfo) para documentar los tiempos."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "desconocida"


def evaluate_image(image, ref, non_green, method, params, kernel, variants) -> list[dict]:
    """Ejecuta un método sobre una imagen y mide cada variante (sin / con postprocesado)."""
    start = time.perf_counter()
    raw = seg.METHODS[method](image, **params)
    raw_ms = (time.perf_counter() - start) * 1000
    rows = []
    for post in variants:
        mask, ms = raw, raw_ms
        if post:
            start = time.perf_counter()
            mask = seg.postprocess(raw, kernel)
            ms += (time.perf_counter() - start) * 1000
        rows.append(
            {
                "method": method,
                "postprocess": post,
                "iou": seg.iou(mask, ref),
                "dice": seg.dice(mask, ref),
                "non_green_cov": seg.non_green_coverage(mask, non_green),
                "ms": ms,
                "mask": mask,
            }
        )
    return rows


def p10(values: pd.Series) -> float:
    """Percentil 10 (cola de peores casos)."""
    return float(values.quantile(0.10))


def stats_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Media, mediana y percentil 10 de IoU, Dice y cobertura no verde, agrupando por ``by``."""
    table = df.groupby(by)[METRICS].agg(["mean", "median", p10]).round(3)
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.reset_index()


def pivot_metric(df: pd.DataFrame, metric: str, stat: str) -> pd.DataFrame:
    """Tabla enfermedad × variante de método con ``stat`` de ``metric``."""
    variant = df["method"] + np.where(df["postprocess"], "+post", "")
    return (
        df.assign(variant=variant)
        .pivot_table(index="disease_name", columns="variant", values=metric, aggfunc=stat)
        .round(3)
        .reset_index()
    )


def low_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Por variante: nº de imágenes con cobertura no verde < 0,5 y su mediana de borde."""
    low = df[df["non_green_cov"] < 0.5]
    table = low.groupby(["method", "postprocess"]).agg(
        images=("image_id", "size"), median_rim_frac=("non_green_rim_frac", "median")
    )
    return table.round(2).reset_index()


def overlay(image: np.ndarray, ref: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    """Imagen con el contorno de la referencia (amarillo) y del método (rojo), reducida a 128 px."""
    out = image.copy()
    for m, color in ((ref, (0, 255, 255)), (mask, (0, 0, 255))):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 2)
    out = cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.putText(out, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def overlay_grid(cases: list[np.ndarray], cols: int = 8) -> np.ndarray:
    """Pega miniaturas 128×128 en una rejilla de ``cols`` columnas (fondo blanco)."""
    rows = -(-len(cases) // cols)
    grid = np.full((rows * 128, cols * 128, 3), 255, np.uint8)
    for i, tile in enumerate(cases):
        r, c = divmod(i, cols)
        grid[r * 128 : (r + 1) * 128, c * 128 : (c + 1) * 128] = tile
    return grid


def main(argv: list[str] | None = None) -> int:
    """Evalúa los métodos en el split pedido y escribe CSV, resumen y rejillas."""
    args = parse_args(argv)
    if not args.metadata.is_file():
        sys.exit("Error: falta data/metadata.csv. Genéralo con python scripts/prepare_dataset.py.")
    if not args.config.is_file():
        sys.exit("Error: falta la configuración. Genérala con python scripts/tune_segmentation.py.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.split == "test" and not config.get("default_method"):
        sys.exit(
            "Error: test solo se ejecuta con todo congelado "
            "(falta default_method, que se elige con val)."
        )

    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    images = seg.select_images(metadata, args.split, args.sample_per_class, args.seed)
    variants = {"both": [False, True], "on": [True], "off": [False]}[args.postprocess]
    kernel = config["postprocess_kernel"]
    print(f"{args.split}: {len(images)} imágenes, métodos {args.methods}")

    records, masks = [], {}
    for row in images.itertuples(index=False):
        color_path = REPO_ROOT / row.image_path
        image = cv2.imread(str(color_path))
        ref = seg.reference_mask(
            cv2.imread(str(seg.segmented_path(color_path))), config["reference_threshold"]
        )
        non_green = seg.non_green_mask(image, ref, **config["non_green"])
        rim = seg.rim_fraction(non_green, ref)
        for method in args.methods:
            for result in evaluate_image(
                image, ref, non_green, method, config["methods"][method], kernel, variants
            ):
                masks[(row.image_id, method, result["postprocess"])] = result.pop("mask")
                records.append(
                    {
                        "image_id": row.image_id,
                        "image_path": row.image_path,
                        "binary_label": row.binary_label,
                        "disease_name": row.disease_name,
                        "non_green_px": int((non_green > 0).sum()),
                        "non_green_rim_frac": rim,
                        **result,
                    }
                )
    df = pd.DataFrame(records)

    args.out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out / "per_image.csv", index=False)
    write_overlays(df, masks, images, config, args.out)
    write_summary(df, args, config)
    print(f"Resultados en {args.out}")
    return 0


def write_overlays(df, masks, images, config, out: Path) -> None:
    """Una rejilla por método (con postprocesado si se evaluó): 8 típicos y los 8 peores por IoU."""
    paths = dict(zip(images["image_id"], images["image_path"], strict=True))
    for method, group in df.groupby("method"):
        post = group["postprocess"].max()
        group = group[group["postprocess"] == post]
        median = group["iou"].median()
        typical = group.iloc[(group["iou"] - median).abs().argsort()[:8]]
        worst = group.nsmallest(8, "iou")
        tiles = []
        for case in pd.concat([typical, worst]).itertuples(index=False):
            color_path = REPO_ROOT / paths[case.image_id]
            image = cv2.imread(str(color_path))
            ref = seg.reference_mask(
                cv2.imread(str(seg.segmented_path(color_path))), config["reference_threshold"]
            )
            label = f"{case.disease_name[:12]} {case.iou:.2f}"
            tiles.append(overlay(image, ref, masks[(case.image_id, method, post)], label))
        cv2.imwrite(
            str(out / f"overlay_{method}.jpg"), overlay_grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80]
        )


def write_summary(df: pd.DataFrame, args: argparse.Namespace, config: dict) -> None:
    """Escribe ``summary.md`` con estadísticas globales, por sano/enfermo y por enfermedad."""
    sample = f" --sample-per-class {args.sample_per_class}" if args.sample_per_class else ""
    timing = df.groupby(["method", "postprocess"])["ms"].mean().round(1).reset_index()
    lines = [
        f"# Segmentación — evaluación en `{args.split}`",
        "",
        f"`python scripts/evaluate_segmentation.py --split {args.split}{sample} --seed {args.seed} "
        f"--postprocess {args.postprocess}` — {df['image_id'].nunique()} imágenes. Parámetros: "
        "`configs/segmentation.json` (ajustados con train). Referencia: `segmented` con umbral "
        f"{config['reference_threshold']} y relleno de huecos. Cobertura no verde: H fuera de "
        f"[{config['non_green']['h_min']}, {config['non_green']['h_max']}] o S < "
        f"{config['non_green']['s_min']} dentro de la referencia "
        "(NaN si no hay píxeles no verdes).",
        "",
        "`postprocess=True`: apertura + cierre (núcleo "
        f"{config['postprocess_kernel']}), mayor componente conexa y relleno de huecos. "
        "Rejillas `overlay_<método>.jpg`: variante con postprocesado; fila 1, 8 casos cerca de la "
        "mediana de IoU; fila 2, los 8 peores. Amarillo = referencia, rojo = método.",
        "",
        "## Global",
        "",
        audit.md_table(stats_table(df, ["method", "postprocess"])),
        "",
        "## Por `binary_label`",
        "",
        audit.md_table(stats_table(df, ["binary_label", "method", "postprocess"])),
        "",
        "## Por enfermedad — IoU media",
        "",
        audit.md_table(pivot_metric(df, "iou", "mean")),
        "",
        "## Por enfermedad — IoU percentil 10",
        "",
        audit.md_table(pivot_metric(df, "iou", p10)),
        "",
        "## Por enfermedad — cobertura no verde media",
        "",
        audit.md_table(pivot_metric(df, "non_green_cov", "mean")),
        "",
        "## Por enfermedad — cobertura no verde percentil 10",
        "",
        audit.md_table(pivot_metric(df, "non_green_cov", p10)),
        "",
        "## Diagnóstico: cobertura no verde baja",
        "",
        "Imágenes con cobertura no verde < 0,5 y mediana de `non_green_rim_frac` (fracción de sus "
        "píxeles no verdes a ≤ 3 px del borde de la referencia). Si es alta, los píxeles no "
        "cubiertos son sobre todo el halo de fondo/sombra de la referencia, "
        "no lesiones interiores.",
        "",
        audit.md_table(low_coverage_table(df)),
        "",
        f"## Tiempo medio por imagen (ms) — {cpu_name()}, {platform.system()}, un hilo de Python",
        "",
        audit.md_table(timing),
        "",
    ]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
