"""Extrae el vector de características de cada imagen y escribe un CSV (T04).

Usa la segmentación congelada de T03 (``configs/segmentation.json``) y la configuración de
características ajustada con train (``configs/features.json``). Calcula, **sin analizarlas**, las
características de ``val`` y ``test``: se usarán en la fase 4.

- ``--subset-only``: imágenes ``in_subset`` de los ``--splits`` (por defecto train val test).
- ``--full-test``: todo el ``test`` de PlantVillage (para el análisis por enfermedad, D014).

El CSV (``data/processed/``, no versionado) no incluye tiempos, para que dos ejecuciones den el
mismo SHA-1. El tiempo medio por imagen y el SHA-1 se anotan en ``reports/features/extraction.md``.

Ejemplos:
    python scripts/extract_features.py --subset-only
    python scripts/extract_features.py --full-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import features as ft  # noqa: E402
from src import segmentation as seg  # noqa: E402

VERSION = "v1"
ID_COLUMNS = ["image_id", "split", "binary_label", "disease_name", "leaf_group_id", "in_subset"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Extracción de características (T04).")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--subset-only", action="store_true", help="Subconjunto (in_subset).")
    which.add_argument("--full-test", action="store_true", help="Todo el test de PlantVillage.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--seg-config", type=Path, default=REPO_ROOT / "configs/segmentation.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/features.json")
    parser.add_argument("--out", type=Path, help="CSV de salida (por defecto en data/processed/).")
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "reports/features/extraction.md")
    args = parser.parse_args(argv)
    name = "subset" if args.subset_only else "test_full"
    args.name = name
    args.out = args.out or REPO_ROOT / f"data/processed/features_{VERSION}_{name}.csv"
    if args.full_test and args.splits != ["train", "val", "test"]:
        parser.error("--splits no se usa con --full-test")
    return args


def select_rows(metadata: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Filas de PlantVillage a procesar según el modo (subconjunto o test completo)."""
    pv = metadata[metadata["source"] == "plantvillage"]
    if args.full_test:
        rows = pv[pv["split"] == "test"]
    else:
        rows = pv[(pv["in_subset"].astype(str) == "True") & pv["split"].isin(args.splits)]
    return rows.sort_values("image_path").reset_index(drop=True)


def cpu_name() -> str:
    """Modelo de CPU (Linux) para documentar los tiempos."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "desconocida"


def update_log(log: Path, name: str, text: str) -> None:
    """Escribe o reemplaza la sección ``## <name>`` del registro de extracción."""
    sections: dict[str, str] = {}
    if log.is_file():
        for block in log.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = block.partition("\n")
            sections[title.strip()] = body.strip()
    sections[name] = text.strip()
    content = "# Extracción de características — registro\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sorted(sections.items())
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(content + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Extrae las características y escribe el CSV y el registro."""
    args = parse_args(argv)
    for path in (args.metadata, args.seg_config, args.config):
        if not path.is_file():
            sys.exit(f"Error: falta {path}. Ver docs/tasks/T02, T03 y T04 para generarlo.")
    seg_config = json.loads(args.seg_config.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = select_rows(pd.read_csv(args.metadata, keep_default_na=False), args)
    if rows.empty:
        sys.exit("Error: no hay imágenes para los splits pedidos.")

    records, elapsed = [], 0.0
    for row in rows.itertuples(index=False):
        image = seg.read_image(REPO_ROOT / row.image_path)
        start = time.perf_counter()
        mask = ft.leaf_mask(image, seg_config)
        feats = ft.extract_features(image, mask, config)
        elapsed += time.perf_counter() - start
        quality = {k: feats[k] for k in ("leaf_area_frac", "border_touch_frac", "mask_solidity")}
        records.append(
            {
                **{column: getattr(row, column) for column in ID_COLUMNS},
                "inconclusive": ft.is_inconclusive(quality, config["inconclusive"]),
                **feats,
            }
        )
    table = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.6g")

    sha1 = hashlib.sha1(args.out.read_bytes()).hexdigest()
    ms = 1000 * elapsed / len(rows)
    n_features = len(table.columns) - len(ID_COLUMNS) - 1
    counts = table["split"].value_counts().sort_index().to_dict()
    flag = "--subset-only" if args.subset_only else "--full-test"
    update_log(
        args.log,
        args.name,
        f"`python scripts/extract_features.py {flag}` → `data/processed/{args.out.name}` "
        f"(no versionado), SHA-1 `{sha1}`.\n\n"
        f"- Imágenes por split: {counts}; características por imagen: {n_features}.\n"
        f"- Tiempo medio por imagen (segmentación + características): {ms:.1f} ms — {cpu_name()}, "
        f"{platform.system()}, un hilo.\n"
        "- Las características de `val` y `test` se calculan pero no se analizan (protocolo T04).",
    )
    print(f"{len(rows)} imágenes, {n_features} características, {ms:.1f} ms/img → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
