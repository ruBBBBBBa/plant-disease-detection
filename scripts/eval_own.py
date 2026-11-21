"""Evaluación única con las fotos propias (T07): modos automático y ROI, variantes B y A.

Antes de abrir ninguna foto valida las anotaciones (``own_photos.csv``, ``own_rois.csv``) contra los
nombres de ``data/raw/own/`` e informa de cuántas hojas tienen cada combinación de fondos; si algo
falla se para sin reservar la ejecución. ``--check-only`` hace solo esa validación.

Guarda (igual que ``eval_external.py``): commit exacto, árbol limpio y se niega si ya existe
**cualquier** ``execution_*.json`` en ``--out-dir``. Las fotos se usan una sola vez.

Unidades: cada foto en modo automático; cada fila de ``own_rois.csv`` (folíolo) en modo ROI.
Guarda las predicciones por unidad en ``predictions.csv.gz``; las tablas, hipótesis y rejillas se
regeneran desde ahí (``scripts/report_own.py``). No calcula métricas.

Ejemplos:
    python scripts/eval_own.py --check-only
    python scripts/eval_own.py --frozen-commit $(git rev-parse HEAD)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_test  # noqa: E402  (guarda de T05: commit exacto, árbol limpio, registro exclusivo)

from src import external as ex  # noqa: E402
from src import inference as inf  # noqa: E402
from src import own_eval as oe  # noqa: E402
from src import training as tr  # noqa: E402


def reserve_own(commit: str, root: Path) -> Path:
    """Guarda de T05 más la norma de T06: las fotos propias solo se evalúan una vez."""
    if any(root.glob("execution_*.json")):
        raise SystemExit("Error: las fotos propias ya se evaluaron (existe un execution_*.json)")
    return eval_test.reserve_execution(commit, root)


def unit_rows(photo, analysis: dict, leaflet_id: str, pipelines: dict, cfg: dict) -> list[dict]:
    """Una fila por variante para una unidad (foto o folíolo) ya segmentada."""
    ranges = cfg["external"]["inconclusive"]["ranges"]
    feats = analysis["features"]
    rows = []
    for variant, pipeline in pipelines.items():
        group = cfg["external"]["variants"][variant]["group"]
        pred = inf.predict_variant(analysis, pipeline, group, ranges)
        rows.append(
            {
                **{c: getattr(photo, c) for c in oe.PHOTO_COLUMNS},
                "mode": analysis["mode"],
                "leaflet_id": leaflet_id,
                "variant": variant,
                "inconclusive": pred["inconclusive"],
                "reasons": "; ".join(pred["reasons"]),
                "prediction": pred["prediction"] or "",
                "score": pred["score"],
                "roi": ",".join(map(str, analysis["roi"])) if analysis["roi"] else "",
                "mask_empty": feats["mask_empty"],
                **{c: feats.get(c) for c in ex.QUALITY_COLUMNS},
                "spot_area_frac": feats.get("spot_area_frac"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Valida, reserva la ejecución y evalúa cada foto y cada folíolo una vez."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frozen-commit")
    parser.add_argument("--check-only", action="store_true", help="solo validar las anotaciones")
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data/raw/own")
    parser.add_argument(
        "--photos", type=Path, default=REPO_ROOT / "data/annotations/own_photos.csv"
    )
    parser.add_argument("--rois", type=Path, default=REPO_ROOT / "data/annotations/own_rois.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/own/v1")
    args = parser.parse_args(argv)
    for path in (args.images_dir, args.photos, args.rois):
        if not path.exists():
            raise SystemExit(
                f"Error: falta {path} (ver docs/tasks/T07-inferencia-y-fotos-propias.md)"
            )
    cfg = inf.load_configs()
    photos = pd.read_csv(args.photos, keep_default_na=False)
    rois = pd.read_csv(args.rois, keep_default_na=False)
    problems = oe.validate_annotations(photos, rois, args.images_dir, cfg["own"])
    print("Hojas sanas por combinación de fondos:")
    print(oe.background_combinations(photos).to_string())
    if problems:
        print("\n".join(["Problemas en las anotaciones (no se corrige nada a mano):", *problems]))
        return 1
    print(f"Anotaciones correctas: {len(photos)} fotos, {len(rois)} ROI")
    if args.check_only:
        return 0
    if not args.frozen_commit:
        raise SystemExit("Error: falta --frozen-commit")

    record = reserve_own(args.frozen_commit, args.out_dir)  # a partir de aquí, sin repetir
    pipelines = {
        v: inf.load_pipeline(cfg["external"]["variants"][v]) for v in cfg["own"]["variants"]
    }
    long_side, margin = cfg["external"]["normalization"]["long_side"], cfg["own"]["roi"]["margin"]
    rows = []
    for photo in photos.sort_values("file").itertuples(index=False):
        image = ex.read_bgr(args.images_dir / photo.file)
        if image is None:
            raise SystemExit(f"Error: no se puede leer {photo.file}")
        units = [(None, "")] + [
            ((r.x, r.y, r.w, r.h), r.leaflet_id)
            for r in rois[rois["file"] == photo.file].itertuples(index=False)
        ]
        for roi, leaflet_id in units:
            analysis = inf.segment_and_extract(
                image, roi, cfg["seg"], cfg["feat"], long_side, margin
            )
            rows += unit_rows(photo, analysis, leaflet_id, pipelines, cfg)
    target = args.out_dir / "predictions.csv.gz"
    pd.DataFrame(rows).to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})

    payload = json.loads(record.read_text())
    payload.update(
        status="complete",
        n_photos=len(photos),
        n_rois=len(rois),
        photos_sha1=tr.sha1_file(args.photos),
        rois_sha1=tr.sha1_file(args.rois),
        own_config_sha1=tr.sha1_file(REPO_ROOT / "configs/own.json"),
        predictions_sha1=tr.sha1_file(target),
    )
    record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(photos)} fotos, {len(rois)} folíolos → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
