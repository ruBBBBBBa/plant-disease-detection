"""Evaluación única de un test externo (PlantDoc, T06; Taiwán, T08): predicciones trazables.

Normaliza (lado mayor 256 px), segmenta (T03), extrae características (T04) y predice con los
modelos ``exp2`` (T05) las imágenes ``external_test`` de ``data/metadata.csv``. Las imágenes
«no concluyentes» (D021, umbrales de PlantVillage en ``configs/external.json``) no se predicen:
cuentan como error en la lectura «todas». Se ejecuta **una sola vez**: la guarda se niega si ya
existe cualquier ``execution_*.json`` en ``--out-dir``. No calcula métricas (eso es
``src/external.py`` sobre las predicciones guardadas).

Con ``--source taiwan`` (T08) usa ``data/taiwan_metadata.csv`` y solo las variantes de
``configs/taiwan.json`` (A, B y B′): B′ ve las características con gray world, con la misma
máscara y el mismo «no concluyente» que B. Exige que todas las imágenes estén anotadas
(``data/annotations/taiwan_conditions.csv``) antes de reservar la ejecución.

Ejemplo:
    python scripts/eval_external.py --frozen-commit $(git rev-parse HEAD)
    python scripts/eval_external.py --source taiwan --frozen-commit $(git rev-parse HEAD)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_test  # noqa: E402  (guarda de T05: commit exacto, árbol limpio, registro exclusivo)

from src import external as ex  # noqa: E402
from src import training as tr  # noqa: E402

ID_COLUMNS = ["image_id", "image_path", "disease_name", "binary_label"]


def reserve_external(commit: str, root: Path) -> Path:
    """Guarda de T05 y una norma más: un test externo se usa una vez, con cualquier commit."""
    if any(root.glob("execution_*.json")):
        raise SystemExit(f"Error: este test externo ya se ejecutó ({root}/execution_*.json)")
    return eval_test.reserve_execution(commit, root)


def save_predictions(table, group, pipeline_path, inconclusive, target) -> tuple[str, str]:
    """Predice con un pipeline congelado, guarda todas las filas y devuelve sus dos sha1.

    Las no concluyentes y los vectores no finitos no se predicen: cuentan como error.
    """
    X, y, _, _ = tr.prepare_xy(table, group, False)
    skip = inconclusive | ~np.isfinite(X).all(axis=1)
    pred, score = tr.predict_with_empty(joblib.load(pipeline_path), X, y, skip)
    out = table[[*ID_COLUMNS, "mask_empty", "read_error"]].copy()
    out["y_true"], out["inconclusive"] = y, skip
    out["y_pred"], out["score"] = pred, score
    out.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
    return tr.sha1_file(target), tr.sha1_file(pipeline_path)


def check_annotations(rows: pd.DataFrame, path: Path) -> None:
    """Taiwán: todas las imágenes deben estar anotadas (y versionadas) antes de ejecutar."""
    if not path.is_file():
        raise SystemExit(f"Error: falta {path}; anota con scripts/annotate_conditions.py")
    missing = set(rows["image_path"]) - set(pd.read_csv(path, keep_default_na=False)["file"])
    if missing:
        raise SystemExit(f"Error: {len(missing)} imágenes sin anotar en {path}")


def main(argv: list[str] | None = None) -> int:
    """Procesa PlantDoc una vez y guarda características y predicciones por combinación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--source", choices=["plantdoc", "taiwan"], default="plantdoc")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/external.json")
    parser.add_argument("--taiwan-config", type=Path, default=REPO_ROOT / "configs/taiwan.json")
    parser.add_argument(
        "--annotations", type=Path, default=REPO_ROOT / "data/annotations/taiwan_conditions.csv"
    )
    parser.add_argument("--metadata", type=Path, help="por defecto, el manifiesto de la fuente")
    parser.add_argument(
        "--exp-dir", type=Path, default=REPO_ROOT / "experiments/classification/exp2"
    )
    parser.add_argument("--out-dir", type=Path, help="por defecto experiments/external/<fuente>_v1")
    parser.add_argument("--features-out", type=Path, default=REPO_ROOT / "data/processed")
    args = parser.parse_args(argv)
    taiwan = args.source == "taiwan"
    if args.metadata is None:
        args.metadata = REPO_ROOT / ("data/taiwan_metadata.csv" if taiwan else "data/metadata.csv")
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / f"experiments/external/{args.source}_v1"
    config = json.loads(args.config.read_text())
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    feat_config = json.loads((REPO_ROOT / "configs/features.json").read_text())
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    rows = meta[meta["split"] == "external_test"].sort_values("image_path").reset_index(drop=True)
    if taiwan:
        check_annotations(rows, args.annotations)

    record = reserve_external(args.frozen_commit, args.out_dir)  # a partir de aquí, sin repetir
    if taiwan:
        return run_taiwan(rows, record, args, config, seg_config, feat_config)

    norm, sigma = config["normalization"], config["blur_sigma"]
    plain, blurred = [], []
    for row in rows.itertuples(index=False):
        feats, feats_blur = ex.extract_image(
            REPO_ROOT / row.image_path, seg_config, feat_config, sigma, norm["long_side"]
        )
        ids = {c: getattr(row, c) for c in ID_COLUMNS}
        plain.append(ids | feats)
        blurred.append(ids | feats_blur)
    tables = {"plain": pd.DataFrame(plain), "blur": pd.DataFrame(blurred)}
    args.features_out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.features_out / f"features_plantdoc_v1_{name}.csv", index=False)

    ranges = config["inconclusive"]["ranges"]
    inconclusive = ex.inconclusive_mask(tables["plain"], ranges)
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({p.name.split("__")[1] for p in args.exp_dir.glob("val__*__*.csv.gz")})
    pipelines, sha = {}, {}
    for group in groups:
        table = tables["blur" if group == "textura_suavizada" else "plain"]
        for model in tr.MODEL_ORDER:
            path = REPO_ROOT / "models" / args.exp_dir.name / f"{group}__{model}.joblib"
            target = pred_dir / f"pred__{group}__{model}.csv.gz"
            sha[target.name], pipelines[path.name] = save_predictions(
                table, group, path, inconclusive, target
            )

    payload = json.loads(record.read_text())
    payload.update(
        status="complete",
        n_images=len(rows),
        n_inconclusive=int(inconclusive.sum()),
        config_sha1=tr.sha1_file(args.config),
        predictions_sha1=sha,
        pipelines_sha1=pipelines,
    )
    record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(rows)} imágenes, {int(inconclusive.sum())} no concluyentes → {pred_dir}")
    return 0


def run_taiwan(rows, record, args, config, seg_config, feat_config) -> int:
    """Taiwán (T08): características sin y con gray world y predicciones de A, B y B′."""
    protocol = json.loads(args.taiwan_config.read_text())
    tables = {}
    args.features_out.mkdir(parents=True, exist_ok=True)
    for name, gray in (("plain", False), ("grayworld", True)):
        out = []
        for row in rows.itertuples(index=False):
            feats, _ = ex.extract_image(
                REPO_ROOT / row.image_path,
                seg_config,
                feat_config,
                config["blur_sigma"],
                config["normalization"]["long_side"],
                gray_world=gray,
            )
            out.append({c: getattr(row, c) for c in ID_COLUMNS} | feats)
        tables[name] = pd.DataFrame(out)
        tables[name].to_csv(args.features_out / f"features_taiwan_v1_{name}.csv", index=False)

    ranges = config["inconclusive"]["ranges"]
    inconclusive = ex.inconclusive_mask(tables["plain"], ranges)
    # B y B′ comparten máscara y calidad (D052): el «no concluyente» tiene que coincidir.
    assert (inconclusive == ex.inconclusive_mask(tables["grayworld"], ranges)).all()
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pipelines, sha = {}, {}
    for name, v in protocol["variants"].items():
        table = tables["grayworld" if v.get("gray_world") else "plain"]
        path = REPO_ROOT / "models" / v["exp"] / f"{v['group']}__{v['model']}.joblib"
        target = pred_dir / f"pred__{name}.csv.gz"
        sha[target.name], pipelines[f"{v['exp']}/{path.name}"] = save_predictions(
            table, v["group"], path, inconclusive, target
        )

    payload = json.loads(record.read_text())
    payload.update(
        status="complete",
        source="taiwan",
        n_images=len(rows),
        n_inconclusive=int(inconclusive.sum()),
        config_sha1=tr.sha1_file(args.config),
        taiwan_config_sha1=tr.sha1_file(args.taiwan_config),
        annotations_sha1=tr.sha1_file(args.annotations),
        predictions_sha1=sha,
        pipelines_sha1=pipelines,
    )
    record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(rows)} imágenes, {int(inconclusive.sum())} no concluyentes → {pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
