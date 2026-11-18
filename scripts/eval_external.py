"""Evaluación única en PlantDoc (T06): pipelines congelados y predicciones trazables.

Normaliza (lado mayor 256 px), segmenta (T03), extrae características (T04) y predice con los
modelos ``exp2`` (T05) las imágenes ``external_test`` de ``data/metadata.csv``. Las imágenes
«no concluyentes» (D021, umbrales de PlantVillage en ``configs/external.json``) no se predicen:
cuentan como error en la lectura «todas». Se ejecuta **una sola vez**: la guarda se niega si ya
existe cualquier ``execution_*.json`` en ``--out-dir``. No calcula métricas (eso es
``src/external.py`` sobre las predicciones guardadas).

Ejemplo:
    python scripts/eval_external.py --frozen-commit $(git rev-parse HEAD)
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
    """Guarda de T05 más una norma propia: PlantDoc solo se usa una vez, con cualquier commit."""
    if any(root.glob("execution_*.json")):
        raise SystemExit("Error: PlantDoc ya se ejecutó (existe un execution_*.json)")
    return eval_test.reserve_execution(commit, root)


def main(argv: list[str] | None = None) -> int:
    """Procesa PlantDoc una vez y guarda características y predicciones por combinación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/external.json")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument(
        "--exp-dir", type=Path, default=REPO_ROOT / "experiments/classification/exp2"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "experiments/external/plantdoc_v1"
    )
    parser.add_argument("--features-out", type=Path, default=REPO_ROOT / "data/processed")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    seg_config = json.loads((REPO_ROOT / "configs/segmentation.json").read_text())
    feat_config = json.loads((REPO_ROOT / "configs/features.json").read_text())
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    rows = meta[meta["split"] == "external_test"].sort_values("image_path").reset_index(drop=True)

    record = reserve_external(args.frozen_commit, args.out_dir)  # a partir de aquí, sin repetir

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
        X, y, _, _ = tr.prepare_xy(table, group, False)
        # Un vector no finito no se puede predecir: se trata como no concluyente.
        skip = inconclusive | ~np.isfinite(X).all(axis=1)
        for model in tr.MODEL_ORDER:
            path = REPO_ROOT / "models" / args.exp_dir.name / f"{group}__{model}.joblib"
            pipeline = joblib.load(path)
            pred, score = tr.predict_with_empty(pipeline, X, y, skip)
            out = table[[*ID_COLUMNS, "mask_empty", "read_error"]].copy()
            out["y_true"], out["inconclusive"] = y, skip
            out["y_pred"], out["score"] = pred, score
            target = pred_dir / f"pred__{group}__{model}.csv.gz"
            out.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
            sha[target.name], pipelines[path.name] = tr.sha1_file(target), tr.sha1_file(path)

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


if __name__ == "__main__":
    raise SystemExit(main())
