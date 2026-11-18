"""Evaluación única del test: pipelines congelados y predicciones trazables (T05)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import training as tr  # noqa: E402


def reserve_execution(commit: str, root: Path) -> Path:
    """Exige commit exacto y árbol limpio; reserva una ejecución sin sobrescribir registros."""
    if tr.get_git_commit_hash() != commit:
        raise SystemExit("Error: HEAD no coincide con el commit congelado")
    if not tr.check_git_clean():
        raise SystemExit("Error: hay cambios sin commitear")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"execution_{commit}.json"
    try:
        with path.open("x") as stream:
            json.dump(
                {
                    "commit": commit,
                    "date": datetime.now(timezone.utc).isoformat(),
                    "status": "started",
                },
                stream,
            )
    except FileExistsError:
        raise SystemExit("Error: ya existe un registro de ejecución para este commit") from None
    return path


def main(argv: list[str] | None = None) -> int:
    """Predice test una vez y guarda todas las filas, incluidas máscaras vacías."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument(
        "--exp-dir", type=Path, default=REPO_ROOT / "experiments/classification/exp2"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    # El commit de congelación contiene la configuración; el registro identifica su HEAD exacto.
    record = reserve_execution(args.frozen_commit, args.exp_dir / "test_evaluation")
    manifest = {}
    groups = sorted({p.name.split("__")[1] for p in args.exp_dir.glob("val__*__*.csv.gz")})
    groups = [g for g in groups if g != "color_manchas_sin_l_v"]
    for mode in ("subset", "test_full"):
        for group in groups:
            prefix = "features_v1_blur" if group == "textura_suavizada" else "features_v1"
            data = pd.read_csv(REPO_ROOT / f"data/processed/{prefix}_{mode}.csv")
            data = data[data["split"] == "test"].reset_index(drop=True)
            X, y, _, _ = tr.prepare_xy(data, group, False)
            empty = data["mask_empty"].to_numpy(bool)
            for model in tr.MODEL_ORDER:
                pipeline = joblib.load(
                    REPO_ROOT / "models" / args.exp_dir.name / f"{group}__{model}.joblib"
                )
                pred, score = tr.predict_with_empty(pipeline, X, y, empty)
                out = data[
                    [
                        "image_id",
                        "split",
                        "binary_label",
                        "disease_name",
                        "leaf_group_id",
                        "mask_empty",
                    ]
                ].copy()
                out["y_true"], out["y_pred"], out["score"] = y, pred, score
                path = record.parent / f"test_{mode}__{group}__{model}.csv.gz"
                out.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
                manifest[path.name] = tr.sha1_file(path)
    payload = json.loads(record.read_text())
    payload.update(
        status="complete",
        predictions_sha1=manifest,
        chosen=config,
        config_sha1=tr.sha1_file(args.config),
        pipelines_sha1={
            p.name: tr.sha1_file(p)
            for p in sorted((REPO_ROOT / "models" / args.exp_dir.name).glob("*.joblib"))
        },
    )
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
