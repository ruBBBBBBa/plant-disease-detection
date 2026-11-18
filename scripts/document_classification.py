"""Reconcilia Entrega y decisiones desde los artefactos T05, sin métricas manuales."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import training as tr  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Regenera documentación desde resultados guardados sin ajustar ni predecir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    root = parser.parse_args(argv).exp_dir
    with (root / "cv_val_results.pkl").open("rb") as stream:
        results = pickle.load(stream)
    sigma = json.loads((REPO_ROOT / "reports/classification/sigma_selection.json").read_text())
    chosen = {
        "A": tr.choose_candidate(results),
        "B": tr.choose_candidate(results, "color_manchas"),
        "C": tr.choose_candidate(results, "color_manchas_sin_l_v"),
    }
    desc = {
        n: f"{g} + {m}: val BA={r['val_metrics']['balanced_accuracy']:.6f}"
        for n, (g, m, r) in chosen.items()
    }
    evaluation = root / "evaluation.md"
    lines = [
        "## Entrega regenerada — Revisión Codex",
        "",
        "CV/val desde " + (root / "run_commit.txt").read_text().strip(),
        "",
        "Ejecución nueva: experiments/classification/exp2; pipelines: models/exp2.",
        "Comandos desde la raíz (Python del entorno virtual):",
        "",
        "```bash",
        "python scripts/train_classifiers.py cv-val --out experiments/classification/exp2",
        "python scripts/train_classifiers.py choose-best --out experiments/classification/exp2",
        "python scripts/report_classification.py --exp-dir experiments/classification/exp2",
        "python scripts/document_classification.py --exp-dir experiments/classification/exp2",
        "```",
        "",
        "Test: commit exacto y fecha en test_evaluation/execution_<commit>.json.",
        "No repetir test. Las tablas y rejillas usan exclusivamente predicciones guardadas.",
        "Scores no calibrados; no probabilidades. C solo se evalúa en val.",
        "",
        *[f"- {n}: {v}" for n, v in desc.items()],
        "",
        "### Sigma (train, registro conservado)",
        "",
        "```json",
        json.dumps(sigma, indent=2),
        "```",
        "",
        (root / "summary.md").read_text(),
        "",
        evaluation.read_text() if evaluation.exists() else "Test pendiente.",
        "",
        "Exp1 invalidada: bootstrap sin multiplicidad y mask_empty omitida en evaluación.",
        "La segunda ejecución test corrige esos fallos; no cambia modelos ni grids por test.",
        "Las diferencias de sesgo son descriptivas, no pruebas de causalidad.",
        "",
    ]
    task = REPO_ROOT / "docs/tasks/T05-clasificacion-evaluacion.md"
    task.write_text(task.read_text().split("## Entrega", 1)[0] + "\n".join(lines))
    decisions = REPO_ROOT / "docs/DECISIONS.md"
    marker = "### Repetición excepcional de test (T05)"
    old = decisions.read_text().split(marker)[0]
    rows = [
        line
        for line in old.splitlines()
        if not any(line.startswith(f"| D{i:03d} |") for i in range(24, 31))
    ]
    data = {
        24: f"Sigma train: {sigma['chosen_sigma']}; detalles en sigma_selection.json.",
        25: "Máximo val; diferencia <0,005; kNN, lineal, DT, RBF, RF; luego grupo alfabético.",
        26: desc["A"],
        27: "Commit exacto, registro exclusivo y SHA-1; informes solo desde predicciones.",
        28: "Sesgo: comparar mismo estimador entre grupos en evaluation.md; no ajustar con test.",
        29: "mask_empty: fuera de train; error en evaluación y ROC. Bootstrap por hoja repetida.",
        30: "; ".join(f"{n}={v}" for n, v in desc.items()) + ". Elección solo con val.",
    }
    for i, text in data.items():
        rows.append(
            f"| D{i:03d} | 2026-11-16 | {text} | Revisión Codex exp2 | — | Aceptada (T05) |"
        )
    rows += [
        "",
        marker,
        "",
        "La segunda ejecución invalida exp1 por fallos de bootstrap y mask_empty.",
        "Modelos, grids y elección se congelan antes de test y no cambian por sus resultados.",
        "El registro conserva commit, fecha y SHA-1 de cada CSV. C no se evalúa en test.",
        "PlantDoc: hipótesis B menos dependiente de textura; C sin l_std/v_std.",
        "B/C no aíslan causalmente v_std: también difieren el estimador y l_std.",
        "",
    ]
    decisions.write_text("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
