# ruff: noqa: E501  (filas de tablas markdown largas)
"""Métricas e hipótesis H7–H9 de Taiwán (T08), sin volver a predecir.

Lee las predicciones guardadas por ``eval_external.py --source taiwan``, las condiciones anotadas
(``data/annotations/taiwan_conditions.csv``) y el manifiesto, y escribe ``metrics.json`` y
``tables.md`` en ``experiments/external/taiwan_v1/``. El protocolo (variantes, hipótesis, regla
de < 20 imágenes) está en ``configs/taiwan.json``; las métricas, en ``src/external.py``.

- **H7** (B, solo enfermas): cobertura de ``leaflet + simple`` > resto (bootstrap no emparejado).
- **H8** (B, solo enfermas): recall de enfermas, lectura «todas», ``leaflet + simple`` > resto.
- **H9**: BA «todas» de B′ > B en todo Taiwán (bootstrap emparejado: mismas imágenes y máscara).

Ejemplo:
    python scripts/report_taiwan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402

CONDITION = ["leaves", "background"]


def fmt(value: float, ci: list[float] | None = None) -> str:
    """Número con tres decimales y, si hay, su IC 95 %; «—» si la métrica no existe (NaN)."""
    if np.isnan(value):
        return "—"
    return f"{value:.3f}" if ci is None else f"{value:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]"


def load_predictions(out_dir: Path, variants: dict, annotations: Path, meta: Path) -> dict:
    """Predicciones de cada variante con su condición anotada y ``seen_in_training``."""
    cond = pd.read_csv(annotations, keep_default_na=False).rename(columns={"file": "image_path"})
    seen = pd.read_csv(meta, keep_default_na=False)[["image_path", "seen_in_training"]]
    out = {}
    for name in variants:
        pred = pd.read_csv(out_dir / "predictions" / f"pred__{name}.csv.gz")
        merged = pred.merge(cond, on="image_path", validate="1:1").merge(
            seen, on="image_path", validate="1:1"
        )
        if len(merged) != len(pred):
            raise SystemExit(f"Error: {name}: hay predicciones sin condición anotada")
        out[name] = merged
    return out


def in_condition(pred: pd.DataFrame, condition: dict) -> pd.Series:
    """Filas de una condición ``{"leaves": ..., "background": ...}``."""
    return (pred["leaves"] == condition["leaves"]) & (pred["background"] == condition["background"])


def group_vectors(pred: pd.DataFrame, focus: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectores 0/1 por imagen enferma de la condición foco y del resto (H7, H8).

    ``coverage``: 1 si la imagen es concluyente. ``recall_diseased_all``: 1 si se predice
    enferma; las no concluyentes ya cuentan como error en ``y_pred`` (lectura «todas»).
    """
    sick = pred[pred["y_true"] == 1]
    hit = ~sick["inconclusive"] if metric == "coverage" else sick["y_pred"] == 1
    mask = in_condition(sick, focus)
    return hit[mask].to_numpy(int), hit[~mask].to_numpy(int)


def cell_metrics(cell: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Cobertura, BA y recalls de una celda (lectura «todas»); NaN si falta la clase necesaria.

    Sin sanas (p. ej. todo el fondo ``simple`` de Taiwán) no existen la BA ni el recall de sanas.
    """
    m = ex.reading_metrics(cell, "all", with_ci=False)
    y, p = cell["y_true"].to_numpy(), cell["y_pred"].to_numpy()
    n_h, n_d = int((y == 0).sum()), int((y == 1).sum())
    both = n_h > 0 and n_d > 0
    nan2 = [np.nan, np.nan]
    return {
        "n": len(cell),
        "n_healthy": n_h,
        "n_diseased": n_d,
        "coverage": m["coverage"],
        "balanced_accuracy": m["balanced_accuracy"] if both else np.nan,
        "recall_healthy": m["recall_healthy"] if n_h else np.nan,
        "recall_diseased": m["recall_diseased"] if n_d else np.nan,
        "ci95": {
            "balanced_accuracy": ex.bootstrap_ci(
                lambda i: ex.balanced_accuracy(y[i], p[i]), len(y), n_boot, seed
            )
            if both
            else nan2,
            "recall_diseased": ex.bootstrap_ci(
                lambda i: float((p[i][y[i] == 1] == 1).mean()), len(y), n_boot, seed
            )
            if n_d
            else nan2,
        },
    }


def condition_rows(pred: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    """Métricas de una variante en cada celda ``leaves × background`` (lectura «todas»)."""
    return [
        {"leaves": leaves, "background": background} | cell_metrics(cell, n_boot, seed)
        for (leaves, background), cell in pred.groupby(CONDITION)
    ]


def main(argv: list[str] | None = None) -> int:
    """Calcula las tablas y los veredictos a partir de las predicciones guardadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "experiments/external/taiwan_v1"
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/taiwan.json")
    parser.add_argument(
        "--annotations", type=Path, default=REPO_ROOT / "data/annotations/taiwan_conditions.csv"
    )
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/taiwan_metadata.csv")
    parser.add_argument(
        "--plantdoc", type=Path, default=REPO_ROOT / "experiments/external/plantdoc_v1/metrics.json"
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    boot, min_n = config["bootstrap"], config["min_class_n"]
    focus, extra = config["focus_condition"], config["descriptive_condition"]
    preds = load_predictions(args.out_dir, config["variants"], args.annotations, args.metadata)
    result: dict = {"variants": {}, "hypotheses": {}}

    # ------------------------------------------------------------------ total y vistas / no vistas
    lines = ["## Variantes en todo Taiwán", ""]
    lines += [
        "| Variante | Subconjunto | Lectura | n | Cobertura | BA [IC95] | Recall sanas [IC95] | Recall enfermas [IC95] | Matriz [[TN, FP], [FN, TP]] |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for name, v in config["variants"].items():
        pred = preds[name]
        val = pd.read_csv(
            REPO_ROOT
            / "experiments/classification"
            / v["exp"]
            / f"val__{v['group']}__{v['model']}.csv.gz"
        )
        entry = {"val_ba": ex.val_balanced_accuracy(val)}
        subsets = {"todas": pred, "vistas (sanas + vistas)": pred[pred["seen_in_training"]]}
        for label, rows in subsets.items():
            for reading in ex.READINGS:
                m = ex.reading_metrics(rows, reading, boot["n_boot_descriptive"], boot["seed"])
                entry[f"{label}/{reading}"] = m
                ci = m["ci95"]
                lines.append(
                    f"| {name} | {label} | {reading} | {m['n']} | {m['coverage']:.1%} | "
                    f"{fmt(m['balanced_accuracy'], ci['balanced_accuracy'])} | "
                    f"{fmt(m['recall_healthy'], ci['recall_healthy'])} | "
                    f"{fmt(m['recall_diseased'], ci['recall_diseased'])} | {m['confusion_matrix']} |"
                )
        for reading in ex.READINGS:
            entry[f"by_disease/{reading}"] = ex.recall_by_disease(pred, reading).to_dict("records")
        entry["by_condition"] = condition_rows(pred, boot["n_boot_descriptive"], boot["seed"])
        result["variants"][name] = entry

    lines += ["", "## Recall por enfermedad (no vistas: black_mold, gray_spot, powdery_mildew)", ""]
    lines += [
        "| Variante | Enfermedad | Vista | n | Recall «todas» | Recall «solo concluyentes» (n) |"
    ]
    lines += ["|---|---|---|---:|---:|---|"]
    for name, pred in preds.items():
        seen = pred.groupby("disease_name")["seen_in_training"].first()
        a = ex.recall_by_disease(pred, "all").set_index("disease_name")
        c = ex.recall_by_disease(pred, "conclusive").set_index("disease_name")
        for disease, r in a.iterrows():
            cr = c.loc[disease] if disease in c.index else {"recall": np.nan, "n": 0}
            lines.append(
                f"| {name} | {disease} | {'sí' if seen[disease] else 'no'} | {int(r['n'])} | "
                f"{r['recall']:.3f} | {cr['recall']:.3f} ({int(cr['n'])}) |"
            )

    # ------------------------------------------------------------------ por condición
    lines += ["", "## Por condición `leaves × background` (lectura «todas»)", ""]
    lines += [
        "| Variante | Hojas | Fondo | n (sanas / enfermas) | Cobertura | BA [IC95] | Recall sanas | Recall enfermas [IC95] |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for name, entry in result["variants"].items():
        for r in entry["by_condition"]:
            ci = r["ci95"]
            lines.append(
                f"| {name} | {r['leaves']} | {r['background']} | {r['n']} ({r['n_healthy']} / {r['n_diseased']}) | "
                f"{r['coverage']:.1%} | {fmt(r['balanced_accuracy'], ci['balanced_accuracy'])} | "
                f"{fmt(r['recall_healthy'])} | {fmt(r['recall_diseased'], ci['recall_diseased'])} |"
            )

    # ------------------------------------------------------------------ hipótesis
    lines += ["", "## Hipótesis (fijadas antes de ejecutar, `configs/taiwan.json`)", ""]
    lines += [
        "| H | Contraste | Grupos (n) | Diferencia [IC95] | Veredicto |",
        "|---|---|---|---|---|",
    ]
    for h in ("H7", "H8"):
        spec = config["hypotheses"][h]
        a, b = group_vectors(preds[spec["variant"]], focus, spec["metric"])
        r = ex.size_rule(
            ex.unpaired_bootstrap_diff(a, b, boot["n_boot"], boot["seed"]),
            {"foco": len(a), "resto": len(b)},
            min_n,
        )
        r |= {
            "n_focus": len(a),
            "n_rest": len(b),
            "focus_mean": float(a.mean()),
            "rest_mean": float(b.mean()),
        }
        result["hypotheses"][h] = spec | r
        lines.append(
            f"| {h} | {spec['metric']} de {spec['variant']} (enfermas): leaflet + simple − resto | "
            f"{len(a)} / {len(b)} ({r['focus_mean']:.3f} / {r['rest_mean']:.3f}) | {fmt(r['diff'], r['ci95'])} | {r['verdict']} |"
        )
    spec = config["hypotheses"]["H9"]
    y = preds["B"]["y_true"].to_numpy()
    p_new, p_old = (
        preds[spec["better"]]["y_pred"].to_numpy(),
        preds[spec["worse"]]["y_pred"].to_numpy(),
    )
    if not (
        preds[spec["better"]]["image_path"].to_numpy()
        == preds[spec["worse"]]["image_path"].to_numpy()
    ).all():
        raise SystemExit("Error: B y B′ no tienen las mismas imágenes en el mismo orden")
    r = ex.paired_bootstrap_diff(y, p_new, p_old, boot["n_boot"], boot["seed"])
    r = ex.size_rule(
        r | {"verdict": ex.verdict(r["ci95"])},
        {"sanas": int((y == 0).sum()), "enfermas": int((y == 1).sum())},
        min_n,
    )
    result["hypotheses"]["H9"] = spec | r
    lines.append(
        f"| H9 | BA «todas»: B′ − B (emparejado) | {int((y == 0).sum())} sanas / {int((y == 1).sum())} enfermas | "
        f"{fmt(r['diff'], r['ci95'])} | {r['verdict']} |"
    )

    # ------------------------------------------------------------------ descriptivo
    healthy = np.flatnonzero(y == 0)
    rec = lambda p: lambda i: float((p[healthy[i]] == 0).mean())  # noqa: E731
    diff_h = float((p_new[healthy] == 0).mean() - (p_old[healthy] == 0).mean())
    ci_h = ex.bootstrap_ci(
        lambda i: rec(p_new)(i) - rec(p_old)(i),
        len(healthy),
        boot["n_boot_descriptive"],
        boot["seed"],
    )
    result["descriptive"] = {"H9_recall_healthy_diff": {"diff": diff_h, "ci95": ci_h}}
    lines += [
        "",
        "## Descriptivo, sin veredicto",
        "",
        f"- Recall de sanas, B′ − B (lectura «todas», emparejado): {fmt(diff_h, ci_h)}.",
    ]
    for label, cond in (("compound + simple", extra),):
        sub = preds["B"][in_condition(preds["B"], cond) & (preds["B"]["y_true"] == 1)]
        cov, recd = float((~sub["inconclusive"]).mean()), float((sub["y_pred"] == 1).mean())
        result["descriptive"]["compound_simple_diseased"] = {
            "n": len(sub),
            "coverage": cov,
            "recall_diseased_all": recd,
        }
        lines.append(
            f"- `{label}` (B, enfermas, n = {len(sub)}): cobertura {cov:.3f}, recall de enfermas «todas» {recd:.3f}."
        )
    lines += [
        "",
        "H7 y H8 por enfermedad (B, enfermas; la mezcla cambia entre condiciones, D054):",
        "",
    ]
    lines += [
        "| Enfermedad | n foco / resto | Cobertura foco / resto | Recall «todas» foco / resto |",
        "|---|---|---|---|",
    ]
    sick = preds["B"][preds["B"]["y_true"] == 1].assign(focus=lambda d: in_condition(d, focus))
    by = sick.assign(conc=~sick["inconclusive"], hit=sick["y_pred"] == 1).groupby(
        ["disease_name", "focus"]
    )
    agg = by.agg(n=("hit", "size"), coverage=("conc", "mean"), recall=("hit", "mean")).unstack(
        "focus"
    )
    agg = agg.reindex(
        columns=pd.MultiIndex.from_product([["n", "coverage", "recall"], [True, False]])
    )
    result["descriptive"]["H7_H8_by_disease"] = json.loads(agg.to_json())
    for disease, r in agg.iterrows():
        lines.append(
            f"| {disease} | {r[('n', True)]:.0f} / {r[('n', False)]:.0f} | "
            f"{r[('coverage', True)]:.3f} / {r[('coverage', False)]:.3f} | {r[('recall', True)]:.3f} / {r[('recall', False)]:.3f} |"
        )
    lines += ["", "H9 por condición (BA «todas», B′ − B; sin IC):", ""]
    lines += [
        "| Hojas | Fondo | n | BA B | BA B′ | Recall sanas B / B′ | Recall enfermas B / B′ |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for old, new in zip(
        result["variants"]["B"]["by_condition"],
        result["variants"]["B_prime"]["by_condition"],
        strict=True,
    ):
        lines.append(
            f"| {old['leaves']} | {old['background']} | {old['n']} | "
            f"{fmt(old['balanced_accuracy'])} | {fmt(new['balanced_accuracy'])} | "
            f"{fmt(old['recall_healthy'])} / {fmt(new['recall_healthy'])} | "
            f"{fmt(old['recall_diseased'])} / {fmt(new['recall_diseased'])} |"
        )
    plantdoc = json.loads(args.plantdoc.read_text())["variants"]
    lines += ["", "Comparación con PlantDoc (T06; otro dataset, otras clases):", ""]
    lines += [
        "| Variante | Fuente | Cobertura | BA «todas» | BA «solo concluyentes» |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ("A", "B"):
        for source, e in (("PlantDoc", plantdoc[name]), ("Taiwán", result["variants"][name])):
            a = e["all"] if source == "PlantDoc" else e["todas/all"]
            c = e["conclusive"] if source == "PlantDoc" else e["todas/conclusive"]
            lines.append(
                f"| {name} | {source} | {a['coverage']:.1%} | {a['balanced_accuracy']:.3f} | {c['balanced_accuracy']:.3f} |"
            )
    lines += [
        "",
        "BA de val de PlantVillage (referencia): "
        + ", ".join(f"{n} {e['val_ba']:.3f}" for n, e in result["variants"].items())
        + ".",
        "",
    ]
    (args.out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
