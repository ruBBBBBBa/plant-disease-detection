# ruff: noqa: E501  (filas de tablas markdown largas)
"""Métricas, hipótesis y diagnóstico de dominio de PlantDoc (T06), sin volver a predecir.

Lee las predicciones guardadas por ``eval_external.py`` y escribe ``metrics.json``,
``tables.md`` y ``domain_shift.png`` en ``experiments/external/plantdoc_v1/``. Toda la lógica de
métricas e hipótesis está en ``src/external.py`` (congelada antes de ejecutar).

Ejemplo:
    python scripts/report_external.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402

DOMAIN_FEATURES = [
    "sharpness_laplacian_var",
    "l_std",
    "v_std",
    "glcm_contrast",
    "leaf_area_frac",
    "border_touch_frac",
]


def fmt(value: float, ci: list[float] | None = None) -> str:
    """Número con tres decimales y, si hay, su IC 95 %."""
    return f"{value:.3f}" if ci is None else f"{value:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]"


def main(argv: list[str] | None = None) -> int:
    """Calcula todas las tablas a partir de las predicciones guardadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "experiments/external/plantdoc_v1"
    )
    parser.add_argument(
        "--exp-dir", type=Path, default=REPO_ROOT / "experiments/classification/exp2"
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/external.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    boot = config["bootstrap"]
    preds = {
        tuple(p.stem.removesuffix(".csv").split("__")[1:]): pd.read_csv(p)
        for p in sorted((args.out_dir / "predictions").glob("pred__*.csv.gz"))
    }
    result: dict = {"variants": {}, "appendix": {}, "hypotheses": {}}
    lines = ["## Variantes preregistradas", ""]
    lines += [
        "| Variante | Lectura | n | Cobertura | BA [IC95] | Recall sanas | Recall enfermas | F1 sanas | F1 enfermas | ROC-AUC [IC95] |"
    ]
    lines += ["|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    states = {}
    for name, v in config["variants"].items():
        pred = preds[(v["group"], v["model"])]
        val = pd.read_csv(args.exp_dir / f"val__{v['group']}__{v['model']}.csv.gz")
        val_ba = ex.val_balanced_accuracy(val)
        y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
        states[name] = {"val_ba": val_ba, "y_true": y, "y_pred": p}
        entry = {"val_ba": val_ba, **ex.delta(val_ba, y, p, boot["n_boot"], boot["seed"])}
        for reading in ex.READINGS:
            m = ex.reading_metrics(pred, reading, boot["n_boot_descriptive"], boot["seed"])
            entry[reading] = m
            ci = m["ci95"]
            lines.append(
                f"| {name} | {reading} | {m['n']} | {m['coverage']:.1%} | "
                f"{fmt(m['balanced_accuracy'], ci['balanced_accuracy'])} | "
                f"{fmt(m['recall_healthy'], ci['recall_healthy'])} | "
                f"{fmt(m['recall_diseased'], ci['recall_diseased'])} | "
                f"{m['f1_healthy']:.3f} | {m['f1_diseased']:.3f} | "
                f"{fmt(m.get('roc_auc', float('nan')), ci['roc_auc'])} |"
            )
            entry[f"{reading}_by_disease"] = ex.recall_by_disease(pred, reading).to_dict("records")
        result["variants"][name] = entry
    lines += ["", "## Caídas Δ = BA(val PlantVillage) − BA(PlantDoc, lectura «todas»)", ""]
    lines += ["| Variante | BA val | BA PlantDoc | Δ [IC95] |", "|---|---:|---:|---|"]
    for name, e in result["variants"].items():
        lines.append(
            f"| {name} | {e['val_ba']:.3f} | {e['ba_plantdoc']:.3f} | {fmt(e['delta'], e['ci95'])} |"
        )
    lines += ["", "## Hipótesis (Δ_peor − Δ_mejor > 0; BA de val fija)", ""]
    lines += ["| H | Contraste | Δ_peor − Δ_mejor [IC95] | Veredicto |", "|---|---|---|---|"]
    for h, spec in config["hypotheses"].items():
        r = ex.evaluate_hypothesis(
            states[spec["worse"]], states[spec["better"]], boot["n_boot"], boot["seed"]
        )
        result["hypotheses"][h] = spec | r
        lines.append(
            f"| {h} | {spec['worse']} − {spec['better']} | {fmt(r['diff_delta'], r['ci95'])} | {r['verdict']} |"
        )
    control = result["variants"]["control"]["all"]["ci95"]["balanced_accuracy"]
    result["control_ba_includes_0_5"] = control[0] <= 0.5 <= control[1]
    lines += [
        "",
        f"Control (descriptivo, sin veredicto): BA «todas» = {result['variants']['control']['all']['balanced_accuracy']:.3f} "
        f"IC [{control[0]:.3f}; {control[1]:.3f}]; incluye 0,5: {result['control_ba_includes_0_5']}.",
        "",
    ]

    lines += [
        "## Apéndice descriptivo: todas las combinaciones (BA «todas» / BA «solo concluyentes»)",
        "",
    ]
    lines += [
        "| Grupo | Modelo | BA todas | BA concluyentes | Cobertura |",
        "|---|---|---:|---:|---:|",
    ]
    for (group, model), pred in preds.items():
        a = ex.reading_metrics(pred, "all", with_ci=False)
        c = ex.reading_metrics(pred, "conclusive", with_ci=False)
        result["appendix"][f"{group}__{model}"] = {"all": a, "conclusive": c}
        lines.append(
            f"| {group} | {model} | {a['balanced_accuracy']:.3f} | {c['balanced_accuracy']:.3f} | {a['coverage']:.1%} |"
        )

    base = preds[("combinadas", "svm_lineal")]
    lines += [
        "",
        "## Casos especiales (variante A)",
        "",
        "| Caso | n | No concluyentes | Recall «todas» | Recall «solo concluyentes» |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, rows in ex.special_cases(base).items():
        ok = (rows["y_pred"] == rows["y_true"]).mean()
        done = rows[~rows["inconclusive"]]
        rec = (done["y_pred"] == done["y_true"]).mean() if len(done) else float("nan")
        lines.append(
            f"| {case} | {len(rows)} | {int(rows['inconclusive'].sum())} | {ok:.3f} | {rec:.3f} (n={len(done)}) |"
        )

    plain = pd.read_csv(REPO_ROOT / "data/processed/features_plantdoc_v1_plain.csv")
    pv = pd.read_csv(REPO_ROOT / "data/processed/features_v1_subset.csv")
    pv = pv[pv["split"] == "train"]
    ranges = config["inconclusive"]["ranges"]
    why = {"mask_empty": plain["mask_empty"].mean() * 100}
    for c, (lo, hi) in ranges.items():
        why[c] = ((plain[c] < lo) | (plain[c] > hi))[~plain["mask_empty"]].mean() * 100
    result["inconclusive_reasons_pct"] = why
    lines += ["", "## Motivos de «no concluyente» en PlantDoc (% de las 743; pueden solaparse)", ""]
    lines += ["| Motivo | PlantDoc % | PlantVillage train+val % (referencia) |", "|---|---:|---:|"]
    ref = config["reference_flag_rate_pct"]
    lines += [f"| {k} | {v:.1f} | {ref[k]:.1f} |" for k, v in why.items()]
    lines += [
        f"| cualquiera | {ex.inconclusive_mask(plain, ranges).mean() * 100:.1f} | {ref['any']:.1f} |"
    ]
    lines += [
        "",
        "## Diagnóstico de dominio (mediana; PlantVillage = train)",
        "",
        "| Característica | PlantVillage | PlantDoc | Cociente |",
        "|---|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), DOMAIN_FEATURES, strict=True):
        a, b = pv[col].median(), plain.loc[~plain["mask_empty"], col].median()
        ratio = f"{b / a:.2f}" if a else "—"
        lines.append(f"| {col} | {a:.4g} | {b:.4g} | {ratio} |")
        doc = plain.loc[~plain["mask_empty"], col]
        log = col in ("sharpness_laplacian_var", "glcm_contrast")
        bins = np.logspace(-1, np.log10(max(pv[col].max(), doc.max()) + 1), 40) if log else 40
        for data, label in ((pv[col], "PlantVillage"), (doc, "PlantDoc")):
            ax.hist(
                data.clip(lower=0.1) if log else data,
                bins=bins,
                alpha=0.5,
                density=True,
                label=label,
            )
        ax.set_title(col)
        if log:
            ax.set_xscale("log")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "domain_shift.png", dpi=80)
    (args.out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
