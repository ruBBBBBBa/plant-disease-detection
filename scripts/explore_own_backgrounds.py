"""Análisis EXPLORATORIO A POSTERIORI de T07: ¿cambia la salida de B con el fondo? (D050).

No cambia ningún veredicto, umbral ni modelo. Usa solo ``predictions.csv.gz`` guardado en la
ejecución única de T07; no vuelve a procesar las fotos ni ejecuta ``eval_own.py``.

1. **Emparejamiento.** ``leaflet_id`` debía nombrar el mismo folíolo en los tres fondos, pero no se
   puede verificar sin mirar las fotos. Se comprueba solo su coherencia geométrica (mismo número de
   ROI por fondo; correlación de Spearman del área relativa de cada ROI frente a gris oscuro).
   Por folíolo solo se analizan las hojas con **una ROI por foto** (emparejamiento sin
   ambigüedad); por hoja (``leaf_id``) se analizan todas, porque la hoja sí está identificada.
2. **Acuerdo** de la predicción entre fondos y **variación de la puntuación** dentro de la unidad
   (entre fondos) frente a entre unidades, con un ICC(1) de ANOVA de un factor.
3. b* y ``b_hist_07`` por fondo **no se pueden calcular**: ``contributions_B_roi.csv`` solo guarda
   medias sobre todos los folíolos, no valores por folíolo.

Ejemplo:
    python scripts/explore_own_backgrounds.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "gris_oscuro"


def roi_units(pred: pd.DataFrame) -> pd.DataFrame:
    """Folíolos sanos de B en modo ROI, con área relativa de su ROI dentro de la foto."""
    r = pred[(pred["mode"] == "roi") & (pred["variant"] == "B") & (pred["label"] == "healthy")]
    r = r.copy()
    wh = r["roi"].str.split(",", expand=True).astype(int)
    r["area"] = wh[2] * wh[3]
    r["rel_area"] = r["area"] / r.groupby("file")["area"].transform("sum")
    r["outcome"] = np.where(r["inconclusive"], "no concluyente", r["prediction"])
    return r


def pairing_table(units: pd.DataFrame) -> pd.DataFrame:
    """Coherencia geométrica de ``leaflet_id`` entre fondos, por hoja (no prueba identidad)."""
    rows = []
    for leaf, g in units.groupby("leaf_id"):
        counts = g.groupby("background").size()
        wide = g.pivot(index="leaflet_id", columns="background", values="rel_area").dropna()
        rho = {
            bg: spearmanr(wide[bg], wide[REFERENCE]).statistic if len(wide) > 2 else np.nan
            for bg in wide.columns
            if bg != REFERENCE
        }
        rows.append(
            {
                "leaf_id": leaf,
                "rois_por_fondo": "/".join(f"{counts.get(b, 0)}" for b in sorted(counts.index)),
                "rho_blanco": rho.get("blanco", np.nan),
                "rho_natural": rho.get("natural", np.nan),
                "emparejable": bool((counts == 1).all()),
            }
        )
    return pd.DataFrame(rows)


def icc1(wide: pd.DataFrame) -> dict:
    """Desviaciones típicas dentro y entre unidades e ICC(1) (ANOVA de un factor, k fondos).

    ``wide``: una fila por unidad y una columna por fondo (solo unidades completas). ICC próximo
    a 1 = la puntuación depende sobre todo de la unidad; próximo a 0 = del fondo o del ruido.
    """
    wide = wide.dropna()
    n, k = wide.shape
    if n < 2:
        return {"n": n, "sd_within": np.nan, "sd_between": np.nan, "icc1": np.nan}
    means = wide.mean(axis=1)
    msw = ((wide.sub(means, axis=0)) ** 2).to_numpy().sum() / (n * (k - 1))
    msb = k * ((means - means.mean()) ** 2).sum() / (n - 1)
    return {
        "n": n,
        "sd_within": float(np.sqrt(msw)),
        "sd_between": float(means.std(ddof=1)),
        "icc1": float((msb - msw) / (msb + (k - 1) * msw)),
    }


def agreement(outcomes: pd.DataFrame) -> dict:
    """Unidades con el mismo resultado (diseased / healthy / no concluyente) en todos los fondos."""
    outcomes = outcomes.dropna()
    same = outcomes.nunique(axis=1) == 1
    return {"n": len(outcomes), "same": int(same.sum())}


def main(argv: list[str] | None = None) -> int:
    """Escribe ``exploratory_backgrounds.md`` en el directorio de T07."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/own/v1")
    args = parser.parse_args(argv)
    units = roi_units(pd.read_csv(args.dir / "predictions.csv.gz", keep_default_na=False))
    units["score"] = pd.to_numeric(units["score"], errors="coerce")
    pairing = pairing_table(units)

    # Por hoja: fracción de folíolos diseased y puntuación media de los concluyentes.
    by_leaf = units.groupby(["leaf_id", "background"]).agg(
        n=("outcome", "size"),
        diseased=("outcome", lambda s: (s == "diseased").mean()),
        score=("score", "mean"),
    )
    leaf_score = by_leaf["score"].unstack()
    leaf_major = (by_leaf["diseased"] > 0.5).map({True: "diseased", False: "healthy"}).unstack()
    # Por folíolo: solo hojas con una ROI por foto.
    single = units[units["leaf_id"].isin(pairing.loc[pairing["emparejable"], "leaf_id"])]
    leaflet_score = single.pivot(index="leaf_id", columns="background", values="score")
    leaflet_out = single.pivot(index="leaf_id", columns="background", values="outcome")

    fmt = {"float_format": lambda v: f"{v:.2f}"}
    lines = [
        "# T07 — Análisis exploratorio a posteriori: salida de B según el fondo",
        "",
        "> **Exploratorio y a posteriori** (D050): no cambia ningún veredicto (H4–H6 siguen «no",
        "> concluyentes»), umbral ni modelo. Solo usa `predictions.csv.gz` de la ejecución única;",
        "> no se han vuelto a procesar las fotos. Regenerable con",
        "> `python scripts/explore_own_backgrounds.py`.",
        "",
        "## 1. ¿Se pueden emparejar los folíolos entre fondos?",
        "",
        "`leaflet_id` debía nombrar el mismo folíolo en las tres fotos de una hoja, pero eso",
        "no se puede verificar sin mirar las fotos. Coherencia geométrica (ROI por fondo en",
        "orden blanco/gris oscuro/natural; ρ de Spearman del área relativa de cada ROI frente",
        "a gris oscuro):",
        "",
        "```",
        pairing.to_string(index=False, **fmt),
        "```",
        "",
        "**Conclusión:** el emparejamiento por `leaflet_id` **no es fiable en todas las hojas**",
        "(p. ej. `c01` tiene una ROI más en blanco y áreas que no cuadran). No se fuerza: por",
        "folíolo solo se analizan las hojas con una ROI por foto; el resto, por hoja.",
        "",
        "## 2. Acuerdo y variación de la puntuación",
        "",
    ]
    a_leaf, a_leaflet = agreement(leaf_major), agreement(leaflet_out)
    i_leaf, i_leaflet = icc1(leaf_score), icc1(leaflet_score)
    lines += [
        "| Nivel | Unidades | Mismo resultado en los 3 fondos | DT dentro (entre fondos) "
        "| DT entre unidades | ICC(1) |",
        "|---|---|---|---|---|---|",
        f"| Hoja (mayoría de folíolos; puntuación media) | {i_leaf['n']} | "
        f"{a_leaf['same']}/{a_leaf['n']} | {i_leaf['sd_within']:.2f} | "
        f"{i_leaf['sd_between']:.2f} | {i_leaf['icc1']:.2f} |",
        f"| Folíolo (solo hojas con una ROI por foto) | {i_leaflet['n']} | "
        f"{a_leaflet['same']}/{a_leaflet['n']} | {i_leaflet['sd_within']:.2f} | "
        f"{i_leaflet['sd_between']:.2f} | {i_leaflet['icc1']:.2f} |",
        "",
        "Puntuación media de B por hoja y fondo (folíolos concluyentes):",
        "",
        "```",
        leaf_score.to_string(**fmt),
        "```",
        "",
        "Folíolos emparejables (resultado y puntuación):",
        "",
        "```",
        pd.concat([leaflet_out, leaflet_score], axis=1, keys=["resultado", "puntuación"]).to_string(
            **fmt
        ),
        "```",
        "",
        "## 3. b* y `b_hist_07` por fondo",
        "",
        "**No se puede calcular con lo guardado.** `contributions_B_roi.csv` solo tiene la",
        "media de cada contribución sobre los 95 folíolos concluyentes, no los valores por",
        "folíolo, así que no hay b* ni `b_hist_07` por fondo ni diferencias emparejadas.",
        "Obtenerlos exigiría volver a procesar las fotos, que esta tarea no permite.",
        "",
    ]
    (args.dir / "exploratory_backgrounds.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
