"""Tablas y veredicto de H4–H6 con fotos propias (T07), solo desde las predicciones guardadas.

No abre fotos ni modelos: lee ``experiments/own/v1/predictions.csv.gz`` (de ``eval_own.py``) y
escribe ``metrics.json`` y ``tables.md`` en la misma carpeta. Se puede ejecutar las veces que haga
falta: siempre da lo mismo con las mismas predicciones.

Ejemplo:
    python scripts/report_own.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import own_eval as oe  # noqa: E402


def block(table: pd.DataFrame, index: bool = False) -> str:
    """Tabla como bloque de texto de ancho fijo (sin dependencias extra para Markdown)."""
    return "```text\n" + table.to_string(index=index, float_format="{:.3f}".format) + "\n```"


def main(argv: list[str] | None = None) -> int:
    """Calcula tablas descriptivas e hipótesis y las escribe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/own/v1")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/own.json")
    args = parser.parse_args(argv)
    source = args.dir / "predictions.csv.gz"
    if not source.is_file():
        raise SystemExit(f"Error: falta {source} (se genera con scripts/eval_own.py)")
    config = json.loads(args.config.read_text())
    pred = pd.read_csv(source, keep_default_na=False, dtype={"leaf_id": str})
    pred["inconclusive"] = pred["inconclusive"].astype(str) == "True"

    rates = oe.rate_table(pred)
    keywords = oe.keyword_table(pred, config["notes_keywords"])
    hypotheses = oe.evaluate_hypotheses(pred, config)
    altered = pred[pred["label"] == "altered"][
        ["file", "mode", "leaflet_id", "variant", "inconclusive", "reasons", "prediction", "score"]
    ]
    metrics = {
        "outcome": config["outcome"],
        "rates": rates.to_dict("records"),
        "hypotheses": hypotheses,
    }
    (args.dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Fotos propias — tablas (generado por scripts/report_own.py)",
        "",
        f"{config['outcome'].capitalize()}. Tasas = media de las tasas por hoja compuesta.",
        "",
        "## Por variante × modo × fondo (sanas)",
        "",
        block(rates),
        "",
        "## Hipótesis (variante " + config["default_variant"] + ", bootstrap por hoja)",
        "",
        block(pd.DataFrame(hypotheses).T, index=True),
        "",
        "## Estratificación por notas (descriptiva, sin veredicto)",
        "",
        block(keywords),
        "",
        "## Hojas alteradas (descriptivo, sin métricas de acierto)",
        "",
        block(altered) if len(altered) else "Sin hojas alteradas.",
        "",
    ]
    (args.dir / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
