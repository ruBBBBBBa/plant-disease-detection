"""Fija el protocolo de PlantDoc (T06) con datos de PlantVillage únicamente.

Calcula los rangos [p0,5 ; p99,5] de las características de calidad de la máscara en
``train`` + ``val`` de PlantVillage y los guarda en ``configs/external.json`` junto con la
normalización, las variantes y las hipótesis preregistradas (D030). No lee imágenes.

Ejemplo:
    python scripts/make_external_config.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402
from src import training as tr  # noqa: E402

VARIANTS = {
    "A": {"group": "combinadas", "model": "svm_lineal"},
    "B": {"group": "color_manchas", "model": "svm_lineal"},
    "C": {"group": "color_manchas_sin_l_v", "model": "svm_rbf"},
    "C_lineal": {"group": "color_manchas_sin_l_v", "model": "svm_lineal"},
    # D030 no fija el modelo del control: se preregistra el mismo estimador que A, B y C_lineal.
    "control": {"group": "control_nitidez", "model": "svm_lineal"},
}
# «worse» debería caer más que «better»: Δ_worse − Δ_better > 0.
HYPOTHESES = {
    "H1": {"worse": "A", "better": "B"},
    "H2": {"worse": "B", "better": "C_lineal"},
    "H3": {"worse": "control", "better": "A"},
}


def main(argv: list[str] | None = None) -> int:
    """Calcula los umbrales con PlantVillage y escribe ``configs/external.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPO_ROOT / "data/processed/features_v1_subset.csv"
    )
    parser.add_argument(
        "--sigma", type=Path, default=REPO_ROOT / "reports/classification/sigma_selection.json"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "configs/external.json")
    args = parser.parse_args(argv)
    table = pd.read_csv(args.features)
    table = table[table["split"].isin(["train", "val"])]
    ranges = ex.quality_ranges(table)
    config = {
        "_comment": "Fijado solo con train+val de PlantVillage antes de procesar PlantDoc (T06).",
        "normalization": {"long_side": 256, "interpolation": "INTER_AREA"},
        "inconclusive": {"percentiles": [0.5, 99.5], "ranges": ranges},
        "reference_flag_rate_pct": ex.reference_flag_rate(table, ranges),
        "blur_sigma": json.loads(args.sigma.read_text())["chosen_sigma"],
        "variants": VARIANTS,
        "hypotheses": HYPOTHESES,
        "bootstrap": {"n_boot": 10000, "n_boot_descriptive": 2000, "seed": 42},
        "reference_features_sha1": tr.sha1_file(args.features),
    }
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({k: config[k] for k in ("inconclusive", "reference_flag_rate_pct")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
