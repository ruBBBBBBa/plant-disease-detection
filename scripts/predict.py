"""Inferencia sobre una foto (T07): figura con máscara, regiones candidatas y predicción + JSON.

Modo ROI por defecto (D045): la ROI se dibuja con el ratón o se da con ``--roi x,y,w,h``
(coordenadas de la imagen original); rectángulo pegado al folíolo, sin margen. ``--auto`` usa el
modo automático (K-means de T03). La puntuación es la del SVM
(``decision_function``) y **no es una probabilidad**. Si la máscara es no concluyente (D031/D035)
no hay predicción: se muestra el motivo.

Ejemplos:
    python scripts/predict.py --image foto.jpg --out reports/inference
    python scripts/predict.py --image foto.jpg --roi 120,80,300,200
    python scripts/predict.py --image foto.jpg --auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import external as ex  # noqa: E402
from src import inference as inf  # noqa: E402


def run(image_path: Path, roi, variant: str, out_dir: Path) -> dict:
    """Analiza una foto y escribe ``<nombre>_<modo>.png`` y ``.json`` en ``out_dir``."""
    cfg = inf.load_configs()
    image = ex.read_bgr(image_path)
    if image is None:
        raise SystemExit(f"Error: no se puede leer la imagen {image_path}")
    variant_cfg = cfg["external"]["variants"][variant]
    analysis = inf.segment_and_extract(
        image,
        roi,
        cfg["seg"],
        cfg["feat"],
        cfg["external"]["normalization"]["long_side"],
        cfg["own"]["roi"]["margin"],
    )
    prediction = inf.predict_variant(
        analysis,
        inf.load_pipeline(variant_cfg),
        variant_cfg["group"],
        cfg["external"]["inconclusive"]["ranges"],
    )
    result = {"image": str(image_path)} | inf.result_json(
        analysis, prediction, variant, variant_cfg
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{analysis['mode']}"  # sin with_suffix: el nombre puede llevar puntos
    inf.save_figure(image, analysis, result, out_dir / f"{stem}.png")
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    """Lee los argumentos, ejecuta la inferencia e imprime el resumen."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--roi", type=inf.parse_roi, help="x,y,w,h en la imagen original")
    group.add_argument("--auto", action="store_true", help="modo automático (K-means) sin ROI")
    parser.add_argument("--variant", default="B", choices=["A", "B"], help="por defecto B (D036)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/inference")
    args = parser.parse_args(argv)
    roi = args.roi
    if roi is None and not args.auto:  # por defecto, ROI dibujada con el ratón
        image = ex.read_bgr(args.image)
        if image is None:
            raise SystemExit(f"Error: no se puede leer la imagen {args.image}")
        rois = inf.select_rois(image, "ROI: arrastra y pulsa ENTER", multiple=False)
        if not rois:
            raise SystemExit("Error: no se ha dibujado ninguna ROI")
        roi = rois[0]
    result = run(args.image, roi, args.variant, args.out)
    print("\n".join(inf.figure_lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
