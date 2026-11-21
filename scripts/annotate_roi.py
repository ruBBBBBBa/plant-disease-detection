"""Dibuja las ROI de los folíolos de las fotos propias (T07) y las guarda en ``own_rois.csv``.

Solo muestra la foto y guarda rectángulos: **no segmenta ni predice**.

Regla fija de dibujo: **rectángulo pegado al folíolo, sin margen: el margen lo añade el recorte**.
Un rectángulo por cada folíolo visible entero, sin elegir folíolos según su aspecto.

Uso, foto a foto (``cv2.selectROIs``):
  - arrastra un rectángulo y pulsa ENTER o ESPACIO para confirmarlo; repite con cada folíolo;
  - pulsa ESC para terminar la foto (si no has dibujado nada, la foto queda sin ROI);
  - en el terminal, escribe el ``leaflet_id`` de cada rectángulo (ENTER acepta ``f1``, ``f2``…
    en el orden en que los dibujaste). Usa el mismo ``leaflet_id`` para el mismo folíolo en las
    fotos de la misma hoja con distinto fondo, cuando lo reconozcas.

Las coordenadas se guardan sobre la imagen original (tal como la lee ``cv2.imread``, con la
rotación EXIF aplicada; ``predict.py`` y ``eval_own.py`` la leen igual). Las fotos ya anotadas se
saltan salvo con ``--redo``; volver a anotar una foto sustituye sus filas.

Ejemplo:
    python scripts/annotate_roi.py
    python scripts/annotate_roi.py --only h03_natural.jpg --redo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import inference as inf  # noqa: E402
from src import own_eval as oe  # noqa: E402


def read_rois(path: Path) -> pd.DataFrame:
    """Lee ``own_rois.csv`` o devuelve una tabla vacía con sus columnas."""
    if path.is_file():
        return pd.read_csv(path, keep_default_na=False)
    return pd.DataFrame(columns=oe.ROI_COLUMNS)


def write_rois(path: Path, file: str, rois: list, leaflet_ids: list[str]) -> pd.DataFrame:
    """Sustituye las filas de ``file`` por sus ROI (coordenadas originales) y guarda el CSV."""
    if len(set(leaflet_ids)) != len(leaflet_ids):
        raise ValueError(f"{file}: leaflet_id repetido {leaflet_ids}")
    bad = [i for i in leaflet_ids if not oe.LEAFLET_PATTERN.match(i)]
    if bad:
        raise ValueError(f"{file}: leaflet_id debe ser f<n>: {bad}")
    table = read_rois(path)
    table = table[table["file"] != file]
    new = pd.DataFrame(
        [
            {"file": file, "leaflet_id": i, "x": x, "y": y, "w": w, "h": h}
            for i, (x, y, w, h) in zip(leaflet_ids, rois, strict=True)
        ],
        columns=oe.ROI_COLUMNS,
    )
    table = pd.concat([table, new]).sort_values(["file", "leaflet_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def ask_leaflet_ids(n: int) -> list[str]:
    """Pregunta en el terminal el ``leaflet_id`` de cada rectángulo (por defecto f1, f2…)."""
    ids = []
    for i in range(1, n + 1):
        answer = input(f"  rectángulo {i}: leaflet_id [f{i}]: ").strip()
        ids.append(answer or f"f{i}")
    return ids


def main(argv: list[str] | None = None) -> int:
    """Recorre las fotos y guarda las ROI dibujadas."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data/raw/own")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/annotations/own_rois.csv")
    parser.add_argument("--only", nargs="*", help="solo estos ficheros")
    parser.add_argument("--redo", action="store_true", help="volver a anotar fotos ya anotadas")
    args = parser.parse_args(argv)
    if not args.images_dir.is_dir():
        raise SystemExit(f"Error: no existe {args.images_dir}")
    files = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.only:
        files = [p for p in files if p.name in set(args.only)]
    done = set(read_rois(args.out)["file"])
    for path in files:
        if path.name in done and not args.redo:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path.name}: no se puede leer, se salta")
            continue
        print(f"{path.name}: dibuja los folíolos (ENTER confirma cada uno, ESC termina)")
        rois = inf.select_rois(image, path.name, multiple=True)
        if not rois:
            print("  sin ROI")
            continue
        while True:
            try:
                write_rois(args.out, path.name, rois, ask_leaflet_ids(len(rois)))
                break
            except ValueError as err:
                print(f"  {err}; vuelve a escribirlos")
        print(f"  {len(rois)} ROI guardadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
