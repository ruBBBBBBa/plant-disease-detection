"""Anotación de condiciones de Taiwán (T08, fase A): ``data/annotations/taiwan_conditions.csv``.

Muestra cada imagen ``external_test`` de ``data/taiwan_metadata.csv`` **sin nombre de fichero ni
clase** y en orden aleatorio fijo (semilla), para que la carpeta (sana / enferma) no influya.
No muestra máscaras ni predicciones. Se guarda tras cada imagen y, al volver a abrirla, sigue por
la primera sin anotar.

Teclas:
  - hojas:  ``1`` = leaflet   ``2`` = compound   ``3`` = multiple
  - fondo:  ``s`` = simple    ``c`` = complex
  - ``b`` = volver a la imagen anterior (para corregirla)     ``q`` = salir

Al tener las dos etiquetas pasa a la siguiente. Reglas fijas del enunciado (D053):

- ``leaflet`` = un solo folíolo, sin otras hojas en primer plano (condición tipo PlantVillage);
  ``compound`` = una sola hoja compuesta (varios folíolos con su raquis), sin otras hojas enteras;
  ``multiple`` = se ven otras hojas enteras o casi enteras, o la planta. **Si hay dudas, multiple.**
- ``simple`` = fondo liso de un solo color (papel, mesa, cartulina); ``complex`` = cualquier otra
  cosa (planta, suelo, objetos, varias texturas). **Si hay dudas, complex.**

Ejemplos:
    python scripts/annotate_conditions.py
    python scripts/annotate_conditions.py --summary    # solo recuentos, sin ventana
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["file", "leaves", "background"]
LEAVES = {"leaflet", "compound", "multiple"}
KEYS = {
    "1": ("leaves", "leaflet"),
    "2": ("leaves", "compound"),
    "3": ("leaves", "multiple"),
    "s": ("background", "simple"),
    "c": ("background", "complex"),
}
HELP = [
    "hojas: 1 foliolo / 2 compuesta / 3 varias o planta (dudas: 3)",
    "fondo: s simple (liso, un color) / c complex (dudas: c)",
    "b atras   q salir",
]


def annotation_order(files: list[str], seed: int) -> list[str]:
    """Orden aleatorio reproducible (no por carpeta, para no agrupar las clases)."""
    order = np.random.default_rng(seed).permutation(len(files))
    return [files[i] for i in order]


def read_annotations(path: Path) -> pd.DataFrame:
    """Anotaciones existentes (vacía si aún no hay fichero)."""
    if path.is_file():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def write_annotation(path: Path, file: str, leaves: str, background: str) -> pd.DataFrame:
    """Guarda (o reemplaza) la fila de ``file`` y reescribe el CSV ordenado por ``file``."""
    if leaves not in LEAVES or background not in {"simple", "complex"}:
        raise ValueError(f"{file}: etiquetas no válidas {leaves!r}, {background!r}")
    table = read_annotations(path)
    table = table[table["file"] != file]
    new = pd.DataFrame([{"file": file, "leaves": leaves, "background": background}])
    table = pd.concat([table, new]).sort_values("file")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def apply_key(state: dict, key: str) -> dict:
    """Nuevo estado ``{"leaves", "background"}`` tras pulsar ``key`` (las demás no cambian nada)."""
    if key in KEYS:
        field, value = KEYS[key]
        return state | {field: value}
    return state


def render(image: np.ndarray, state: dict, position: str, size: int) -> np.ndarray:
    """Imagen ampliada a ``size`` px con una banda de ayuda debajo (sin nombre ni clase)."""
    scale = size / max(image.shape[:2])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    band = np.full((118, shown.shape[1], 3), 255, np.uint8)
    current = (
        f"{position}   hojas: {state.get('leaves', '-')}   fondo: {state.get('background', '-')}"
    )
    for i, text in enumerate([current, *HELP]):
        cv2.putText(band, text, (6, 22 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.vstack([shown, band])


def summary(table: pd.DataFrame, total: int) -> str:
    """Cuántas imágenes hay anotadas y su reparto ``leaves × background``."""
    cross = pd.crosstab(table["leaves"], table["background"], margins=True) if len(table) else ""
    return f"Anotadas {len(table)} de {total}.\n{cross}"


def main(argv: list[str] | None = None) -> int:
    """Bucle de anotación con OpenCV."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/taiwan_metadata.csv")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/annotations/taiwan_conditions.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=600, help="lado de la imagen en pantalla")
    parser.add_argument("--summary", action="store_true", help="solo imprime los recuentos")
    args = parser.parse_args(argv)
    if not args.metadata.is_file():
        raise SystemExit(f"Error: falta {args.metadata}; ejecuta scripts/prepare_taiwan.py")
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    files = annotation_order(
        sorted(meta.loc[meta["split"] == "external_test", "image_path"]), args.seed
    )
    if args.summary:
        print(summary(read_annotations(args.out), len(files)))
        return 0

    done = set(read_annotations(args.out)["file"])
    i = next((k for k, f in enumerate(files) if f not in done), len(files))
    window = "Condiciones (T08)"
    while i < len(files):
        image = cv2.imread(str(REPO_ROOT / files[i]))
        state: dict = {}
        while not {"leaves", "background"} <= state.keys():
            cv2.imshow(window, render(image, state, f"{i + 1}/{len(files)}", args.size))
            key = chr(cv2.waitKey(0) & 0xFF).lower()
            if key == "q":
                cv2.destroyAllWindows()
                print(summary(read_annotations(args.out), len(files)))
                return 0
            if key == "b" and i > 0:
                break
            state = apply_key(state, key)
        else:
            write_annotation(args.out, files[i], state["leaves"], state["background"])
            i += 1
            continue
        i -= 1  # «b»: volver a la anterior
    cv2.destroyAllWindows()
    print(summary(read_annotations(args.out), len(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
