"""Audita las imágenes de una especie en PlantVillage o PlantDoc (tarea T01).

Genera en ``--out``:
- ``inventory.csv``: una fila por fichero (ruta, clase, split original, extensión, legible,
  alto, ancho, canales, bytes, SHA-1). Regenerable; no se versiona.
- ``summary.md``: recuentos, extensiones, resoluciones, ilegibles, duplicados exactos y,
  para PlantVillage, emparejamiento ``color``/``segmented`` y agrupación por hoja.
- ``samples_<clase>.jpg``: rejilla de ejemplos aleatorios (semilla fija) por clase.

Suposición: carpeta = clase. El script solo lee ``--root``; nunca escribe en ``data/raw/``.

Ejemplos:
    python scripts/audit_dataset.py --source plantvillage
    python scripts/audit_dataset.py --source plantdoc \\
        --compare-with reports/audit/plantvillage/inventory.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Auditoría de un dataset de hojas (carpeta = clase). Solo lee --root."
    )
    parser.add_argument("--source", required=True, choices=["plantvillage", "plantdoc"])
    parser.add_argument(
        "--root", type=Path, help="Raíz del dataset (por defecto data/raw/<source>)."
    )
    parser.add_argument(
        "--species", default="tomato", help="Filtro por nombre de carpeta, sin mayúsculas."
    )
    parser.add_argument("--out", type=Path, help="Carpeta de salida (reports/audit/<source>/).")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la selección de ejemplos.")
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="inventory.csv de otra fuente para buscar duplicados exactos entre fuentes.",
    )
    args = parser.parse_args(argv)
    args.root = (args.root or REPO_ROOT / "data" / "raw" / args.source).resolve()
    # Las rutas del inventario son relativas a la raíz del repo (o a --root si está fuera).
    args.base = REPO_ROOT if args.root.is_relative_to(REPO_ROOT) else args.root
    args.out = args.out or REPO_ROOT / "reports" / "audit" / args.source
    return args


def md_table(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown sencilla (sin dependencias extra)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def display_path(path: Path) -> str:
    """Ruta relativa a la raíz del repo si está dentro de él (para no imprimir rutas absolutas)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def summary_sections(inv: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Secciones comunes a cualquier fuente: recuentos, formatos, resoluciones y duplicados."""
    counts = audit.class_counts(inv)
    stats, top_sizes = audit.resolution_stats(inv)
    dups = audit.exact_duplicates(inv)
    unreadable = inv[~inv["readable"]]
    lines = [
        f"# Auditoría — {args.source} ({args.species})",
        "",
        f"Generado con `python scripts/audit_dataset.py --source {args.source} "
        f"--species {args.species} --seed {args.seed}`. Raíz: `{display_path(args.root)}`. "
        "Suposición: carpeta = clase.",
        "",
        f"- Ficheros: **{len(inv)}** en {inv['class_name'].nunique()} clases.",
        f"- Ilegibles para OpenCV: **{len(unreadable)}**.",
        "",
        "## Recuentos por clase",
        "",
        md_table(counts),
        "",
        "## Extensiones y canales",
        "",
        md_table(inv["extension"].value_counts().rename_axis("extension").reset_index()),
        "",
        md_table(inv["channels"].value_counts().rename_axis("channels").reset_index()),
        "",
        "## Resoluciones (píxeles)",
        "",
        md_table(stats.reset_index(names="stat")),
        "",
        "Más frecuentes (ancho×alto):",
        "",
        md_table(top_sizes.rename_axis("size").reset_index()),
        "",
        "## Ficheros ilegibles",
        "",
        md_table(unreadable[["path", "class_name"]]) if len(unreadable) else "Ninguno.",
        "",
        "## Duplicados exactos (mismo SHA-1)",
        "",
        f"- Grupos dentro de la misma clase: {(dups['scope'] == 'same_class').sum()}",
        f"- Grupos entre clases distintas: {(dups['scope'] == 'cross_class').sum()}",
        f"- Ficheros implicados: {int(dups['files'].sum())}",
        "",
        md_table(dups[["files", "scope", "classes", "paths"]]) if len(dups) else "",
    ]
    if args.compare_with:
        other = pd.read_csv(args.compare_with)
        cross = audit.cross_source_duplicates(inv, other)
        lines += [
            "## Duplicados exactos con otra fuente",
            "",
            f"Comparado con `{display_path(args.compare_with)}`: **{len(cross)}** pares idénticos.",
            "",
            md_table(cross) if len(cross) else "",
        ]
    return lines


def plantvillage_sections(inv: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Secciones propias de PlantVillage: pares color/segmented y agrupación por hoja."""
    pairs = audit.pair_color_segmented(args.root / "raw" / "color", args.root / "raw" / "segmented")
    pairs = pairs[pairs["class_name"].str.lower().str.contains(args.species.lower())]
    lines = ["## Emparejamiento color ↔ segmented (por nombre base)", "", md_table(pairs), ""]

    leaf_map_path = args.root / "leaf-map.json"
    if not leaf_map_path.is_file():
        return lines + ["`leaf-map.json` no encontrado: sin análisis de agrupación por hoja.", ""]
    leaf_map = audit.load_leaf_map(leaf_map_path)
    lines += [
        "## Agrupación por hoja (`leaf-map.json`, lógica de `plant_village.py`)",
        "",
        "`distinct_keys` = valores distintos de la parte del nombre tras `___`; "
        "`with_leaf_id` = imágenes encontradas en el mapa.",
        "",
        md_table(audit.leaf_group_stats(inv, leaf_map)),
        "",
    ]
    mapping_dir = args.root / "data_distribution_for_SVM"
    mapping_files = {s: mapping_dir / f"{s}_mapping.txt" for s in ("train", "test")}
    if all(p.is_file() for p in mapping_files.values()):
        leak = audit.svm_split_leakage(mapping_files, leaf_map, args.species)
        lines += [
            "## Partición `data_distribution_for_SVM` frente a la agrupación por hoja",
            "",
            md_table(pd.DataFrame([leak])),
            "",
        ]
    return lines


def write_sample_grids(inv: pd.DataFrame, args: argparse.Namespace) -> None:
    """Guarda una rejilla ``samples_<clase>.jpg`` por clase (JPEG calidad 80)."""
    for class_name, group in inv[inv["readable"]].groupby("class_name"):
        paths = [args.base / p for p in group["path"]]
        grid = audit.sample_grid(paths, args.samples_per_class, args.seed)
        if grid is not None:
            out = args.out / f"samples_{audit.slugify(class_name)}.jpg"
            cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la auditoría completa y escribe inventario, resumen y rejillas."""
    args = parse_args(argv)
    if not args.root.is_dir():
        sys.exit(
            f"Error: no existe {display_path(args.root)}. "
            "Descarga los datos siguiendo data/README.md (sección «Descarga»)."
        )
    class_dirs = audit.find_class_dirs(args.root, args.source, args.species)
    if not class_dirs:
        sys.exit(
            f"Error: no hay carpetas de clase con '{args.species}' en {display_path(args.root)}."
        )

    print(f"Auditando {len(class_dirs)} carpetas de clase en {display_path(args.root)}...")
    inv = audit.build_inventory(class_dirs, args.base)
    if inv.empty:
        sys.exit(f"Error: las carpetas de '{args.species}' no contienen ficheros.")

    args.out.mkdir(parents=True, exist_ok=True)
    inv.to_csv(args.out / "inventory.csv", index=False)
    lines = summary_sections(inv, args)
    if args.source == "plantvillage":
        lines += plantvillage_sections(inv, args)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_sample_grids(inv, args)
    print(f"{len(inv)} ficheros auditados. Salidas en {display_path(args.out)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
