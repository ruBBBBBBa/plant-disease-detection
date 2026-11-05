"""Genera el manifiesto ``data/metadata.csv`` con splits sin fugas (tarea T02).

Aplica las decisiones D006–D011 (``docs/DECISIONS.md``):
- PlantVillage ``color``: ``train``/``val``/``test`` solo con imágenes de hoja conocida, por
  grupos de hoja y estratificado por enfermedad; el resto → ``excluded`` con motivo.
- PlantDoc: todo ``external_test``, sin duplicados exactos.
- ``segmented`` no entra en el manifiesto.

Salidas: ``data/metadata.csv``, ``reports/data/split_summary.md`` y
``reports/data/target_spot_leaves.jpg`` (comprobación visual). Termina con error si
la comprobación de fugas encuentra algún problema. Solo lee ``data/raw/``.

Ejemplo:
    python scripts/prepare_dataset.py --seed 42 --near-dup-threshold 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402
from src import manifest as mf  # noqa: E402

MANIFEST_COLUMNS = [
    "image_id",
    "source",
    "image_path",
    "species",
    "disease_name",
    "binary_label",
    "leaf_group_id",
    "split",
    "exclusion_reason",
    "in_subset",
    "modality",
    "folder",
    "original_split",
    "sha1",
    "dhash",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Manifiesto y splits sin fugas (T02).")
    parser.add_argument("--plantvillage-root", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--plantdoc-root", type=Path, default=Path("data/raw/plantdoc"))
    parser.add_argument("--mapping", type=Path, default=Path("configs/label_mapping.csv"))
    parser.add_argument("--species", default="tomato")
    parser.add_argument("--out", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports/data"))
    parser.add_argument(
        "--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TR", "VA", "TE")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near-dup-threshold",
        type=int,
        default=2,
        help="Distancia de Hamming máxima entre dHash (64 bits) para considerar casi-duplicados.",
    )
    parser.add_argument("--subset-healthy", type=int, default=500)
    parser.add_argument("--subset-diseased", type=int, default=500)
    args = parser.parse_args(argv)
    if abs(sum(args.fractions) - 1) > 1e-9:
        parser.error("--fractions debe sumar 1")
    # Rutas relativas a la raíz del repo (no al directorio desde el que se ejecuta).
    for name in ("plantvillage_root", "plantdoc_root", "mapping", "out", "reports"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else REPO_ROOT / path)
    return args


def rel(path: Path) -> str:
    """Ruta relativa a la raíz del repo si está dentro (para mensajes e informes)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_source(source: str, root: Path, species: str) -> pd.DataFrame:
    """Recorre una fuente con el inventario de T01 y añade el dHash de cada imagen legible."""
    if not root.is_dir():
        sys.exit(f"Error: no existe {rel(root)}. Descarga los datos siguiendo data/README.md.")
    class_dirs = audit.find_class_dirs(root, source, species)
    if not class_dirs:
        sys.exit(f"Error: no hay carpetas de '{species}' en {rel(root)}.")
    base = REPO_ROOT if root.resolve().is_relative_to(REPO_ROOT) else root.resolve()
    df = audit.build_inventory(class_dirs, base).rename(columns={"path": "image_path"})
    df["source"] = source
    df["dhash"] = [
        mf.dhash(cv2.imread(str(base / p))) if ok else ""
        for p, ok in zip(df["image_path"], df["readable"], strict=True)
    ]
    return df


def target_spot_check(pv: pd.DataFrame, root: Path) -> tuple[dict[str, str] | None, str]:
    """Comprueba la correspondencia 1:1 ``Com.G_TgS_FL nnnn`` ↔ ``Com.G_FL nnnn`` (CSV).

    Devuelve el mapa número → hoja si se cumple (y ``None`` si no) y una frase para el informe.
    """
    csv_path = root / "leaf_grouping" / "filtered_leafmaps" / "Tomato___Target_Spot.csv"
    files = pv.loc[pv["class_name"] == "Tomato___Target_Spot", "image_path"]
    numbers = [mf.target_spot_number(Path(p).name) for p in files]
    csv_map = mf.target_spot_leaf_map(csv_path) if csv_path.is_file() else {}
    ok = (
        len(numbers) > 0
        and None not in numbers
        and len(set(numbers)) == len(numbers)
        and set(numbers) == set(csv_map)
    )
    detail = (
        f"{len(numbers)} ficheros, {len(set(n for n in numbers if n))} números distintos; "
        f"{len(csv_map)} números en el CSV; coinciden 1:1: **{'sí' if ok else 'no'}**."
    )
    return (csv_map if ok else None), detail


def mark_plantdoc_duplicates(pd_df: pd.DataFrame) -> pd.DataFrame:
    """Excluye las copias exactas de PlantDoc (conserva la primera por ruta) y marca conflictos.

    Si las copias tenían enfermedades distintas (``tomato_V8.jpg``), la copia conservada pasa a
    ``disease_name = "conflict:<a>|<b>"``; ``binary_label`` se mantiene si todas coinciden.
    """
    pd_df = pd_df.sort_values("image_path").copy()
    for _, group in pd_df.groupby("sha1"):
        if len(group) < 2:
            continue
        keep = group.index[0]
        diseases = sorted(group["disease_name"].unique())
        if len(diseases) > 1:
            pd_df.loc[keep, "disease_name"] = "conflict:" + "|".join(diseases)
        if group["binary_label"].nunique() > 1:
            pd_df.loc[keep, "binary_label"] = "conflict"
        dropped = group.index[1:]
        pd_df.loc[dropped, "split"] = "excluded"
        pd_df.loc[dropped, "exclusion_reason"] = "exact_duplicate_of:" + pd_df.loc[keep, "image_id"]
    return pd_df


def build_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Construye el manifiesto completo y devuelve también datos para el informe."""
    mapping = mf.load_label_mapping(args.mapping)
    pv = load_source("plantvillage", args.plantvillage_root, args.species)
    pdoc = load_source("plantdoc", args.plantdoc_root, args.species)
    df = mf.apply_label_mapping(pd.concat([pv, pdoc], ignore_index=True), mapping)
    df["image_id"] = [
        mf.image_id(s, p) for s, p in zip(df["source"], df["image_path"], strict=True)
    ]
    df["modality"] = "color"
    df["folder"] = df["class_name"]
    df["split"] = np.where(df["source"] == "plantdoc", "external_test", "")
    df["exclusion_reason"] = ""
    df.loc[~df["readable"], ["split", "exclusion_reason"]] = ["excluded", "unreadable"]

    # --- PlantVillage: hoja conocida (D006, D009) ---
    is_pv = df["source"] == "plantvillage"
    leaf_map = audit.load_leaf_map(args.plantvillage_root / "leaf-map.json")
    ts_map, ts_detail = target_spot_check(df[is_pv], args.plantvillage_root)
    df["leaf_id"] = ""
    df.loc[is_pv, "leaf_id"] = mf.assign_leaf_ids(df[is_pv], leaf_map, ts_map)
    leaf_class = df["leaf_id"].str.split(":::").str[0]
    mismatches = int((is_pv & (df["leaf_id"] != "") & (leaf_class != df["class_name"])).sum())
    no_leaf = is_pv & (df["leaf_id"] == "") & (df["split"] != "excluded")
    df.loc[no_leaf, ["split", "exclusion_reason"]] = ["excluded", "no_leaf_id"]
    df["leaf_group_id"] = np.where(df["leaf_id"] != "", "pv:" + df["leaf_id"], "")

    # --- PlantDoc: grupo = imagen, sin duplicados exactos (D010) ---
    is_pd = df["source"] == "plantdoc"
    df.loc[is_pd, "leaf_group_id"] = df.loc[is_pd, "image_id"]
    cols = ["disease_name", "binary_label", "split", "exclusion_reason"]
    df.loc[is_pd, cols] = mark_plantdoc_duplicates(df[is_pd])[cols]

    # --- Duplicados exactos y casi-duplicados en un mismo grupo; partición; subconjunto ---
    usable = is_pv & (df["split"] != "excluded")
    pv_used = df[usable].reset_index()
    groups_before = pv_used["leaf_group_id"].nunique()
    pairs = mf.near_duplicate_pairs(pv_used["dhash"].tolist(), args.near_dup_threshold)
    pv_used["leaf_group_id"] = mf.merge_groups(pv_used, pairs)
    assignment = mf.split_groups(pv_used, tuple(args.fractions), args.seed)
    pv_used["split"] = pv_used["leaf_group_id"].map(assignment)
    pv_used["in_subset"] = mf.select_subset(
        pv_used, args.subset_healthy, args.subset_diseased, tuple(args.fractions), args.seed
    )
    pv_used = pv_used.set_index("index")
    df["in_subset"] = False
    df.loc[pv_used.index, ["leaf_group_id", "split", "in_subset"]] = pv_used[
        ["leaf_group_id", "split", "in_subset"]
    ]
    df["in_subset"] = df["in_subset"].astype(bool)

    info = {
        "ts_map": ts_map,
        "ts_detail": ts_detail,
        "mismatches": mismatches,
        "pairs": len(pairs),
        "groups_before": groups_before,
        "groups_after": pv_used["leaf_group_id"].nunique(),
        "exact_dup_rows_used": int(pv_used["sha1"].duplicated(keep=False).sum()),
    }
    manifest = df.sort_values(["source", "image_path"]).reset_index(drop=True)
    return manifest, info


def target_spot_grid(manifest: pd.DataFrame, seed: int, out: Path) -> list[str]:
    """Rejilla con las fotos de 3 hojas de *Target_Spot* al azar (una fila por hoja)."""
    ts = manifest[
        (manifest["folder"] == "Tomato___Target_Spot") & (manifest["split"] != "excluded")
    ]
    leaves = random.Random(seed).sample(sorted(ts["leaf_group_id"].unique()), 3)
    rows = []
    for leaf in leaves:
        paths = sorted(ts.loc[ts["leaf_group_id"] == leaf, "image_path"])
        thumbs = [audit.letterbox(cv2.imread(str(REPO_ROOT / p)), 160) for p in paths]
        thumbs += [np.full((160, 160, 3), 255, np.uint8)] * (4 - len(thumbs))
        rows.append(np.hstack(thumbs[:4]))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return leaves


def count_table(df: pd.DataFrame, rows: list[str], column: str = "split") -> pd.DataFrame:
    """Tabla de recuentos ``rows`` × ``column`` con totales, lista para Markdown."""
    table = pd.crosstab([df[r] for r in rows], df[column], margins=True, margins_name="total")
    return table.reset_index()


def write_summary(manifest: pd.DataFrame, info: dict, args: argparse.Namespace, leaves) -> str:
    """Escribe ``split_summary.md`` con distribución original, usada, subconjunto y fugas."""
    used = manifest[manifest["split"] != "excluded"]
    excluded = manifest[manifest["split"] == "excluded"]
    subset = manifest[manifest["in_subset"]]
    groups = used.groupby("split")["leaf_group_id"].nunique().rename("groups")
    images = used.groupby("split").size().rename("images")
    lines = [
        "# Manifiesto y splits — resumen",
        "",
        f"Generado con `python scripts/prepare_dataset.py --seed {args.seed} "
        f"--fractions {' '.join(str(f) for f in args.fractions)} "
        f"--near-dup-threshold {args.near_dup_threshold} --subset-healthy {args.subset_healthy} "
        f"--subset-diseased {args.subset_diseased}`. Decisiones: D006–D011 (`docs/DECISIONS.md`).",
        "",
        f"- Filas del manifiesto: **{len(manifest)}** ({rel(args.out)}).",
        f"- Usadas: **{len(used)}**; excluidas: **{len(excluded)}**.",
        "",
        "## Distribución original (antes de excluir)",
        "",
        audit.md_table(count_table(manifest, ["binary_label", "disease_name"], "source")),
        "",
        "## Excluidas por motivo",
        "",
        audit.md_table(
            count_table(
                excluded.assign(
                    reason=excluded["exclusion_reason"].str.replace(r":.*", "", regex=True)
                ),
                ["source", "disease_name"],
                "reason",
            )
        ),
        "",
        "## Distribución usada por split",
        "",
        audit.md_table(pd.concat([images, groups], axis=1).reset_index()),
        "",
        audit.md_table(count_table(used, ["source", "binary_label", "disease_name"])),
        "",
        "## Subconjunto inicial (`in_subset`)",
        "",
        audit.md_table(count_table(subset, ["binary_label", "disease_name"])),
        "",
        "## Grupos, duplicados y comprobaciones",
        "",
        f"- Imágenes cuyo `leaf_id` es de otra clase que su carpeta: **{info['mismatches']}**.",
        f"- Target_Spot (`Com.G_TgS_FL nnnn` ↔ CSV `Com.G_FL nnnn`): {info['ts_detail']}",
        f"- Duplicados exactos entre las imágenes usadas de PlantVillage: "
        f"{info['exact_dup_rows_used']} filas.",
        f"- Casi-duplicados (dHash 64 bits, Hamming ≤ {args.near_dup_threshold}) entre imágenes "
        f"usadas de PlantVillage: **{info['pairs']}** pares.",
        f"- Grupos de hoja: {info['groups_before']} antes de unir duplicados y "
        f"{info['groups_after']} después.",
        "- Comprobación de fugas (`check_leakage`): **sin problemas**.",
    ]
    if leaves:
        lines += [
            "",
            "Rejilla de comprobación visual de Target_Spot: `target_spot_leaves.jpg` "
            f"(una fila por hoja: {', '.join(f'`{leaf}`' for leaf in leaves)}).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Genera manifiesto, comprueba fugas y escribe el informe."""
    args = parse_args(argv)
    manifest, info = build_manifest(args)
    problems = mf.check_leakage(manifest, args.near_dup_threshold)
    if problems:
        print("\n".join(problems[:20]), file=sys.stderr)
        sys.exit(f"Error: {len(problems)} fugas detectadas; no se escribe el manifiesto.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    manifest[MANIFEST_COLUMNS].to_csv(args.out, index=False)
    leaves = None
    if info["ts_map"]:
        leaves = target_spot_grid(manifest, args.seed, args.reports / "target_spot_leaves.jpg")
    summary = write_summary(manifest, info, args, leaves)
    (args.reports / "split_summary.md").write_text(summary, encoding="utf-8")
    counts = manifest["split"].value_counts().to_dict()
    print(f"Manifiesto: {rel(args.out)} ({len(manifest)} filas) {counts}")
    print(f"Subconjunto: {int(manifest['in_subset'].sum())} imágenes. Fugas: ninguna.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
