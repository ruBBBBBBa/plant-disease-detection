"""Manifiesto del segundo test externo, Taiwán (T08, fase A): ``data/taiwan_metadata.csv``.

Parte del inventario de ``audit_dataset.py --source taiwan`` (solo ``Preprocessed data``; los
aumentos no se leen). Para cada imagen:

- etiqueta según ``configs/label_mapping.csv`` (una carpeta sin mapear es un error) y
  ``seen_in_training``: si su ``disease_name`` existe en PlantVillage;
- **casi-duplicados frente a PlantVillage** con el dHash de T02 (``src/manifest.py``) y el mismo
  umbral de Hamming: la imagen con pareja se excluye (``near_duplicate_of_pv:<image_id>``);
- ilegibles y duplicados exactos dentro de Taiwán, excluidos con su motivo.

Todo lo demás es ``external_test``: el reparto Train/Test del dataset se conserva en
``original_split`` pero no se usa. No segmenta, no extrae características ni usa modelos. Va en
un manifiesto aparte para no tocar ``data/metadata.csv`` ni los splits de PlantVillage.

Ejemplo:
    python scripts/audit_dataset.py --source taiwan
    python scripts/prepare_taiwan.py --near-dup-threshold 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src import data_audit as audit  # noqa: E402
from src import manifest as mf  # noqa: E402

COLUMNS = [
    "image_id",
    "source",
    "image_path",
    "species",
    "disease_name",
    "binary_label",
    "seen_in_training",
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


def build_manifest(
    inventory: pd.DataFrame, mapping: pd.DataFrame, pv: pd.DataFrame, threshold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manifiesto de Taiwán y tabla de pares casi duplicados con PlantVillage.

    ``inventory`` ya trae ``dhash`` (vacío si la imagen es ilegible). ``pv`` son las filas de
    PlantVillage de ``data/metadata.csv`` (todas, también las excluidas: cualquier copia de
    PlantVillage solaparía con el entrenamiento o su origen).
    """
    df = mf.apply_label_mapping(inventory.assign(source="taiwan"), mapping)
    df["image_path"] = df["path"]
    df["image_id"] = [mf.image_id("taiwan", p) for p in df["image_path"]]
    df["leaf_group_id"] = df["image_id"]  # sin información de hoja: cada imagen es su grupo
    df["folder"] = df["class_name"]
    df["modality"], df["in_subset"] = "color", False
    df["seen_in_training"] = (df["binary_label"] == "healthy") | df["disease_name"].isin(
        set(pv["disease_name"])
    )
    df["exclusion_reason"] = ""
    df.loc[~df["readable"], "exclusion_reason"] = "unreadable"
    first = df.groupby("sha1")["image_id"].transform("first")
    exact = df["sha1"].duplicated() & df["readable"]
    df.loc[exact, "exclusion_reason"] = "exact_duplicate_of:" + first[exact]

    ok = np.flatnonzero(df["readable"].to_numpy())
    pairs = mf.cross_near_duplicates(df["dhash"].iloc[ok].tolist(), pv["dhash"].tolist(), threshold)
    near = pd.DataFrame(
        {
            "taiwan_id": df["image_id"].iloc[ok[pairs[:, 0]]].to_numpy(),
            "taiwan_path": df["image_path"].iloc[ok[pairs[:, 0]]].to_numpy(),
            "pv_id": pv["image_id"].iloc[pairs[:, 1]].to_numpy(),
            "pv_path": pv["image_path"].iloc[pairs[:, 1]].to_numpy(),
            "pv_split": pv["split"].iloc[pairs[:, 1]].to_numpy(),
            "hamming": pairs[:, 2],
        }
    ).sort_values(["taiwan_id", "hamming", "pv_id"])
    closest = near.drop_duplicates("taiwan_id").set_index("taiwan_id")["pv_id"]
    hit = df["image_id"].isin(closest.index) & (df["exclusion_reason"] == "")
    df.loc[hit, "exclusion_reason"] = "near_duplicate_of_pv:" + df.loc[hit, "image_id"].map(closest)
    df["split"] = np.where(df["exclusion_reason"] == "", "external_test", "excluded")
    return df[COLUMNS], near


def report(df: pd.DataFrame, near: pd.DataFrame, within: int, args: argparse.Namespace) -> str:
    """Resumen en Markdown: recuentos, exclusiones y pares casi duplicados."""
    used = df[df["split"] == "external_test"]
    counts = pd.crosstab(
        [df["folder"], df["disease_name"], df["seen_in_training"]], df["split"], margins=True
    )
    return "\n".join(
        [
            "# Taiwán — manifiesto del test externo (T08, fase A)",
            "",
            "Generado con `python scripts/prepare_taiwan.py --near-dup-threshold "
            f"{args.near_dup_threshold}`. Dataset: Huang, M.-L. y Chang, Y.-H. (2020), *Dataset of "
            "Tomato Leaves*, Mendeley Data, V1, DOI 10.17632/ngdgg79rzb.1, licencia CC BY 4.0. "
            "Solo `Preprocessed data`; `data augmentation` no se lee.",
            "",
            f"- Imágenes: **{len(df)}**; `external_test`: **{len(used)}**; excluidas: "
            f"**{(df['split'] == 'excluded').sum()}**.",
            f"- Casi-duplicados frente a PlantVillage (dHash, Hamming ≤ {args.near_dup_threshold}, "
            f"contra {args.n_pv} imágenes de PlantVillage): **{near['taiwan_id'].nunique()}** "
            f"imágenes de Taiwán, en {len(near)} pares.",
            "- Distancia mínima de cada imagen de Taiwán a PlantVillage (descriptivo): "
            + ", ".join(
                f"{label} {n}"
                for label, n in pd.Series(
                    pd.cut(
                        args.min_dist, [-1, 2, 5, 10, 64], labels=["≤ 2:", "3–5:", "6–10:", "> 10:"]
                    )
                )
                .value_counts(sort=False)
                .items()
            )
            + f" (mínimo {args.min_dist.min()}).",
            f"- Pares casi duplicados **dentro** de Taiwán (mismo umbral; solo informativo, no se "
            f"excluyen): **{within}**.",
            "",
            "## Recuento por carpeta (y si la enfermedad se vio al entrenar)",
            "",
            audit.md_table(counts.reset_index()),
            "",
            "## Exclusiones",
            "",
            audit.md_table(df.loc[df["split"] == "excluded", ["image_path", "exclusion_reason"]])
            if (df["split"] == "excluded").any()
            else "Ninguna.",
            "",
            "## Pares casi duplicados con PlantVillage",
            "",
            audit.md_table(near) if len(near) else "Ninguno.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Lee el inventario, calcula dHash, busca casi-duplicados y escribe manifiesto e informe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=REPO_ROOT / "reports/audit/taiwan/inventory.csv"
    )
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--mapping", type=Path, default=REPO_ROOT / "configs/label_mapping.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/taiwan_metadata.csv")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "reports/data/taiwan_summary.md")
    parser.add_argument("--near-dup-threshold", type=int, default=2, help="como en T02")
    args = parser.parse_args(argv)
    if not args.inventory.is_file():
        raise SystemExit(f"Error: falta {args.inventory}; ejecuta audit_dataset.py --source taiwan")
    inventory = pd.read_csv(args.inventory, keep_default_na=False)
    inventory["readable"] = inventory["readable"].astype(str) == "True"
    inventory["dhash"] = [
        mf.dhash(cv2.imread(str(REPO_ROOT / p))) if ok else ""
        for p, ok in zip(inventory["path"], inventory["readable"], strict=True)
    ]
    meta = pd.read_csv(args.metadata, keep_default_na=False)
    pv = meta[meta["source"] == "plantvillage"].reset_index(drop=True)
    args.n_pv = len(pv)
    df, near = build_manifest(
        inventory, mf.load_label_mapping(args.mapping), pv, args.near_dup_threshold
    )
    readable = df.loc[df["dhash"] != "", "dhash"].tolist()
    within = len(mf.near_duplicate_pairs(readable, args.near_dup_threshold))
    # Descriptivo: distancia mínima de cada imagen de Taiwán a PlantVillage (¿hay casos límite?).
    args.min_dist = mf.hamming_matrix(readable, pv["dhash"].tolist()).min(axis=1)
    df.to_csv(args.out, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(df, near, within, args), encoding="utf-8")
    print(
        f"{(df['split'] == 'external_test').sum()} imágenes en external_test; "
        f"{(df['split'] == 'excluded').sum()} excluidas. Informe: {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
