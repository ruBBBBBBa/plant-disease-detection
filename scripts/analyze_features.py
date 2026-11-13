"""Análisis exploratorio de las características **solo con train** (T04).

Lee ``data/processed/features_v1_subset.csv`` y **descarta todo lo que no sea train** nada más
cargarlo: ``val`` y ``test`` no se miran (protocolo T04). Escribe en ``reports/features/``:

- ``features_train.md``: AUC univariante sano/enfermo de cada característica (ordenado por
  |AUC − 0,5|), resumen por familia, pares muy correlacionados y conclusión factual.
- ``boxplots_train.png``: las 10 características más separadoras, sano frente a enfermo.
- ``spots_train.jpg``: 3 hojas por enfermedad con su máscara de regiones candidatas.

Ejemplo:
    python scripts/analyze_features.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402
from src import features as ft  # noqa: E402
from src import segmentation as seg  # noqa: E402

NON_FEATURES = {"image_id", "split", "binary_label", "disease_name", "leaf_group_id", "in_subset"}
QUALITY = {"leaf_area_frac", "border_touch_frac", "mask_solidity"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Análisis exploratorio con train (T04).")
    parser.add_argument(
        "--features", type=Path, default=REPO_ROOT / "data/processed/features_v1_subset.csv"
    )
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--seg-config", type=Path, default=REPO_ROOT / "configs/segmentation.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/features.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    return parser.parse_args(argv)


def family(name: str) -> str:
    """Familia de una característica a partir de su nombre."""
    if name in QUALITY:
        return "calidad"
    if name.startswith(("glcm_", "lbp_")):
        return "textura"
    if name.startswith("spot_"):
        return "manchas"
    return "color"


def auc_table(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """AUC univariante (1 = enferma) de cada característica y su separación |AUC − 0,5|."""
    labels = train["binary_label"] != "healthy"
    rows = []
    for column in columns:
        auc = roc_auc_score(labels, train[column])
        rows.append(
            {
                "feature": column,
                "family": family(column),
                "auc": round(auc, 3),
                "sep": round(abs(auc - 0.5), 3),
                "median_healthy": round(float(train.loc[~labels, column].median()), 4),
                "median_diseased": round(float(train.loc[labels, column].median()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["sep", "feature"], ascending=[False, True])


def correlated_pairs(train: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    """Pares de características con |r de Pearson| ≥ ``threshold`` (posible redundancia)."""
    corr = train[columns].corr()
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append(
                    {"a": a, "b": b, "family_a": family(a), "family_b": family(b), "r": round(r, 3)}
                )
    return pd.DataFrame(pairs, columns=["a", "b", "family_a", "family_b", "r"]).sort_values(
        "r", key=abs, ascending=False
    )


def boxplots(train: pd.DataFrame, top: list[str], out: Path) -> None:
    """Boxplots sano frente a enfermo de las características más separadoras."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, column in zip(axes.ravel(), top, strict=False):
        data = [train.loc[train["binary_label"] == b, column] for b in ("healthy", "diseased")]
        ax.boxplot(data, tick_labels=["sana", "enferma"], showfliers=False)
        ax.set_title(column, fontsize=9)
    fig.suptitle("train: 10 características con mayor |AUC − 0,5| (sin valores atípicos dibujados)")
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)


def spots_grid(train: pd.DataFrame, args: argparse.Namespace) -> None:
    """3 hojas de train por enfermedad: contorno de hoja (verde) y regiones candidatas (magenta)."""
    metadata = pd.read_csv(args.metadata, keep_default_na=False).set_index("image_id")
    seg_config = json.loads(args.seg_config.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for disease, group in train.groupby("disease_name"):
        tiles = []
        for image_id in group["image_id"].sample(3, random_state=args.seed):
            image = seg.read_image(REPO_ROOT / metadata.loc[image_id, "image_path"])
            mask = ft.prepare_mask(image, ft.leaf_mask(image, seg_config), config["mask_variant"])
            spots = ft.spot_mask(image, mask, config)
            out = image.copy()
            for m, color in ((mask, (0, 255, 0)), (spots, (255, 0, 255))):
                contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(out, contours, -1, color, 1)
            out = cv2.resize(out, (192, 192), interpolation=cv2.INTER_AREA)
            cv2.putText(
                out, disease[:18], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            tiles.append(out)
        rows.append(np.hstack(tiles))
    left, right = rows[: (len(rows) + 1) // 2], rows[(len(rows) + 1) // 2 :]
    right += [np.full_like(rows[0], 255)] * (len(left) - len(right))
    grid = np.hstack([np.vstack(left), np.vstack(right)])
    cv2.imwrite(str(args.out / "spots_train.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def sharpness_diagnostic(train: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Nitidez de la captura: varianza del laplaciano dentro de la hoja, por enfermedad (train).

    Es un **diagnóstico**, no una característica: si las sanas son mucho más nítidas que las
    enfermas, parte de la separación de la textura puede venir de cómo se fotografiaron los lotes
    (enfoque, iluminación, cámara) y no de los síntomas.
    """
    metadata = pd.read_csv(args.metadata, keep_default_na=False).set_index("image_id")
    seg_config = json.loads(args.seg_config.read_text(encoding="utf-8"))
    values = []
    for image_id in train["image_id"]:
        image = seg.read_image(REPO_ROOT / metadata.loc[image_id, "image_path"])
        mask = cv2.erode(ft.leaf_mask(image, seg_config), seg.ellipse_kernel(5)) > 0
        laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        values.append(float(laplacian[mask].var()))
    data = train[["disease_name", "binary_label"]].assign(laplacian_var=values)
    labels = data["binary_label"] != "healthy"
    table = data.groupby("disease_name")["laplacian_var"].median().round(1).reset_index()
    table.attrs["auc"] = round(roc_auc_score(labels, data["laplacian_var"]), 3)
    return table


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el análisis con train y escribe informe y figuras."""
    args = parse_args(argv)
    if not args.features.is_file():
        sys.exit(
            "Error: faltan las características. Ejecuta scripts/extract_features.py --subset-only."
        )
    table = pd.read_csv(args.features)
    train = table[table["split"] == "train"].reset_index(drop=True)  # val/test fuera desde aquí
    del table
    n_inconclusive = int(train["inconclusive"].sum())
    train = train[~train["inconclusive"]].reset_index(drop=True)
    columns = [c for c in train.columns if c not in NON_FEATURES | {"inconclusive"}]
    constant = [c for c in columns if train[c].nunique() <= 1]
    columns = [c for c in columns if c not in constant]

    aucs = auc_table(train, columns)
    by_family = (
        aucs.groupby("family")["sep"]
        .agg(features="size", median_sep="median", max_sep="max")
        .round(3)
        .sort_values("median_sep", ascending=False)
        .reset_index()
    )
    pairs = correlated_pairs(train, columns, args.corr_threshold)
    redundancy = (
        pd.concat([pairs["a"], pairs["b"]]).map(family).value_counts().rename("apariciones")
        if len(pairs)
        else pd.Series(dtype=int, name="apariciones")
    )
    top = aucs["feature"].head(10).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    boxplots(train, top, args.out / "boxplots_train.png")
    spots_grid(train, args)

    sharpness = sharpness_diagnostic(train, args)
    best_symptom = aucs[aucs["family"] != "calidad"].iloc[0]
    quality_aucs = aucs[aucs["family"] == "calidad"]
    lines = [
        "# Características — análisis exploratorio (solo train)",
        "",
        f"`python scripts/analyze_features.py --seed {args.seed}` sobre "
        f"`data/processed/{args.features.name}`. Solo filas `train` del subconjunto "
        f"({len(train)} imágenes; {n_inconclusive} «no concluyentes» excluidas). `val` y `test` "
        "no se leen. AUC: 1 = enferma; `sep` = |AUC − 0,5| (0 = no separa, 0,5 = separa del todo).",
        "",
        f"Características constantes en train (sin información, excluidas del análisis): "
        f"{constant or 'ninguna'}.",
        "",
        "## Resumen por familia",
        "",
        audit.md_table(by_family),
        "",
        "## AUC univariante de cada característica (ordenado por separación)",
        "",
        audit.md_table(aucs),
        "",
        f"## Pares muy correlacionados (|r| ≥ {args.corr_threshold})",
        "",
        f"{len(pairs)} pares. Apariciones por familia: {redundancy.to_dict()}.",
        "",
        audit.md_table(pairs) if len(pairs) else "Ninguno.",
        "",
        "## Conclusión factual (train)",
        "",
        f"- Familia con mayor separación mediana: **{by_family.iloc[0]['family']}** "
        f"({by_family.iloc[0]['median_sep']}); la menor: **{by_family.iloc[-1]['family']}** "
        f"({by_family.iloc[-1]['median_sep']}).",
        f"- Característica más separadora (sin contar calidad): `{best_symptom['feature']}` "
        f"(AUC {best_symptom['auc']}).",
        "- Las características de **calidad** de la máscara no son síntomas; si separan, delatan "
        "diferencias de captura entre clases (encuadre, tamaño de la hoja): "
        + ", ".join(f"`{r.feature}` AUC {r.auc}" for r in quality_aucs.itertuples())
        + ".",
        "- **Diagnóstico de captura** (nitidez = varianza del laplaciano dentro de la hoja, no "
        f"es una característica): AUC sano/enfermo {sharpness.attrs['auc']}. Mediana por "
        "enfermedad: "
        + ", ".join(f"{r.disease_name} {r.laplacian_var}" for r in sharpness.itertuples())
        + ". Si las sanas son mucho más nítidas, la separación de la textura puede deberse en "
        "parte a la captura de cada lote y no a los síntomas.",
        "- Separación univariante no es capacidad de clasificación: no se anticipa ningún "
        "resultado de la fase 4.",
        "",
        "Figuras: `boxplots_train.png` (10 más separadoras), `spots_train.jpg` (3 hojas por "
        "enfermedad: contorno de la hoja en verde, regiones candidatas en magenta; las regiones "
        "candidatas son evidencia visual de tonos anómalos, no la zona que usa el clasificador).",
        "",
    ]
    (args.out / "features_train.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"train: {len(train)} imágenes analizadas; {len(pairs)} pares correlacionados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
