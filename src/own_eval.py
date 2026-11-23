"""Test con fotos propias (T07): validación de anotaciones, tasas por hoja e hipótesis H4–H6.

Todo se fija antes de procesar las fotos (commit congelado). Las fotos solo tienen hojas sanas
(y «alteradas», que se describen aparte), así que se mide el **fallo** en sanas:

    fallo = no concluyente o falso positivo (predicción ``diseased``), sobre todas las unidades.

Así un fondo no sale ganando por abstenerse. Las unidades son la **foto** en modo automático y el
**folíolo** (fila de ``own_rois.csv``) en modo ROI. Las tasas se agregan primero por hoja compuesta
(``leaf_id``) y el bootstrap remuestrea **hojas**, nunca folíolos ni fotos: los folíolos de una
misma hoja no son observaciones independientes (igual que el agrupamiento por hoja de T02).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import external as ex

PHOTO_COLUMNS = ["file", "leaf_id", "background", "label", "notes"]
ROI_COLUMNS = ["file", "leaflet_id", "x", "y", "w", "h"]
LEAFLET_PATTERN = re.compile(r"^f\d+$")

# --------------------------------------------------------------------------- anotaciones


def note_keywords(notes: str) -> list[str]:
    """Palabras clave de ``notes`` (separadas por ``;``, sin espacios sobrantes)."""
    return [k.strip() for k in str(notes).split(";") if k.strip()]


def validate_annotations(
    photos: pd.DataFrame, rois: pd.DataFrame, images_dir: Path, config: dict
) -> list[str]:
    """Problemas de las anotaciones; lista vacía si todo cuadra.

    Solo mira **nombres** de fichero, no abre ninguna foto. Comprueba columnas, que cada fila
    del CSV exista en ``images_dir`` y que no haya fotos sin anotar, valores permitidos de
    ``label``, ``background`` y ``notes``, que cada foto sana tenga al menos una ROI, que cada
    ROI apunte a una foto anotada, con ancho y alto positivos y ``leaflet_id`` ``f<n>`` sin
    repetirse dentro de la foto.
    """
    problems = []
    for name, table, cols in (
        ("own_photos", photos, PHOTO_COLUMNS),
        ("own_rois", rois, ROI_COLUMNS),
    ):
        missing = [c for c in cols if c not in table.columns]
        if missing:
            return [f"{name}.csv: faltan columnas {missing}"]
    on_disk = {
        p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    listed = set(photos["file"])
    problems += [
        f"{f}: en own_photos.csv pero no en {images_dir}" for f in sorted(listed - on_disk)
    ]
    problems += [f"{f}: foto sin anotar en own_photos.csv" for f in sorted(on_disk - listed)]
    problems += [
        f"{f}: repetida en own_photos.csv" for f in photos["file"][photos["file"].duplicated()]
    ]
    for row in photos.itertuples(index=False):
        if row.label not in config["labels"]:
            problems.append(f"{row.file}: label {row.label!r} no permitido")
        if row.background not in config["backgrounds"]:
            problems.append(f"{row.file}: background {row.background!r} no permitido")
        bad = [k for k in note_keywords(row.notes) if k not in config["notes_keywords"]]
        if bad:
            problems.append(f"{row.file}: palabras clave no previstas en notes {bad}")
    with_roi = set(rois["file"])
    healthy = set(photos.loc[photos["label"] == "healthy", "file"])
    problems += [f"{f}: foto sana sin ROI" for f in sorted(healthy - with_roi)]
    problems += [
        f"{f}: ROI de una foto que no está en own_photos.csv" for f in sorted(with_roi - listed)
    ]
    for row in rois.itertuples(index=False):
        if not LEAFLET_PATTERN.match(str(row.leaflet_id)):
            problems.append(f"{row.file}: leaflet_id {row.leaflet_id!r} no es f<n>")
        if row.w <= 0 or row.h <= 0:
            problems.append(f"{row.file} {row.leaflet_id}: ROI con ancho o alto no positivo")
    dup = rois[rois.duplicated(["file", "leaflet_id"], keep=False)]
    problems += [f"{f}: leaflet_id repetido en la foto" for f in sorted(set(dup["file"]))]
    return problems


def background_combinations(photos: pd.DataFrame) -> pd.Series:
    """Cuántas hojas sanas tienen cada combinación de fondos (no se exige tener los tres)."""
    healthy = photos[photos["label"] == "healthy"]
    combos = healthy.groupby("leaf_id")["background"].agg(lambda s: "+".join(sorted(set(s))))
    return combos.value_counts().sort_index()


# --------------------------------------------------------------------------- tasas


def add_outcomes(pred: pd.DataFrame) -> pd.DataFrame:
    """Añade ``fp`` (predicha ``diseased``) y ``fail`` (no concluyente o FP) a cada unidad."""
    out = pred.copy()
    out["fp"] = out["prediction"].eq("diseased") & ~out["inconclusive"]
    out["fail"] = out["inconclusive"] | out["fp"]
    return out


def leaf_rates(pred: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Tasas por hoja (y ``keys``): fallo, no concluyente y FP sobre concluyentes.

    Primer paso de la agregación: cada hoja compuesta pesa lo mismo, tenga los folíolos que tenga.
    ``fp_conclusive`` es NaN si la hoja no tiene ninguna unidad concluyente en ese grupo.
    """
    g = add_outcomes(pred).groupby(["leaf_id", *keys])
    out = g.agg(
        n_units=("fail", "size"),
        fail=("fail", "mean"),
        inconclusive=("inconclusive", "mean"),
        fp_sum=("fp", "sum"),
    )
    conclusive = g["inconclusive"].agg(lambda s: int((~s).sum()))
    out["n_conclusive"] = conclusive
    out["fp_conclusive"] = np.where(
        conclusive > 0, out["fp_sum"] / conclusive.clip(lower=1), np.nan
    )
    return out.drop(columns="fp_sum").reset_index()


def rate_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Tabla descriptiva por variante × modo × fondo (y «todos»): media de las tasas por hoja.

    Cada fila indica cuántas hojas y unidades (fotos o folíolos) entran; la tasa de FP «solo
    concluyentes» va con su cobertura (1 − tasa de no concluyentes).
    """
    healthy = pred[pred["label"] == "healthy"]
    per_bg = leaf_rates(healthy, ["variant", "mode", "background"])
    per_all = leaf_rates(healthy, ["variant", "mode"]).assign(background="todos")
    rows = pd.concat([per_bg, per_all]).groupby(["variant", "mode", "background"])
    table = rows.agg(
        n_leaves=("leaf_id", "nunique"),
        n_units=("n_units", "sum"),
        fail_rate=("fail", "mean"),
        inconclusive_rate=("inconclusive", "mean"),
        fp_rate_conclusive=("fp_conclusive", "mean"),
    )
    table["coverage"] = 1 - table["inconclusive_rate"]
    return table.reset_index()


def keyword_table(pred: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Estratificación descriptiva por palabra clave de ``notes`` (con / sin), sin veredicto."""
    healthy = add_outcomes(pred[pred["label"] == "healthy"])
    rows = []
    for key in keywords:
        has = healthy["notes"].map(lambda n, k=key: k in note_keywords(n))
        for (variant, mode), part in healthy.groupby(["variant", "mode"]):
            for flag in (True, False):
                sub = part[has.loc[part.index] == flag]
                rows.append(
                    {
                        "keyword": key,
                        "present": flag,
                        "variant": variant,
                        "mode": mode,
                        "n_leaves": sub["leaf_id"].nunique(),
                        "n_units": len(sub),
                        "fail_rate": sub["fail"].mean() if len(sub) else np.nan,
                        "inconclusive_rate": sub["inconclusive"].mean() if len(sub) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hipótesis


def paired_leaf_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    """IC 95 % de la media de las diferencias por hoja, remuestreando **hojas** con reemplazo.

    ``diffs`` tiene una entrada por hoja compuesta (ya emparejada): d = tasa(peor) − tasa(mejor).
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) == 0:
        return {"diff": np.nan, "ci95": [np.nan, np.nan], "n_leaves": 0}
    index = np.random.default_rng(seed).integers(0, len(diffs), size=(n_boot, len(diffs)))
    reps = diffs[index].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff": float(diffs.mean()), "ci95": [float(lo), float(hi)], "n_leaves": len(diffs)}


def paired_diffs(pred: pd.DataFrame, spec: dict) -> pd.Series:
    """Diferencias de tasa de fallo por hoja para una hipótesis (solo sanas).

    - ``compare = background``: en ``spec["mode"]``, solo hojas con unidades en los dos fondos.
    - ``compare = mode``: juntando fondos, tasa del modo ``worse`` frente al ``better`` por hoja.
    """
    healthy = pred[pred["label"] == "healthy"]
    if spec["compare"] == "background":
        rates = leaf_rates(healthy[healthy["mode"] == spec["mode"]], ["background"])
        wide = rates.pivot(index="leaf_id", columns="background", values="fail")
    else:
        wide = leaf_rates(healthy, ["mode"]).pivot(index="leaf_id", columns="mode", values="fail")
    if spec["worse"] not in wide or spec["better"] not in wide:
        return pd.Series(dtype=float)
    wide = wide[[spec["worse"], spec["better"]]].dropna()
    return wide[spec["worse"]] - wide[spec["better"]]


def evaluate_hypotheses(pred: pd.DataFrame, config: dict) -> dict:
    """Veredicto de H4–H6 con la variante por defecto y la regla de T06 (``ex.verdict``)."""
    boot = config["bootstrap"]
    part = pred[pred["variant"] == config["default_variant"]]
    out = {}
    for name, spec in config["hypotheses"].items():
        diffs = paired_diffs(part, spec)
        res = paired_leaf_bootstrap(diffs.to_numpy(), boot["n_boot"], boot["seed"])
        n_units = part[part["leaf_id"].isin(diffs.index) & (part["label"] == "healthy")]
        if spec["compare"] == "background":
            n_units = n_units[
                (n_units["mode"] == spec["mode"])
                & n_units["background"].isin([spec["worse"], spec["better"]])
            ]
        res["n_units"] = int(len(n_units))
        res["verdict"] = ex.verdict(res["ci95"]) if res["n_leaves"] else "sin datos"
        out[name] = spec | res
    return out
