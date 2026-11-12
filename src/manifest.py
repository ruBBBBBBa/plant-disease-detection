"""Manifiesto, grupos por hoja y particiones sin fugas (T02).

Pasos (ver ``scripts/prepare_dataset.py``):

1. Mapeo explícito carpeta → ``disease_name`` / ``binary_label`` (``configs/label_mapping.csv``).
2. ``leaf_group_id``: hoja según ``leaf-map.json`` (D009); sin hoja conocida → ``excluded`` (D006).
3. Casi-duplicados con un *hash* perceptual (dHash) y distancia de Hamming; los duplicados
   exactos (mismo SHA-1) y los casi-duplicados se unen al mismo grupo con *union-find*.
4. Partición por grupos, estratificada por enfermedad (70/15/15 por defecto).
5. Subconjunto inicial equilibrado con hojas completas (D007).
6. Comprobación de fugas.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data_audit import resolve_leaf_id

EVAL_SPLITS = ("train", "val", "test", "external_test")


# --------------------------------------------------------------------------- mapeo de etiquetas


def load_label_mapping(path: Path) -> pd.DataFrame:
    """Lee ``configs/label_mapping.csv`` (una fila por ``source`` + ``folder``)."""
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False)
    if mapping.duplicated(["source", "folder"]).any():
        raise ValueError(f"{path}: hay carpetas repetidas en el mapeo")
    return mapping


def apply_label_mapping(inventory: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Añade ``species``, ``disease_name`` y ``binary_label`` según el mapeo explícito.

    Una carpeta sin fila en el mapeo es un **error** (no se ignora en silencio).
    """
    merged = inventory.merge(
        mapping[["source", "folder", "species", "disease_name", "binary_label"]],
        left_on=["source", "class_name"],
        right_on=["source", "folder"],
        how="left",
    ).drop(columns="folder")
    unmapped = merged.loc[merged["binary_label"].isna(), ["source", "class_name"]]
    if len(unmapped):
        pairs = sorted({f"{s}/{c}" for s, c in unmapped.itertuples(index=False)})
        raise ValueError("Carpetas sin mapear en configs/label_mapping.csv: " + ", ".join(pairs))
    return merged


# --------------------------------------------------------------------------- hojas


def target_spot_leaf_map(csv_path: Path) -> dict[str, str]:
    """Número de foto → hoja para *Target_Spot*, a partir de su CSV oficial.

    El CSV lista ``Com.G_FL nnnn`` pero los ficheros se llaman ``Com.G_TgS_FL nnnn``, por eso
    ``leaf-map.json`` no los encuentra. Aquí se indexa por el número ``nnnn`` (como texto, con
    sus ceros a la izquierda). El identificador sigue el formato de ``leaf-map.json``.
    """
    leaves = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            match = re.fullmatch(r"Com\.G_FL (\d+)\.\w+", row["File Name"].strip(), re.IGNORECASE)
            if match:
                leaves[match.group(1)] = f"Tomato___Target_Spot:::{row['Leaf #']}"
    return leaves


def target_spot_number(file_name: str) -> str | None:
    """Número ``nnnn`` de un fichero ``…___Com.G_TgS_FL nnnn.JPG`` (o ``None``)."""
    match = re.search(r"Com\.G_TgS_FL (\d+)\.\w+$", file_name, re.IGNORECASE)
    return match.group(1) if match else None


def assign_leaf_ids(
    df: pd.DataFrame, leaf_map: dict[str, list[str]], target_spot: dict[str, str] | None
) -> pd.Series:
    """``leaf_id`` de cada imagen de PlantVillage (``""`` si no se conoce).

    Usa ``resolve_leaf_id`` (D009) y, solo para *Target_Spot*, el mapa por número
    (``target_spot`` = ``None`` si la correspondencia 1:1 no se cumplió).
    """
    ids = []
    for path, class_name in zip(df["image_path"], df["class_name"], strict=True):
        name = Path(path).name
        leaf_id = resolve_leaf_id(name, class_name, leaf_map)
        if not leaf_id and class_name == "Tomato___Target_Spot" and target_spot:
            leaf_id = target_spot.get(target_spot_number(name) or "", "")
        ids.append(leaf_id)
    return pd.Series(ids, index=df.index)


# --------------------------------------------------------------------------- casi-duplicados


def dhash(image_bgr: np.ndarray) -> str:
    """*Difference hash* de 64 bits como 16 caracteres hexadecimales.

    Se pasa a gris y se reduce a 9×8 píxeles (``INTER_AREA``); cada bit indica si un píxel es
    más claro que su vecino de la izquierda. Imágenes casi iguales (recompresión, pequeños
    cambios de brillo) dan *hashes* a poca distancia de Hamming. Es una técnica clásica de
    gradiente horizontal sobre una imagen muy suavizada, sin aprendizaje.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return np.packbits(bits.ravel()).tobytes().hex()


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(hashes: list[str], threshold: int, chunk: int = 500) -> np.ndarray:
    """Pares ``(i, j, distancia)`` con ``i < j`` y distancia de Hamming ``<= threshold``.

    Compara todos contra todos por bloques: XOR byte a byte y recuento de bits con una tabla
    de 256 entradas (vale para cualquier versión de NumPy).
    """
    codes = np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    found = []
    for start in range(0, len(codes), chunk):
        block = codes[start : start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ codes[None, :, :]].sum(axis=2, dtype=np.int32)
        i, j = np.nonzero(dist <= threshold)
        i = i + start
        keep = i < j
        found.append(np.column_stack([i[keep], j[keep], dist[i[keep] - start, j[keep]]]))
    return np.concatenate(found) if found else np.empty((0, 3), dtype=int)


class UnionFind:
    """Estructura *union-find* (conjuntos disjuntos) para fusionar grupos."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        """Representante del conjunto de ``item`` (con compresión de caminos)."""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        """Une los conjuntos de ``a`` y ``b``; el representante es el menor (determinista)."""
        ra, rb = sorted((self.find(a), self.find(b)))
        self.parent[rb] = ra


def merge_groups(df: pd.DataFrame, pairs: np.ndarray) -> pd.Series:
    """Fusiona ``leaf_group_id`` de duplicados exactos (SHA-1) y de los pares casi-duplicados.

    ``pairs`` son índices posicionales de ``df``. Devuelve el grupo final de cada fila.
    """
    uf = UnionFind(sorted(df["leaf_group_id"].unique()))
    for _, group in df.groupby("sha1"):
        first = group["leaf_group_id"].iloc[0]
        for other in group["leaf_group_id"].iloc[1:]:
            uf.union(first, other)
    groups = df["leaf_group_id"].to_numpy()
    for i, j, _ in pairs:
        uf.union(groups[i], groups[j])
    return df["leaf_group_id"].map(uf.find)


# --------------------------------------------------------------------------- partición


def group_strata(df: pd.DataFrame) -> pd.Series:
    """Enfermedad que representa a cada grupo (la más frecuente; empate → orden alfabético)."""
    return df.groupby("leaf_group_id")["disease_name"].agg(lambda s: sorted(s.mode())[0])


def split_groups(
    df: pd.DataFrame, fractions: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Asigna cada ``leaf_group_id`` a ``train``/``val``/``test``, estratificando por enfermedad.

    Para cada enfermedad se barajan sus grupos (semilla fija) y se recorren acumulando imágenes:
    van a ``train`` hasta el 70 % de las imágenes de esa enfermedad, luego a ``val`` hasta el
    85 % y el resto a ``test``. Una hoja nunca se reparte entre splits.
    """
    sizes = df.groupby("leaf_group_id").size()
    strata = group_strata(df)
    rng = random.Random(seed)
    assignment = {}
    cut_train, cut_val = fractions[0], fractions[0] + fractions[1]
    for disease in sorted(strata.unique()):
        groups = sorted(strata.index[strata == disease])
        rng.shuffle(groups)
        total, seen = sizes[groups].sum(), 0
        for group in groups:
            position = seen / total
            assignment[group] = (
                "train" if position < cut_train else "val" if position < cut_val else "test"
            )
            seen += sizes[group]
    return assignment


def per_disease_targets(diseases: list[str], total: int, seed: int) -> dict[str, int]:
    """Reparte ``total`` imágenes a partes iguales entre ``diseases``.

    El resto de la división (p. ej. 500 = 8 × 62 + 4) se da, una imagen más, a enfermedades
    elegidas al azar con la semilla.
    """
    base, extra = divmod(total, len(diseases))
    lucky = set(random.Random(seed).sample(sorted(diseases), extra))
    return {d: base + (d in lucky) for d in sorted(diseases)}


def split_targets(n: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    """Reparte ``n`` imágenes en ``train``/``val``/``test`` (redondeo; test se queda el resto)."""
    train, val = round(n * fractions[0]), round(n * fractions[1])
    return {"train": train, "val": val, "test": n - train - val}


def select_subset(
    df: pd.DataFrame,
    n_healthy: int | None,
    n_diseased: int | None,
    fractions: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    """Marca ``in_subset`` con hojas completas, sin superar ningún objetivo (D007).

    - Sanas: ``n_healthy=None`` → todas las disponibles; con un número, se reparte 70/15/15.
    - Enfermas, split a split: ``n_diseased=None`` → tantas como sanas elegidas en ese split;
      con un número, se reparte 70/15/15. El total del split se divide a partes iguales entre
      enfermedades (``per_disease_targets``).
    - En cada celda (enfermedad, split) se barajan las hojas (semilla) y se añade una hoja entera
      si cabe sin pasarse. Lo que una enfermedad no llena (las hojas tienen 3–4 fotos) pasa a la
      siguiente, así que el total del split queda a pocas imágenes del objetivo.
    """
    sizes = df.groupby("leaf_group_id").size()
    strata = group_strata(df)
    group_split = df.groupby("leaf_group_id")["split"].first()
    rng = random.Random(seed)
    chosen: set[str] = set()

    def take(disease: str, split: str, target: int) -> int:
        """Añade hojas completas de (enfermedad, split) sin superar ``target``; devuelve cuántas."""
        groups = sorted(strata.index[(strata == disease) & (group_split == split)])
        rng.shuffle(groups)
        count = 0
        for group in groups:
            if count + sizes[group] <= target:
                chosen.add(group)
                count += sizes[group]
        return count

    available = df[df["disease_name"] == "healthy"]["split"].value_counts()
    healthy_targets = (
        {s: int(available.get(s, 0)) for s in ("train", "val", "test")}
        if n_healthy is None
        else split_targets(n_healthy, fractions)
    )
    healthy_taken = {s: take("healthy", s, t) for s, t in healthy_targets.items()}

    diseases = sorted(set(df["disease_name"]) - {"healthy"})
    diseased_totals = healthy_taken if n_diseased is None else split_targets(n_diseased, fractions)
    for split, total in diseased_totals.items():
        carry = 0
        for disease, share in per_disease_targets(diseases, total, seed).items():
            target = share + carry
            carry = target - take(disease, split, target)
    return df["leaf_group_id"].isin(chosen)


# --------------------------------------------------------------------------- fugas


def check_leakage(manifest: pd.DataFrame, threshold: int) -> list[str]:
    """Devuelve la lista de fugas encontradas (vacía si no hay ninguna).

    Solo cuentan las filas con split de evaluación (``excluded`` no participa). Fuga si:
    un ``leaf_group_id`` o un SHA-1 aparece en más de un split; dos imágenes de splits
    distintos son casi-duplicadas (dHash a distancia ``<= threshold``); o una imagen de
    PlantDoc está fuera de ``external_test``.
    """
    used = manifest[manifest["split"].isin(EVAL_SPLITS)].reset_index(drop=True)
    problems = []
    for column in ("leaf_group_id", "sha1"):
        n_splits = used.groupby(column)["split"].nunique()
        for value in n_splits.index[n_splits > 1]:
            problems.append(f"{column} {value} en varios splits")
    splits = used["split"].to_numpy()
    for i, j, d in near_duplicate_pairs(used["dhash"].tolist(), threshold):
        if splits[i] != splits[j]:
            problems.append(
                f"casi-duplicados (d={d}) en {splits[i]}/{splits[j]}: "
                f"{used['image_path'][i]} | {used['image_path'][j]}"
            )
    plantdoc = manifest[(manifest["source"] == "plantdoc") & (manifest["split"] != "excluded")]
    for path in plantdoc.loc[plantdoc["split"] != "external_test", "image_path"]:
        problems.append(f"PlantDoc fuera de external_test: {path}")
    return problems


def image_id(source: str, image_path: str) -> str:
    """Identificador estable: prefijo de fuente + 12 hex del SHA-1 de la ruta relativa."""
    prefix = {"plantvillage": "pv", "plantdoc": "pd"}[source]
    return f"{prefix}_{hashlib.sha1(image_path.encode('utf-8')).hexdigest()[:12]}"
