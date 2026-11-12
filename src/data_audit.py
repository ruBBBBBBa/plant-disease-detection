"""Auditoría de un dataset de imágenes organizado en carpetas (T01).

Suposición explícita: **carpeta = clase**. Cada imagen se describe con lo más básico de una
imagen digital (alto, ancho, número de canales, tamaño en bytes) y con un resumen SHA-1 de su
contenido, que permite detectar duplicados *exactos* (mismos bytes). Los casi-duplicados
(hash perceptual) quedan para T02.

Nada de este módulo escribe en la carpeta del dataset: solo lee.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

INVENTORY_COLUMNS = [
    "path",
    "class_name",
    "original_split",
    "extension",
    "readable",
    "height",
    "width",
    "channels",
    "bytes",
    "sha1",
]


# --------------------------------------------------------------------------- recorrido


def find_class_dirs(root: Path, source: str, species: str) -> list[tuple[Path, str]]:
    """Devuelve las carpetas de clase a auditar como pares ``(carpeta, split_original)``.

    - PlantVillage: ``<root>/raw/color/<clase>/`` (sin split original → cadena vacía).
    - PlantDoc: ``<root>/{train,test}/<clase>/`` (el split original es el nombre de la carpeta).

    Solo se conservan las clases cuyo nombre contiene ``species`` sin distinguir mayúsculas.
    """
    if source == "plantvillage":
        parents = [(root / "raw" / "color", "")]
    elif source == "plantdoc":
        parents = [(root / "train", "train"), (root / "test", "test")]
    else:
        raise ValueError(f"Fuente desconocida: {source!r}")

    class_dirs = []
    for parent, split in parents:
        if not parent.is_dir():
            continue
        for class_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
            if species.lower() in class_dir.name.lower():
                class_dirs.append((class_dir, split))
    return class_dirs


def describe_file(path: Path) -> dict:
    """Lee un fichero y devuelve sus propiedades básicas y el SHA-1 de su contenido.

    Se leen los bytes una sola vez: con ellos se calcula el SHA-1 y se decodifica la imagen con
    ``cv2.imdecode`` (el mismo decodificador que ``cv2.imread``). ``IMREAD_UNCHANGED`` conserva
    el número real de canales (1 = gris, 3 = BGR, 4 = BGRA). Si OpenCV no puede decodificarla,
    ``readable`` es ``False`` y las dimensiones quedan vacías.
    """
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    readable = image is not None
    return {
        "extension": path.suffix.lower(),
        "readable": readable,
        "height": image.shape[0] if readable else None,
        "width": image.shape[1] if readable else None,
        "channels": (1 if image.ndim == 2 else image.shape[2]) if readable else None,
        "bytes": len(data),
        "sha1": hashlib.sha1(data).hexdigest(),
    }


def build_inventory(class_dirs: Iterable[tuple[Path, str]], base: Path) -> pd.DataFrame:
    """Construye el inventario: una fila por fichero, con la ruta relativa a ``base``."""
    rows = []
    for class_dir, split in class_dirs:
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            rows.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "class_name": class_dir.name,
                    "original_split": split,
                    **describe_file(path),
                }
            )
    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    # Enteros "nullable": las imágenes ilegibles no tienen dimensiones.
    return inventory.astype({"height": "Int64", "width": "Int64", "channels": "Int64"})


# --------------------------------------------------------------------------- estadísticas


def class_counts(inventory: pd.DataFrame) -> pd.DataFrame:
    """Recuentos por clase (y por split original si lo hay), con ficheros ilegibles."""
    return (
        inventory.groupby(["class_name", "original_split"])
        .agg(files=("path", "size"), unreadable=("readable", lambda s: int((~s).sum())))
        .reset_index()
    )


def resolution_stats(inventory: pd.DataFrame, top: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Mín/mediana/máx de alto y ancho, y las ``top`` resoluciones ``AnchoxAlto`` más frecuentes."""
    readable = inventory[inventory["readable"]]
    stats = readable[["width", "height"]].astype(float).agg(["min", "median", "max"])
    sizes = readable["width"].astype(str) + "x" + readable["height"].astype(str)
    return stats, sizes.value_counts().head(top)


def exact_duplicates(inventory: pd.DataFrame) -> pd.DataFrame:
    """Grupos de ficheros con el mismo SHA-1, clasificados en «misma clase» o «entre clases».

    Devuelve una fila por grupo con el número de ficheros, las clases implicadas y las rutas.
    """
    groups = []
    for sha1, group in inventory.groupby("sha1"):
        if len(group) < 2:
            continue
        classes = sorted(group["class_name"].unique())
        groups.append(
            {
                "sha1": sha1,
                "files": len(group),
                "scope": "same_class" if len(classes) == 1 else "cross_class",
                "classes": " | ".join(classes),
                "paths": " | ".join(group["path"]),
            }
        )
    return pd.DataFrame(groups, columns=["sha1", "files", "scope", "classes", "paths"])


def cross_source_duplicates(inventory: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Pares de ficheros idénticos (mismo SHA-1) entre dos inventarios de fuentes distintas."""
    return inventory[["sha1", "path", "class_name"]].merge(
        other[["sha1", "path", "class_name"]], on="sha1", suffixes=("", "_other")
    )


def segmented_stem(name: str) -> str:
    """Nombre base de un fichero segmentado de PlantVillage (quita ``_final_masked``)."""
    stem = Path(name).stem
    return stem[: -len("_final_masked")] if stem.endswith("_final_masked") else stem


def pair_color_segmented(color_dir: Path, segmented_dir: Path) -> pd.DataFrame:
    """Empareja ``color/<clase>`` con ``segmented/<clase>`` por nombre base, clase a clase.

    Ejemplo: ``abc___RS_HL 0105.JPG`` ↔ ``abc___RS_HL 0105_final_masked.jpg``.
    """
    rows = []
    for class_dir in sorted(p for p in color_dir.iterdir() if p.is_dir()):
        color = {Path(p.name).stem for p in class_dir.iterdir() if p.is_file()}
        seg_dir = segmented_dir / class_dir.name
        seg = (
            {segmented_stem(p.name) for p in seg_dir.iterdir() if p.is_file()}
            if seg_dir.is_dir()
            else set()
        )
        rows.append(
            {
                "class_name": class_dir.name,
                "paired": len(color & seg),
                "only_color": len(color - seg),
                "only_segmented": len(seg - color),
            }
        )
    return pd.DataFrame(rows, columns=["class_name", "paired", "only_color", "only_segmented"])


# --------------------------------------------------------------------------- agrupación por hoja


def leaf_lookup_key(file_name: str) -> str:
    """Clave para buscar una imagen de PlantVillage en ``leaf-map.json``.

    Reproduce la lógica de ``plant_village.py`` (script oficial de Hugging Face): quita
    ``_final_masked``, se queda con lo que va tras el último ``___``, corta en ``copy``,
    elimina la extensión y pasa a minúsculas. P. ej. ``abc___RS_HL 0105.JPG`` → ``rs_hl 0105``.
    """
    key = file_name.replace("_final_masked", "")
    if "___" in key:
        key = key.split("___")[-1]
    key = key.split("copy")[0]
    key = re.sub(r"\.(jpg|png)$", "", key.strip(), flags=re.IGNORECASE)
    return key.strip().lower()


def resolve_leaf_id(file_name: str, class_name: str, leaf_map: dict[str, list[str]]) -> str:
    """Identificador de hoja de una imagen según ``leaf-map.json``, o ``""`` si no se encuentra.

    El mapa asocia cada clave a una lista de candidatos ``"<clase>:::<nº hoja>"``. Como en
    ``plant_village.py``: con un candidato se usa ese; con varios, el que contiene la clase;
    si ninguno encaja (o la clave no existe) la imagen queda sin hoja conocida.
    """
    candidates = leaf_map.get(leaf_lookup_key(file_name), [])
    if len(candidates) == 1:
        return candidates[0]
    matches = [c for c in candidates if class_name in c]
    return matches[0] if matches else ""


def load_leaf_map(path: Path) -> dict[str, list[str]]:
    """Carga ``leaf-map.json`` (clave de imagen → lista de candidatos ``clase:::hoja``)."""
    return json.loads(path.read_text(encoding="utf-8"))


def leaf_group_stats(inventory: pd.DataFrame, leaf_map: dict[str, list[str]]) -> pd.DataFrame:
    """Por clase: cobertura del mapa de hojas, nº de hojas e imágenes por hoja (mín/mediana/máx).

    También cuenta cuántas claves distintas hay (la parte del nombre tras ``___``): si hay
    tantas claves como imágenes, esa parte identifica la **foto**, no la hoja.
    """
    df = inventory[["path", "class_name"]].copy()
    names = df["path"].map(lambda p: Path(p).name)
    df["key"] = names.map(leaf_lookup_key)
    df["leaf_id"] = [
        resolve_leaf_id(n, c, leaf_map) for n, c in zip(names, df["class_name"], strict=True)
    ]
    rows = []
    for class_name, group in df.groupby("class_name"):
        known = group[group["leaf_id"] != ""]
        per_leaf = known.groupby("leaf_id").size()
        rows.append(
            {
                "class_name": class_name,
                "images": len(group),
                "distinct_keys": group["key"].nunique(),
                "with_leaf_id": len(known),
                "leaves": len(per_leaf),
                "imgs_per_leaf_min": int(per_leaf.min()) if len(per_leaf) else 0,
                "imgs_per_leaf_median": float(per_leaf.median()) if len(per_leaf) else 0.0,
                "imgs_per_leaf_max": int(per_leaf.max()) if len(per_leaf) else 0,
            }
        )
    return pd.DataFrame(rows)


def svm_split_leakage(
    mapping_files: dict[str, Path], leaf_map: dict[str, list[str]], species: str
) -> dict[str, int]:
    """Comprueba si la partición de ``data_distribution_for_SVM`` respeta la agrupación por hoja.

    Cada línea de ``<split>_mapping.txt`` es ``raw/color/<clase>/<fichero>\\t<destino>``.
    Devuelve, para la especie dada, cuántas imágenes hay en cada split, cuántas tienen hoja
    conocida y cuántas hojas aparecen a la vez en ``train`` y ``test`` (fuga si > 0). Las
    imágenes sin hoja conocida no se pueden comprobar.
    """
    leaves: dict[str, set[str]] = {}
    result: dict[str, int] = {}
    for split, path in mapping_files.items():
        leaves[split] = set()
        total = with_leaf = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            source_path = line.split("\t")[0]
            parts = source_path.split("/")
            if len(parts) < 4 or species.lower() not in parts[2].lower():
                continue
            total += 1
            leaf_id = resolve_leaf_id(parts[3], parts[2], leaf_map)
            if leaf_id:
                with_leaf += 1
                leaves[split].add(leaf_id)
        result[f"{split}_images"] = total
        result[f"{split}_with_leaf_id"] = with_leaf
        result[f"{split}_leaves"] = len(leaves[split])
    result["leaves_in_both"] = len(leaves.get("train", set()) & leaves.get("test", set()))
    return result


def md_table(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown sencilla (sin dependencias extra)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


# --------------------------------------------------------------------------- ejemplos visuales


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Reduce la imagen para que quepa en ``size × size`` sin deformarla y rellena de gris."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    scale = size / max(image.shape[:2])
    new_w, new_h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    # INTER_AREA es la interpolación recomendada para reducir (promedia y evita aliasing).
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    y, x = (size - new_h) // 2, (size - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def sample_grid(
    paths: list[Path], n: int, seed: int, thumb: int = 160, cols: int = 4
) -> np.ndarray | None:
    """Rejilla BGR con ``n`` imágenes elegidas al azar (semilla fija) entre ``paths``.

    Devuelve ``None`` si no hay ninguna imagen legible.
    """
    chosen = random.Random(seed).sample(sorted(paths), min(n, len(paths)))
    thumbs = [letterbox(img, thumb) for p in chosen if (img := cv2.imread(str(p))) is not None]
    if not thumbs:
        return None
    rows = -(-len(thumbs) // cols)  # división entera hacia arriba
    grid = np.full((rows * thumb, cols * thumb, 3), 255, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * thumb : (r + 1) * thumb, c * thumb : (c + 1) * thumb] = t
    return grid


def slugify(name: str) -> str:
    """Convierte un nombre de clase en un nombre de fichero seguro (``a-z0-9_``)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
