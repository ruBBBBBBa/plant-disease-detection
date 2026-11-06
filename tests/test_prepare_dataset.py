"""Tests de T02 (mapeo, casi-duplicados, grupos, splits, subconjunto, fugas); datos sintéticos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts import prepare_dataset
from src import manifest as mf


def noise_image(seed: int, size: int = 64) -> np.ndarray:
    """Imagen BGR de ruido suavizado: cada semilla da una imagen distinta (dHash muy diferente)."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (9, 9), 0)


def toy_frame(n_groups: int = 20, per_group: int = 4) -> pd.DataFrame:
    """Imágenes de dos enfermedades, ``per_group`` fotos por hoja."""
    rows = []
    for g in range(n_groups):
        disease = "healthy" if g % 2 == 0 else "early_blight"
        for k in range(per_group):
            rows.append({"leaf_group_id": f"leaf{g:02d}", "disease_name": disease, "k": k})
    return pd.DataFrame(rows)


def test_unmapped_folder_is_an_error() -> None:
    """Una carpeta sin fila en el mapeo detiene el proceso (no se ignora en silencio)."""
    inventory = pd.DataFrame({"source": ["plantdoc"], "class_name": ["Tomato raro"]})
    mapping = pd.DataFrame(
        {
            "source": ["plantdoc"],
            "folder": ["Tomato leaf"],
            "species": ["tomato"],
            "disease_name": ["healthy"],
            "binary_label": ["healthy"],
        }
    )
    with pytest.raises(ValueError, match="Tomato raro"):
        mf.apply_label_mapping(inventory, mapping)


def test_dhash_detects_near_duplicates() -> None:
    """Un cambio leve de brillo + recompresión JPEG queda cerca; otra imagen queda lejos."""
    original = noise_image(1)
    brighter = cv2.convertScaleAbs(
        original, alpha=1.0, beta=8
    )  # suma con saturación (sin desbordar)
    _, jpeg = cv2.imencode(".jpg", brighter, [cv2.IMWRITE_JPEG_QUALITY, 70])
    recompressed = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    hashes = [mf.dhash(original), mf.dhash(recompressed), mf.dhash(noise_image(2))]

    pairs = mf.near_duplicate_pairs(hashes, threshold=4)
    assert [tuple(p[:2]) for p in pairs] == [(0, 1)]


def test_merge_groups_by_sha1_and_near_pairs() -> None:
    """Mismo SHA-1 o par casi-duplicado → mismo grupo final; el resto no se toca."""
    df = pd.DataFrame(
        {"leaf_group_id": ["a", "b", "c", "d"], "sha1": ["x", "x", "y", "z"]},
    )
    merged = mf.merge_groups(df, np.array([[2, 3, 1]]))
    assert merged.tolist() == ["a", "a", "c", "c"]


def test_split_groups_is_deterministic_grouped_and_stratified() -> None:
    """Misma semilla → mismo reparto; una hoja nunca cae en dos splits; cada enfermedad en todos."""
    df = toy_frame()
    a = mf.split_groups(df, (0.7, 0.15, 0.15), seed=42)
    assert a == mf.split_groups(df, (0.7, 0.15, 0.15), seed=42)
    df["split"] = df["leaf_group_id"].map(a)
    assert (df.groupby("leaf_group_id")["split"].nunique() == 1).all()
    by_disease = df.groupby("disease_name")["split"].value_counts().unstack()
    assert set(by_disease.columns) == {"train", "val", "test"}
    assert (by_disease["train"] > by_disease["val"]).all()


def test_select_subset_uses_whole_leaves_without_exceeding_targets() -> None:
    """El subconjunto nunca supera el objetivo de una celda y toma hojas completas."""
    df = toy_frame(n_groups=40)
    df["split"] = df["leaf_group_id"].map(mf.split_groups(df, (0.7, 0.15, 0.15), seed=0))
    df["in_subset"] = mf.select_subset(df, 30, 30, (0.7, 0.15, 0.15), seed=0)
    chosen = df[df["in_subset"]]
    assert (df.groupby("leaf_group_id")["in_subset"].nunique() == 1).all()
    for disease in ("healthy", "early_blight"):
        counts = chosen[chosen["disease_name"] == disease]["split"].value_counts()
        assert counts.get("train", 0) <= 21 and counts.get("val", 0) <= 4
        assert counts.sum() <= 30


def test_per_disease_targets_split_remainder() -> None:
    """500 entre 8 enfermedades: cuatro reciben 63 y cuatro 62; total exacto."""
    targets = mf.per_disease_targets([f"d{i}" for i in range(8)], 500, seed=42)
    assert sum(targets.values()) == 500
    assert sorted(set(targets.values())) == [62, 63]


def test_target_spot_number_and_csv(tmp_path: Path) -> None:
    """El número de ``Com.G_TgS_FL nnnn`` indexa la hoja del CSV ``Com.G_FL nnnn``."""
    csv_path = tmp_path / "ts.csv"
    csv_path.write_text('"File Name","Leaf #"\n"Com.G_FL 0004.JPG","1.0"\n')
    assert mf.target_spot_number("uuid___Com.G_TgS_FL 0004.JPG") == "0004"
    assert mf.target_spot_number("uuid___RS_HL 0004.JPG") is None
    assert mf.target_spot_leaf_map(csv_path) == {"0004": "Tomato___Target_Spot:::1.0"}


def leak_frame() -> pd.DataFrame:
    """Manifiesto mínimo sin fugas."""
    return pd.DataFrame(
        {
            "image_path": ["a", "b", "c", "d"],
            "source": ["plantvillage", "plantvillage", "plantdoc", "plantvillage"],
            "leaf_group_id": ["g1", "g2", "p1", "g3"],
            "split": ["train", "test", "external_test", "excluded"],
            "sha1": ["s1", "s2", "s3", "s1"],
            "dhash": [mf.dhash(noise_image(i)) for i in range(4)],
        }
    )


def test_check_leakage_clean_and_detects_each_kind() -> None:
    """Sin fugas → lista vacía; cada tipo de fuga se detecta. ``excluded`` no cuenta."""
    clean = leak_frame()
    assert mf.check_leakage(clean, threshold=2) == []

    same_group = clean.assign(leaf_group_id=["g1", "g1", "p1", "g3"])
    assert any("leaf_group_id" in p for p in mf.check_leakage(same_group, 2))

    same_sha = clean.assign(sha1=["s1", "s1", "s3", "s4"])
    assert any("sha1" in p for p in mf.check_leakage(same_sha, 2))

    near = clean.copy()
    near.loc[1, "dhash"] = near.loc[0, "dhash"]
    assert any("casi-duplicados" in p for p in mf.check_leakage(near, 2))

    plantdoc_in_train = clean.assign(split=["train", "test", "train", "excluded"])
    assert any("PlantDoc" in p for p in mf.check_leakage(plantdoc_in_train, 2))


@pytest.fixture
def fake_raw(tmp_path: Path) -> dict[str, Path]:
    """Mini PlantVillage (hojas de 2 fotos + 1 sin hoja) y mini PlantDoc (copia en conflicto)."""
    pv = tmp_path / "plantvillage"
    leaf_map: dict[str, list[str]] = {}
    seed = 0
    for folder in ("Tomato___healthy", "Tomato___Early_blight"):
        d = pv / "raw" / "color" / folder
        d.mkdir(parents=True)
        for leaf in range(10):
            for shot in range(2):
                name = f"L{leaf}S{shot} {seed:04d}"
                cv2.imwrite(str(d / f"u{seed}___{name}.JPG"), noise_image(seed))
                leaf_map[name.lower()] = [f"{folder}:::{leaf}.0"]
                seed += 1
    cv2.imwrite(
        str(pv / "raw" / "color" / "Tomato___healthy" / "x___NOMAP 1.JPG"), noise_image(999)
    )
    (pv / "leaf-map.json").write_text(json.dumps(leaf_map))

    pdoc = tmp_path / "plantdoc"
    for split, folder in (("train", "Tomato leaf"), ("test", "Tomato Early blight leaf")):
        (pdoc / split / folder).mkdir(parents=True)
    cv2.imwrite(str(pdoc / "train" / "Tomato leaf" / "a.jpg"), noise_image(500))
    cv2.imwrite(str(pdoc / "train" / "Tomato leaf" / "v8.jpg"), noise_image(501))
    shutil.copy(
        pdoc / "train" / "Tomato leaf" / "v8.jpg", pdoc / "test" / "Tomato Early blight leaf"
    )

    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "source,folder,species,disease_name,binary_label,notes\n"
        "plantvillage,Tomato___healthy,tomato,healthy,healthy,\n"
        "plantvillage,Tomato___Early_blight,tomato,early_blight,diseased,\n"
        "plantdoc,Tomato leaf,tomato,healthy,healthy,\n"
        "plantdoc,Tomato Early blight leaf,tomato,early_blight,diseased,\n"
    )
    return {"pv": pv, "pd": pdoc, "mapping": mapping, "tmp": tmp_path}


def run_prepare(paths: dict[str, Path], out_name: str) -> pd.DataFrame:
    """Ejecuta el script sobre el dataset sintético y devuelve el manifiesto."""
    out = paths["tmp"] / out_name
    args = [
        "--plantvillage-root", str(paths["pv"]),
        "--plantdoc-root", str(paths["pd"]),
        "--mapping", str(paths["mapping"]),
        "--out", str(out),
        "--reports", str(paths["tmp"] / "reports"),
        "--subset-healthy", "6",
        "--subset-diseased", "6",
    ]  # fmt: skip
    assert prepare_dataset.main(args) == 0
    return pd.read_csv(out, keep_default_na=False)


def test_prepare_end_to_end(fake_raw: dict[str, Path]) -> None:
    """Manifiesto: exclusiones, PlantDoc sin duplicados y con conflicto, splits deterministas."""
    m = run_prepare(fake_raw, "m1.csv")

    nomap = m[m["image_path"].str.contains("NOMAP")].iloc[0]
    assert (nomap["split"], nomap["exclusion_reason"]) == ("excluded", "no_leaf_id")

    pdoc = m[m["source"] == "plantdoc"]
    assert set(pdoc["split"]) == {"external_test", "excluded"}
    kept = pdoc[pdoc["image_path"].str.contains("v8") & (pdoc["split"] == "external_test")]
    assert len(kept) == 1
    assert kept.iloc[0]["disease_name"] == "conflict:early_blight|healthy"
    assert kept.iloc[0]["binary_label"] == "conflict"

    pv = m[(m["source"] == "plantvillage") & (m["split"] != "excluded")]
    assert set(pv["split"]) == {"train", "val", "test"}
    assert (pv.groupby("leaf_group_id")["split"].nunique() == 1).all()
    assert m["in_subset"].map(str).isin(["True", "False"]).all()

    m2 = run_prepare(fake_raw, "m2.csv")
    pd.testing.assert_frame_equal(m, m2)
    assert (fake_raw["tmp"] / "reports" / "split_summary.md").is_file()


@pytest.mark.parametrize(
    "options",
    [
        ["--near-dup-threshold", "-1"],
        ["--near-dup-threshold", "65"],
        ["--fractions", "1", "-0.1", "0.1"],
        ["--fractions", "nan", "0", "1"],
        ["--subset-healthy", "-1"],
    ],
)
def test_invalid_cli_parameters(options: list[str]) -> None:
    """Rechaza parámetros que invalidan la partición o desactivan el control perceptual."""
    with pytest.raises(SystemExit, match="2"):
        prepare_dataset.parse_args(options)


@pytest.mark.parametrize("option", ["--out", "--reports"])
def test_outputs_cannot_write_raw(option: str, tmp_path: Path) -> None:
    """Protege los datos originales incluso si la salida usa un enlace simbólico."""
    raw = tmp_path / "originals"
    raw.mkdir()
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(SystemExit, match="2"):
        prepare_dataset.parse_args(["--plantvillage-root", str(raw), option, str(link / "output")])
    with pytest.raises(SystemExit, match="2"):
        prepare_dataset.parse_args([option, "data/raw/output"])
    assert list(raw.iterdir()) == []
