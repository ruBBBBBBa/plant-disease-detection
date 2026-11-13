"""Ajusta la extracción de características **solo con train** y escribe ``configs/features.json``.

Protocolo de T04: aquí no se lee ``val`` ni ``test``. Con las imágenes ``train`` del subconjunto:

1. Variante de máscara frente a sombras (D019): (a) tal cual, (b) erosionada, (c) sin sombra
   conectada al borde. Criterio fijado de antemano (``docs/tasks/T04-caracteristicas.md``):
   AUC de la fracción de píxeles oscuros (V < 60) entre sanas y late/early blight.
2. Umbrales de tonos anómalos a partir de los píxeles de las hojas **sanas**.
3. Rangos de los histogramas de a* y b* (p1–p99 de los píxeles de hoja).
4. Parámetros de GLCM y LBP: media de |AUC − 0,5| de las características de la familia.
5. Criterio «no concluyente» (D017): percentiles extremos de la calidad de la máscara.

Ejemplo:
    python scripts/tune_features.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data_audit as audit  # noqa: E402
from src import features as ft  # noqa: E402
from src import segmentation as seg  # noqa: E402

NECROSIS = {"late_blight", "early_blight"}
TIE_MARGIN = 0.01  # desempate: la variante más simple salvo mejora ≥ 0,01 de AUC
VARIANT_GRID = (
    [{"kind": "raw"}]
    + [{"kind": "erode", "n": n} for n in (2, 4, 6, 8)]
    + [{"kind": "shadow", "v_max": v, "s_max": s} for v in (50, 70, 90) for s in (40, 60, 80)]
)
TEXTURE_GRID = {
    "glcm": [{"glcm_levels": 32, "glcm_distance": d} for d in (1, 2, 4)],
    "lbp": [{"lbp_points": 8 * r, "lbp_radius": r} for r in (1, 2, 3)],
}
SIMPLICITY = {"raw": 0, "erode": 1, "shadow": 2}
V_HUE_MIN = 50  # por debajo, H no es fiable (picos espurios en los píxeles oscuros de sanas)
SENSITIVITY_DARK_V = (30, 15)
# Reglas «no concluyente» activas por defecto. Revisando inconclusive_train.jpg: área y borde marcan
# sobre todo hojas bien segmentadas (pequeñas o en primer plano); la solidez recoge la mayoría de
# las máscaras fragmentadas. Las otras dos quedan en "inconclusive_candidates" para activarlas.
ACTIVE_RULES = (
    "mask_solidity_min",
)  # análisis post hoc del criterio; no interviene en la elección


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee las opciones de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Ajuste de características con train (T04).")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data/metadata.csv")
    parser.add_argument("--seg-config", type=Path, default=REPO_ROOT / "configs/segmentation.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/features.json")
    parser.add_argument("--reports", type=Path, default=REPO_ROOT / "reports/features")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def auc(labels: pd.Series, values: list[float]) -> float:
    """AUC univariante (1 = enferma): probabilidad de que una enferma puntúe más que una sana."""
    return float(roc_auc_score(labels, values))


def compare_variants(samples: list[dict], rows: pd.DataFrame) -> pd.DataFrame:
    """Criterio de sombras para cada variante de la rejilla (ver docstring del módulo)."""
    is_necrosis = rows["disease_name"].isin(NECROSIS).to_numpy()
    is_healthy = (rows["binary_label"] == "healthy").to_numpy()
    results = []
    for variant in VARIANT_GRID:
        dark, lost = [], []
        sensitivity = {v: [] for v in SENSITIVITY_DARK_V}
        for sample in samples:
            mask = ft.prepare_mask(sample["image"], sample["mask"], variant)
            dark.append(ft.dark_fraction(sample["image"], mask))
            for v in SENSITIVITY_DARK_V:
                sensitivity[v].append(ft.dark_fraction(sample["image"], mask, v))
            raw_dark = sample["dark_px"]
            kept_dark = (sample["dark"] & (mask > 0)).sum()
            lost.append(1 - kept_dark / raw_dark if raw_dark else 0.0)
        dark, lost = np.array(dark), np.array(lost)
        primary = is_healthy | is_necrosis
        results.append(
            {
                "variant": json.dumps(variant),
                "kind": variant["kind"],
                "auc_healthy_vs_blight": round(auc(~is_healthy[primary], dark[primary]), 4),
                "auc_healthy_vs_diseased": round(auc(~is_healthy, dark), 4),
                "median_dark_healthy": round(float(np.median(dark[is_healthy])), 4),
                "median_dark_blight": round(float(np.median(dark[is_necrosis])), 4),
                "dark_px_removed_blight": round(float(lost[is_necrosis].mean()), 4),
                **{
                    f"posthoc_auc_blight_v{v}": round(
                        auc(~is_healthy[primary], np.array(values)[primary]), 4
                    )
                    for v, values in sensitivity.items()
                },
            }
        )
    return pd.DataFrame(results)


def choose_variant(table: pd.DataFrame) -> dict:
    """Mejor parámetro de cada tipo por el criterio principal y desempate por simplicidad."""
    best = table.loc[table.groupby("kind")["auc_healthy_vs_blight"].idxmax()]
    best = best.assign(order=best["kind"].map(SIMPLICITY)).sort_values("order")
    chosen = best.iloc[0]
    for row in best.iloc[1:].itertuples(index=False):
        if row.auc_healthy_vs_blight >= chosen["auc_healthy_vs_blight"] + TIE_MARGIN:
            chosen = pd.Series(row._asdict())
    return json.loads(chosen["variant"])


def tone_thresholds(samples: list[dict], rows: pd.DataFrame, variant: dict) -> dict:
    """Umbrales de tonos con los píxeles de hojas **sanas** de train (percentil 1).

    - ``v_dark`` = p1 de V.
    - ``h_green_min`` = p1 de H entre píxeles de tono fiable (S ≥ 40 y V ≥ 50): en los píxeles
      oscuros H es ruido y daba un p1 espurio (≈ 15).
    - ``h_brown_yellow`` = 20 (40°), por teoría del color: por debajo, naranjas y marrones.
    """
    v_values, h_values = [], []
    for sample, label_name in zip(samples, rows["binary_label"], strict=True):
        if label_name != "healthy":
            continue
        mask = ft.prepare_mask(sample["image"], sample["mask"], variant) > 0
        hsv = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2HSV)[mask]
        v_values.append(hsv[:, 2])
        h_values.append(hsv[(hsv[:, 1] >= 40) & (hsv[:, 2] >= V_HUE_MIN), 0])
    return {
        "v_dark": int(np.percentile(np.concatenate(v_values), 1)),
        "s_min": 40,
        "v_hue_min": V_HUE_MIN,
        "h_brown_yellow": 20,
        "h_green_min": int(np.percentile(np.concatenate(h_values), 1)),
    }


def hist_ranges(samples: list[dict], variant: dict, seed: int) -> dict:
    """Rango p1–p99 de a* y b* (con signo) en los píxeles de hoja de train (muestra por imagen)."""
    rng = np.random.default_rng(seed)
    a_values, b_values = [], []
    for sample in samples:
        mask = ft.prepare_mask(sample["image"], sample["mask"], variant) > 0
        lab = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2LAB)[mask].astype(np.float64)
        pick = rng.choice(len(lab), size=min(2000, len(lab)), replace=False)
        a_values.append(lab[pick, 1] - 128)
        b_values.append(lab[pick, 2] - 128)
    a, b = np.concatenate(a_values), np.concatenate(b_values)
    return {
        "a": [float(np.percentile(a, 1)), float(np.percentile(a, 99))],
        "b": [float(np.percentile(b, 1)), float(np.percentile(b, 99))],
    }


def tune_texture(samples: list[dict], rows: pd.DataFrame, variant: dict) -> tuple[dict, list]:
    """Elige distancia GLCM y (P, R) de LBP por la media de |AUC − 0,5| de su familia."""
    labels = rows["binary_label"] != "healthy"
    grays, masks = [], []
    for sample in samples:
        grays.append(cv2.cvtColor(sample["image"], cv2.COLOR_BGR2GRAY))
        masks.append(ft.prepare_mask(sample["image"], sample["mask"], variant))
    chosen, log = {}, []
    for family, options in TEXTURE_GRID.items():
        best, best_score = None, -1.0
        for params in options:
            if family == "glcm":
                feats = [
                    ft.glcm_features(g, m, params["glcm_levels"], params["glcm_distance"])
                    for g, m in zip(grays, masks, strict=True)
                ]
            else:
                feats = [
                    ft.lbp_features(g, m, params["lbp_points"], params["lbp_radius"])
                    for g, m in zip(grays, masks, strict=True)
                ]
            table = pd.DataFrame(feats)
            score = float(np.mean([abs(auc(labels, table[c]) - 0.5) for c in table.columns]))
            log.append({"family": family, "params": json.dumps(params), "mean_abs_auc": score})
            if score > best_score:
                best, best_score = params, score
        chosen |= best
    return chosen, log


def inconclusive_rules(samples: list[dict]) -> tuple[dict, pd.DataFrame]:
    """Umbrales candidatos «no concluyente» con la calidad de la máscara de T03 en train (p1 / p99).

    Se calculan las tres reglas; cuál se activa se decide revisando ``inconclusive_train.jpg``
    (ver ``ACTIVE_RULES``).
    """
    quality = pd.DataFrame([ft.quality_features(s["mask"]) for s in samples])
    rules = {
        "leaf_area_frac_min": round(float(quality["leaf_area_frac"].quantile(0.01)), 4),
        "mask_solidity_min": round(float(quality["mask_solidity"].quantile(0.01)), 4),
        "border_touch_frac_max": round(float(quality["border_touch_frac"].quantile(0.99)), 4),
    }
    return rules, quality


def tile(image: np.ndarray, mask: np.ndarray, label_text: str, size: int = 128) -> np.ndarray:
    """Miniatura con el contorno de la máscara en rojo y una etiqueta."""
    out = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    cv2.putText(out, label_text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def variants_grid(samples, rows, table: pd.DataFrame, out: Path) -> None:
    """Rejilla: 4 sanas con más «oscuro» en (a) y 4 blight con más oscuro en el borde.

    Columnas: máscara (a) y la mejor (b) y (c) según el criterio.
    """
    best = table.loc[table.groupby("kind")["auc_healthy_vs_blight"].idxmax()]
    variants = [
        json.loads(v) for v in best.sort_values("kind", key=lambda k: k.map(SIMPLICITY))["variant"]
    ]
    raw_dark = np.array([s["raw_dark_frac"] for s in samples])
    rim_dark = np.array([s["rim_dark_frac"] for s in samples])
    healthy = np.flatnonzero((rows["binary_label"] == "healthy").to_numpy())
    blight = np.flatnonzero(rows["disease_name"].isin(NECROSIS).to_numpy())
    picks = list(healthy[np.argsort(-raw_dark[healthy])[:4]]) + list(
        blight[np.argsort(-rim_dark[blight])[:4]]
    )
    lines = []
    for i in picks:
        sample = samples[i]
        tiles = []
        for variant in variants:
            mask = ft.prepare_mask(sample["image"], sample["mask"], variant)
            dark = ft.dark_fraction(sample["image"], mask)
            tiles.append(tile(sample["image"], mask, f"{variant['kind']} oscuro={dark:.2f}"))
        lines.append(np.hstack(tiles))
    cv2.imwrite(str(out), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 80])


RULE_TESTS = {
    "area": lambda q, r: q["leaf_area_frac"] < r["leaf_area_frac_min"],
    "solidez": lambda q, r: q["mask_solidity"] < r["mask_solidity_min"],
    "borde": lambda q, r: q["border_touch_frac"] > r["border_touch_frac_max"],
}


def inconclusive_grid(samples, rows, quality: pd.DataFrame, rules: dict, out: Path) -> int:
    """Rejilla de las imágenes de train marcadas «no concluyentes», agrupadas por regla.

    Cada bloque de filas es una regla (área, solidez, borde); la etiqueta de cada miniatura dice
    la regla y la enfermedad. Una imagen que cumple varias reglas aparece en cada bloque.
    """
    cols, blocks = 8, []
    for rule, test in RULE_TESTS.items():
        tiles = [
            tile(s["image"], s["mask"], f"{rule} {rows['disease_name'].iloc[i][:10]}")
            for i, s in enumerate(samples)
            if test(quality.iloc[i], rules)
        ]
        if tiles:
            tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
            blocks += [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    if blocks:
        cv2.imwrite(str(out), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return sum(ft.is_inconclusive(q.to_dict(), rules) for _, q in quality.iterrows())


def load_train(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    """Imágenes train del subconjunto con su máscara de T03 y datos para el criterio de sombras."""
    metadata = pd.read_csv(args.metadata, keep_default_na=False)
    rows = seg.select_images(metadata, "train", None, args.seed)
    seg_config = json.loads(args.seg_config.read_text(encoding="utf-8"))
    samples = []
    for path in rows["image_path"]:
        image = seg.read_image(REPO_ROOT / path)
        mask = ft.leaf_mask(image, seg_config)
        v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = (mask > 0) & (v < ft.DARK_V)
        rim = (mask > 0) & (cv2.erode(mask, seg.ellipse_kernel(9)) == 0)
        samples.append(
            {
                "image": image,
                "mask": mask,
                "dark": dark,
                "dark_px": int(dark.sum()),
                "raw_dark_frac": ft.dark_fraction(image, mask),
                "rim_dark_frac": float((dark & rim).sum() / max(1, (mask > 0).sum())),
            }
        )
    return rows, samples


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el ajuste con train y escribe configuración, informe y rejillas."""
    args = parse_args(argv)
    for path in (args.metadata, args.seg_config):
        if not path.is_file():
            sys.exit(f"Error: falta {path.name}. Ver docs/tasks/T02 y T03 para regenerarlo.")
    rows, samples = load_train(args)
    print(f"train: {len(rows)} imágenes")
    args.reports.mkdir(parents=True, exist_ok=True)

    variants = compare_variants(samples, rows)
    variant = choose_variant(variants)
    tones = tone_thresholds(samples, rows, variant)
    ranges = hist_ranges(samples, variant, args.seed)
    texture, texture_log = tune_texture(samples, rows, variant)
    rules, quality = inconclusive_rules(samples)

    config = {
        "_comment": "Ajustado solo con train por scripts/tune_features.py (T04).",
        "tuned_on": {"split": "train", "subset": True, "images": len(rows), "seed": args.seed},
        "mask_variant": variant,
        "tones": tones,
        "hist_ranges": ranges,
        "texture": texture,
        "spots": {"min_area": 10},
        "inconclusive": {name: rules[name] for name in ACTIVE_RULES},
        "inconclusive_candidates": rules,
    }
    args.config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    variants_grid(samples, rows, variants, args.reports / "shadow_variants_train.jpg")
    inconclusive_grid(samples, rows, quality, rules, args.reports / "inconclusive_train.jpg")
    active = config["inconclusive"]
    flags = [ft.is_inconclusive(q.to_dict(), active) for _, q in quality.iterrows()]
    n_flagged = sum(flags)
    flagged_by = {
        "leaf_area_frac < min": int(
            (quality["leaf_area_frac"] < rules["leaf_area_frac_min"]).sum()
        ),
        "mask_solidity < min": int((quality["mask_solidity"] < rules["mask_solidity_min"]).sum()),
        "border_touch_frac > max": int(
            (quality["border_touch_frac"] > rules["border_touch_frac_max"]).sum()
        ),
    }
    any_rule = [ft.is_inconclusive(q.to_dict(), rules) for _, q in quality.iterrows()]
    flagged_labels = rows.loc[flags, "binary_label"].value_counts()
    report = [
        "# Ajuste de características con train",
        "",
        f"`python scripts/tune_features.py --seed {args.seed}` — {len(rows)} imágenes `train` del "
        "subconjunto. Nada de `val` ni `test`.",
        "",
        "## Variantes de máscara frente a sombras (D019)",
        "",
        "Criterio fijado de antemano (`docs/tasks/T04-caracteristicas.md`): AUC de la fracción de "
        "píxeles oscuros (V < 60) entre sanas y late/early blight; desempate por simplicidad salvo "
        f"mejora ≥ {TIE_MARGIN}. `dark_px_removed_blight`: fracción media de píxeles oscuros de "
        "la máscara (a) que la variante quita en hojas de blight (pérdida de lesión). Las "
        "columnas `posthoc_auc_blight_v30/v15` repiten el criterio con V < 30 y V < 15: son un "
        "**análisis de sensibilidad posterior**, no intervienen en la elección.",
        "",
        audit.md_table(variants.drop(columns="kind")),
        "",
        f"**Variante elegida:** `{json.dumps(variant)}`. Rejilla: `shadow_variants_train.jpg` "
        "(filas 1–4: sanas con más oscuro en (a); filas 5–8: blight con más oscuro en el borde; "
        "columnas: mejor (a), (b) y (c)).",
        "",
        "## Umbrales de tonos (píxeles de hojas sanas)",
        "",
        f"`{json.dumps(tones)}` (H en escala OpenCV 0–179). Rangos de histograma a*/b*: "
        f"`{json.dumps({k: [round(x, 2) for x in v] for k, v in ranges.items()})}`.",
        "",
        "## Textura (media de |AUC − 0,5| de la familia)",
        "",
        audit.md_table(pd.DataFrame(texture_log).round(4)),
        "",
        f"Elegido: `{json.dumps(texture)}`.",
        "",
        "## Criterio «no concluyente» (D017)",
        "",
        f"Reglas candidatas (p1 / p99 de train): `{json.dumps(rules)}`. Con las tres juntas se "
        f"marcarían {sum(any_rule)} de {len(rows)} imágenes de train ({json.dumps(flagged_by)}).",
        "",
        "Revisión visual (`inconclusive_train.jpg`, un bloque por regla): las reglas de área y de "
        "borde marcan sobre todo hojas bien segmentadas (pequeñas o en primer plano); la de "
        "solidez recoge la mayoría de las máscaras fragmentadas. **Regla activa por defecto:** "
        f"`{json.dumps(active)}` → **{n_flagged}** imágenes de train no concluyentes "
        f"(por etiqueta: {flagged_labels.to_dict()}).",
        "",
    ]
    (args.reports / "tuning_train.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Variante: {variant}; no concluyentes en train: {n_flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
