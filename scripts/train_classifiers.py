"""Entrena y evalúa clasificadores clásicos con CV agrupada (T05).

Uso:
  # 1. Calcular sigma óptimo para textura_suavizada
  python scripts/train_classifiers.py --compute-sigma --seed 42

  # 2. Generar CSV v1_blur (textura recalculada con blur)
  python scripts/train_classifiers.py --generate-blur --sigma 1.5 --seed 42

  # 3. CV en train + evaluación en val para todos los grupos × modelos
  python scripts/train_classifiers.py --cv --val --seed 42 --out experiments/classification/exp1

  # 4. Evaluar en test (congelado) - solo una vez al final
  python scripts/eval_test.py --config configs/classification.json --frozen-commit <hash>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import training as tr  # noqa: E402

# --------------------------------------------------------------------------- feature group helpers

FEATURE_GROUPS = {
    "color": [
        "h_circ_mean_deg",
        "h_circ_var",
        "s_mean",
        "s_std",
        "v_mean",
        "v_std",
        "l_mean",
        "l_std",
        "a_mean",
        "a_std",
        "b_mean",
        "b_std",
        *[f"h_hist_{i:02d}" for i in range(12)],
        *[f"a_hist_{i:02d}" for i in range(8)],
        *[f"b_hist_{i:02d}" for i in range(8)],
        "frac_yellow",
        "frac_brown",
        "frac_dark",
    ],
    "textura": [
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        *[f"lbp_{i:02d}" for i in range(18)],
    ],
    "manchas": [
        "spot_count",
        "spot_area_frac",
        "spot_mean_area_frac",
        "spot_max_area_frac",
        "spot_mean_solidity",
        "spot_mean_eccentricity",
        "spot_mean_circularity",
    ],
    "control_nitidez": ["sharpness_laplacian_var"],
}


FEATURE_GROUPS["combinadas"] = (
    FEATURE_GROUPS["color"] + FEATURE_GROUPS["textura"] + FEATURE_GROUPS["manchas"]
)
FEATURE_GROUPS["color_manchas"] = FEATURE_GROUPS["color"] + FEATURE_GROUPS["manchas"]
# Variante C: color_manchas sin l_std ni v_std
FEATURE_GROUPS["color_manchas_sin_l_v"] = [
    c for c in FEATURE_GROUPS["color_manchas"] if c not in ("l_std", "v_std")
]
# textura_suavizada se define dinámicamente (mismas columnas que textura)

CANDIDATE_GROUPS = [
    "color",
    "textura",
    "manchas",
    "combinadas",
    "color_manchas",
    "color_manchas_sin_l_v",
    "textura_suavizada",
]
ALL_GROUPS = CANDIDATE_GROUPS + ["control_nitidez"]


def get_feature_columns(group_name: str, df: pd.DataFrame) -> list[str]:
    if group_name == "textura_suavizada":
        cols = FEATURE_GROUPS["textura"]
    else:
        cols = FEATURE_GROUPS[group_name]
    if cols is None:
        raise ValueError(f"Grupo {group_name} no definido")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return cols


def prepare_xy(df: pd.DataFrame, group_name: str, exclude_mask_empty: bool = True):
    work = df.copy()
    if exclude_mask_empty and "mask_empty" in work.columns:
        work = work[~work["mask_empty"]].copy()
    feature_cols = get_feature_columns(group_name, work)
    X = work[feature_cols].to_numpy(dtype=np.float64)
    y = (work["binary_label"] != "healthy").astype(int).to_numpy()
    groups = (
        work["leaf_group_id"].to_numpy()
        if "leaf_group_id" in work.columns
        else np.arange(len(work))
    )
    return X, y, groups, feature_cols


ID_COLUMNS = ["image_id", "split", "binary_label", "disease_name", "leaf_group_id"]

# --------------------------------------------------------------------------- commands


def cmd_compute_sigma(args: argparse.Namespace) -> int:
    """Calcula sigma óptimo para GaussianBlur (criterio: mediana sana en IQR enferma)."""
    subset_csv = REPO_ROOT / "data/processed/features_v1_subset.csv"
    if not subset_csv.is_file():
        sys.exit(f"Error: falta {subset_csv}. Ejecuta extract_features.py --subset-only primero.")

    print("Calculando sigma óptimo con train...")
    sigma, details = tr.compute_blur_sigma(subset_csv, seed=args.seed)

    out_dir = REPO_ROOT / "reports/classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "criterion": (
            "menor sigma donde mediana(sharpness sano) dentro de "
            "IQR(sharpness enfermo) tras blur común"
        ),
        "sigma_candidates": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "chosen_sigma": sigma,
        "details": details,
    }
    (out_dir / "sigma_selection.json").write_text(json.dumps(report, indent=2))
    print(f"Sigma elegido: {sigma}")
    print(f"Detalles guardados en {out_dir / 'sigma_selection.json'}")
    return 0


def cmd_generate_blur(args: argparse.Namespace) -> int:
    """Genera CSV v1_blur con textura recalculada tras GaussianBlur(sigma)."""
    sigma = args.sigma
    if sigma is None:
        # Leer del archivo si existe
        sigma_file = REPO_ROOT / "reports/classification/sigma_selection.json"
        if sigma_file.is_file():
            sigma = json.loads(sigma_file.read_text())["chosen_sigma"]
        else:
            sys.exit("Error: especifica --sigma o ejecuta --compute-sigma primero.")

    splits = ["subset", "test_full"]
    for split in splits:
        in_csv = REPO_ROOT / f"data/processed/features_v1_{split}.csv"
        out_csv = REPO_ROOT / f"data/processed/features_v1_blur_{split}.csv"
        if not in_csv.is_file():
            print(f"Saltando {split}: no existe {in_csv}")
            continue
        print(f"Generando {out_csv.name} con sigma={sigma}...")
        tr.generate_v1_blur(in_csv, out_csv, sigma, seed=args.seed)
        print(f"  SHA-1: {tr.sha1_file(out_csv)}")

    print("Hecho. Usa features_v1_blur_*.csv para el grupo 'textura_suavizada'.")
    return 0


def cmd_cv_val(args: argparse.Namespace) -> int:
    """CV agrupada en train + evaluación en val para todos los grupos × modelos."""
    subset_csv = REPO_ROOT / "data/processed/features_v1_subset.csv"
    blur_csv = REPO_ROOT / "data/processed/features_v1_blur_subset.csv"

    if not subset_csv.is_file():
        sys.exit(f"Error: falta {subset_csv}")

    use_blur = blur_csv.is_file()
    print(f"Usando {'v1_blur' if use_blur else 'v1'} para textura_suavizada")

    df = pd.read_csv(subset_csv)
    if use_blur:
        df_blur = pd.read_csv(blur_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if use_blur:
        train_blur = df_blur[df_blur["split"] == "train"].reset_index(drop=True)
        val_blur = df_blur[df_blur["split"] == "val"].reset_index(drop=True)

    models_grids = tr.get_models_and_grids(args.seed)
    groups = [
        "color",
        "textura",
        "manchas",
        "combinadas",
        "color_manchas",
        "color_manchas_sin_l_v",
        "control_nitidez",
    ]
    if use_blur:
        groups.append("textura_suavizada")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_commit.txt").write_text(tr.get_git_commit_hash() + "\n")
    all_results = {}
    baseline_thresholds = {}

    for group in groups:
        print(f"\n=== Grupo: {group} ===")
        # Preparar datos
        if group == "textura_suavizada":
            X_train, y_train, g_train, feat_cols = prepare_xy(train_blur, "textura_suavizada")
            X_val, y_val, g_val, _ = prepare_xy(val_blur, "textura_suavizada", False)
        else:
            X_train, y_train, g_train, feat_cols = prepare_xy(train_df, group)
            X_val, y_val, g_val, _ = prepare_xy(val_df, group, False)

        print(f"  Train: {len(X_train)} muestras, {X_train.shape[1]} features")
        print(f"  Val:   {len(X_val)} muestras")

        # Baseline thresholds (solo para grupos que tengan spot/frac_brown)
        if group in ("manchas", "combinadas", "color_manchas"):
            baseline_thresholds[group] = tr.compute_baseline_thresholds(X_train, y_train, feat_cols)

        group_results = {"features": feat_cols, "models": {}}

        for model_name, (base_est, grid) in models_grids.items():
            print(f"  Modelo: {model_name}...")
            start = time.perf_counter()

            # CV en train
            cv_res = tr.cv_grouped(
                X_train,
                y_train,
                g_train,
                model_name,
                base_est,
                grid,
                cv_folds=args.cv_folds,
                seed=args.seed,
            )

            # Entrenar en todo train con mejores params
            best_model = tr.train_full(
                X_train, y_train, model_name, base_est, cv_res["best_params"], args.seed
            )

            # Evaluar en val
            eval_df = val_blur if group == "textura_suavizada" else val_df
            empty = eval_df["mask_empty"].to_numpy(bool)
            val_metrics = tr.evaluate_model(best_model, X_val, y_val, g_val, empty)
            import joblib

            model_dir = REPO_ROOT / "models" / out_dir.name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_dir / f"{group}__{model_name}.joblib")
            pred, score = tr.predict_with_empty(best_model, X_val, y_val, empty)
            predictions = eval_df[ID_COLUMNS].copy()
            predictions["y_true"], predictions["y_pred"], predictions["score"] = y_val, pred, score
            predictions["mask_empty"] = empty
            predictions.to_csv(
                out_dir / f"val__{group}__{model_name}.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )

            # Baseline si aplica
            baseline_metrics = None
            if group in baseline_thresholds:
                y_pred_base = tr.baseline_rules_predict(
                    np.nan_to_num(X_val), feat_cols, baseline_thresholds[group]
                )
                y_pred_base[empty] = 1 - y_val[empty]
                baseline_metrics = tr.compute_metrics(y_val, y_pred_base)

            elapsed = time.perf_counter() - start
            print(f"    CV best balanced_acc: {cv_res['best_score']:.4f} ({cv_res['best_params']})")
            print(
                f"    Val balanced_acc: {val_metrics['balanced_accuracy']:.4f}, "
                f"recall_diseased: {val_metrics['recall_diseased']:.4f}"
            )
            if baseline_metrics:
                print(f"    Baseline balanced_acc: {baseline_metrics['balanced_accuracy']:.4f}")

            group_results["models"][model_name] = {
                "cv_best_params": cv_res["best_params"],
                "cv_best_score": cv_res["best_score"],
                "val_metrics": val_metrics,
                "baseline_metrics": baseline_metrics,
                "train_time_sec": elapsed,
            }

        all_results[group] = group_results

    # Guardar resultados
    import pickle

    with open(out_dir / "cv_val_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Resumen legible
    summary_lines = ["# Resultados CV (train) + Val por grupo × modelo", ""]
    for group, gres in all_results.items():
        summary_lines.append(f"## Grupo: {group}")
        summary_lines.append(f"Features: {len(gres['features'])}")
        summary_lines.append("")
        summary_lines.append(
            "| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |"
        )
        summary_lines.append("|---|---:|---:|---:|---:|")
        for mname, mres in gres["models"].items():
            vm = mres["val_metrics"]
            auc = vm.get("roc_auc", float("nan"))
            summary_lines.append(
                f"| {mname} | {mres['cv_best_score']:.4f} | "
                f"{vm['balanced_accuracy']:.4f} | {vm['recall_diseased']:.4f} | {auc:.4f} |"
            )
        summary_lines.append("")

    (out_dir / "summary.md").write_text("\n".join(summary_lines))
    print(f"\nResultados guardados en {out_dir}")
    print(f"Resumen: {out_dir / 'summary.md'}")
    return 0


def cmd_cv_val_variant(args: argparse.Namespace) -> int:
    """Un solo grupo × modelo con el mismo procedimiento que ``cv-val`` (B′ de T08).

    Misma rejilla, CV agrupada por hoja en ``train`` (StratifiedGroupKFold, ``--cv-folds``,
    ``--seed``), reentreno en todo ``train`` y evaluación en ``val``. Solo cambia el CSV de
    características (p. ej. con gray world). Escribe el modelo en ``models/<out>/``, las
    predicciones de ``val`` y ``results.json`` en ``--out``. No toca ``test``.
    """
    import joblib

    df = pd.read_csv(args.features)
    if set(df["split"]) - {"train", "val"}:
        sys.exit(f"Error: {args.features} contiene splits distintos de train/val")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    X_train, y_train, g_train, feat_cols = prepare_xy(train_df, args.group)
    X_val, y_val, g_val, _ = prepare_xy(val_df, args.group, False)
    base_est, grid = tr.get_models_and_grids(args.seed)[args.model]
    cv_res = tr.cv_grouped(
        X_train, y_train, g_train, args.model, base_est, grid, args.cv_folds, args.seed
    )
    model = tr.train_full(X_train, y_train, args.model, base_est, cv_res["best_params"], args.seed)
    empty = val_df["mask_empty"].to_numpy(bool)
    val_metrics = tr.evaluate_model(model, X_val, y_val, g_val, empty)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = REPO_ROOT / "models" / out_dir.name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{args.group}__{args.model}.joblib")
    pred, score = tr.predict_with_empty(model, X_val, y_val, empty)
    predictions = val_df[ID_COLUMNS].copy()
    predictions["y_true"], predictions["y_pred"], predictions["score"] = y_val, pred, score
    predictions["mask_empty"] = empty
    predictions.to_csv(
        out_dir / f"val__{args.group}__{args.model}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    results = {
        "commit": tr.get_git_commit_hash(),
        "features_csv": Path(args.features).name,
        "features_sha1": tr.sha1_file(Path(args.features)),
        "group": args.group,
        "model": args.model,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "cv_best_params": cv_res["best_params"],
        "cv_best_score": cv_res["best_score"],
        "val_metrics": val_metrics,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n"
    )
    print(
        f"{args.group} + {args.model}: CV BA {cv_res['best_score']:.6f} "
        f"({cv_res['best_params']}); val BA {val_metrics['balanced_accuracy']:.6f}"
    )
    return 0


def cmd_choose_best(args: argparse.Namespace) -> int:
    """Elige la mejor config según val (mayor balanced_acc) y guarda en configs."""
    results_file = Path(args.out) / "cv_val_results.pkl"
    if not results_file.is_file():
        sys.exit(f"Error: falta {results_file}. Ejecuta --cv --val primero.")

    import pickle

    with open(results_file, "rb") as f:
        all_results = pickle.load(f)

    best_group, best_model, best = tr.choose_candidate(all_results)

    commit_hash = tr.get_git_commit_hash()
    git_clean = tr.check_git_clean()

    config = {
        "chosen_group": best_group,
        "chosen_model": best_model,
        "chosen_by": "max balanced_accuracy on val (tie-break: simpler model, margin 0.005)",
        "val_balanced_accuracy": best["val_metrics"]["balanced_accuracy"],
        "val_recall_diseased": best["val_metrics"]["recall_diseased"],
        "cv_best_params": best["cv_best_params"],
        "frozen_commit": commit_hash,
        "git_clean_at_choice": git_clean,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = REPO_ROOT / "configs/classification.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Configuración elegida: {best_group} + {best_model}")
    print(f"Val balanced_acc: {best['val_metrics']['balanced_accuracy']:.4f}")
    print(f"Val recall_diseased: {best['val_metrics']['recall_diseased']:.4f}")
    print(f"Commit congelado: {commit_hash}")
    print(f"Git clean: {git_clean}")
    print(f"Guardado en {config_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación T05.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sigma = sub.add_parser("compute-sigma", help="Calcula sigma óptimo para GaussianBlur")
    p_sigma.add_argument("--seed", type=int, default=42)

    p_blur = sub.add_parser("generate-blur", help="Genera CSV v1_blur con textura recalculada")
    p_blur.add_argument("--sigma", type=float, help="Sigma del GaussianBlur")
    p_blur.add_argument("--seed", type=int, default=42)

    p_cv = sub.add_parser("cv-val", help="CV en train + evaluación en val")
    p_cv.add_argument("--cv-folds", type=int, default=5)
    p_cv.add_argument("--seed", type=int, default=42)
    p_cv.add_argument("--out", type=Path, default=REPO_ROOT / "experiments/classification/exp1")

    p_var = sub.add_parser("cv-val-variant", help="Un grupo × modelo con otro CSV (B′, T08)")
    p_var.add_argument("--features", type=Path, required=True)
    p_var.add_argument("--group", default="color_manchas")
    p_var.add_argument("--model", default="svm_lineal")
    p_var.add_argument("--cv-folds", type=int, default=5)
    p_var.add_argument("--seed", type=int, default=42)
    p_var.add_argument("--out", type=Path, required=True)

    p_choose = sub.add_parser(
        "choose-best", help="Elige mejor config según val y guarda en configs/classification.json"
    )
    p_choose.add_argument("--out", type=Path, default=REPO_ROOT / "experiments/classification/exp1")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compute-sigma":
        return cmd_compute_sigma(args)
    elif args.command == "generate-blur":
        return cmd_generate_blur(args)
    elif args.command == "cv-val":
        return cmd_cv_val(args)
    elif args.command == "cv-val-variant":
        return cmd_cv_val_variant(args)
    elif args.command == "choose-best":
        return cmd_choose_best(args)
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
