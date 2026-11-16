"""Evaluación única en test con configuración congelada (T05).

- Se niega a ejecutarse si hay cambios sin commitear.
- Registra el hash del commit congelado.
- Evalúa TODAS las combinaciones grupo × modelo (con sus mejores hiperparámetros de CV)
  y marca la elegida con val.
- Permite cambiar la elección al cerrar la tarea sin volver a mirar el test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import training as tr  # noqa: E402


# --------------------------------------------------------------------------- feature groups (replicado para evitar circular)

FEATURE_GROUPS = {
    "color": [
        "h_circ_mean_deg", "h_circ_var",
        "s_mean", "s_std", "v_mean", "v_std",
        "l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std",
        *[f"h_hist_{i:02d}" for i in range(12)],
        *[f"a_hist_{i:02d}" for i in range(8)],
        *[f"b_hist_{i:02d}" for i in range(8)],
        "frac_yellow", "frac_brown", "frac_dark",
    ],
    "textura": [
        "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
        *[f"lbp_{i:02d}" for i in range(18)],
    ],
    "manchas": [
        "spot_count", "spot_area_frac", "spot_mean_area_frac", "spot_max_area_frac",
        "spot_mean_solidity", "spot_mean_eccentricity", "spot_mean_circularity",
    ],
    "control_nitidez": ["sharpness_laplacian_var"],
}

FEATURE_GROUPS["combinadas"] = (
    FEATURE_GROUPS["color"] + FEATURE_GROUPS["textura"] + FEATURE_GROUPS["manchas"]
)
FEATURE_GROUPS["color_manchas"] = FEATURE_GROUPS["color"] + FEATURE_GROUPS["manchas"]

ALL_GROUPS = ["color", "textura", "manchas", "combinadas", "color_manchas", "control_nitidez", "textura_suavizada"]


def get_feature_columns(group_name: str, df: pd.DataFrame) -> list[str]:
    if group_name == "textura_suavizada":
        cols = FEATURE_GROUPS["textura"]
    else:
        cols = FEATURE_GROUPS[group_name]
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
    groups = work["leaf_group_id"].to_numpy() if "leaf_group_id" in work.columns else np.arange(len(work))
    return X, y, groups, feature_cols


def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def check_git_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip() == ""
    except Exception:
        return False


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación única en test (T05).")
    parser.add_argument("--config", type=Path, required=True, help="configs/classification.json")
    parser.add_argument("--frozen-commit", required=True, help="Hash del commit congelado")
    parser.add_argument("--exp-dir", type=Path, default=REPO_ROOT / "experiments/classification/exp1",
                        help="Directorio con cv_val_results.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # 1. Verificar git clean
    if not check_git_clean():
        sys.exit("ERROR: Hay cambios sin commitear. Haz commit o stash antes de evaluar test.")

    # 2. Verificar commit congelado
    current_commit = get_git_commit_hash()
    if current_commit != args.frozen_commit:
        sys.exit(
            f"ERROR: Commit actual ({current_commit[:8]}) != commit congelado ({args.frozen_commit[:8]}).\n"
            f"El repositorio ha cambiado desde la elección en val. No se puede evaluar test."
        )

    # 3. Cargar configuración elegida
    config = json.loads(args.config.read_text())
    chosen_group = config["chosen_group"]
    chosen_model = config["chosen_model"]
    print(f"Configuración elegida (val): {chosen_group} + {chosen_model}")
    print(f"Commit congelado: {current_commit[:8]}")

    # 4. Cargar resultados de CV/val para obtener hiperparámetros de todos los modelos
    results_file = args.exp_dir / "cv_val_results.pkl"
    if not results_file.is_file():
        sys.exit(f"Error: falta {results_file}. Ejecuta train_classifiers.py --cv --val primero.")

    import pickle
    with open(results_file, "rb") as f:
        all_results = pickle.load(f)

    # 5. Cargar datos de test
    subset_csv = REPO_ROOT / "data/processed/features_v1_subset.csv"
    full_csv = REPO_ROOT / "data/processed/features_v1_test_full.csv"
    blur_subset = REPO_ROOT / "data/processed/features_v1_blur_subset.csv"
    blur_full = REPO_ROOT / "data/processed/features_v1_blur_test_full.csv"

    if not subset_csv.is_file() or not full_csv.is_file():
        sys.exit("Error: faltan CSVs de test. Ejecuta extract_features.py --subset-only y --full-test.")

    df_subset = pd.read_csv(subset_csv)
    df_full = pd.read_csv(full_csv)

    use_blur = blur_subset.is_file() and blur_full.is_file()
    if use_blur:
        df_blur_subset = pd.read_csv(blur_subset)
        df_blur_full = pd.read_csv(blur_full)

    # Splits a evaluar
    test_dfs = {
        "test_subset": df_subset[df_subset["split"] == "test"].reset_index(drop=True),
        "test_full": df_full.reset_index(drop=True),
    }
    if use_blur:
        test_blur_dfs = {
            "test_subset": df_blur_subset[df_blur_subset["split"] == "test"].reset_index(drop=True),
            "test_full": df_blur_full.reset_index(drop=True),
        }

    # 6. Entrenar todos los modelos en train (con mejores params de CV) y evaluar en test
    models_grids = tr.get_models_and_grids(args.seed)

    # También necesitamos train_df para re-entrenar
    train_df = df_subset[df_subset["split"] == "train"].reset_index(drop=True)
    if use_blur:
        train_blur = df_blur_subset[df_blur_subset["split"] == "train"].reset_index(drop=True)

    out_dir = args.exp_dir / "test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_test_results = {}
    chosen_result = None

    for group in ALL_GROUPS:
        print(f"\n=== Test: Grupo {group} ===")

        # Datos de train para este grupo
        if group == "textura_suavizada":
            if not use_blur:
                print("  Saltando: no hay v1_blur")
                continue
            X_train, y_train, g_train, feat_cols = prepare_xy(train_blur, "textura_suavizada")
        else:
            X_train, y_train, g_train, feat_cols = prepare_xy(train_df, group)

        group_results = {"features": feat_cols, "models": {}}

        for model_name, (base_est, grid) in models_grids.items():
            # Obtener mejores params de CV
            if group not in all_results or model_name not in all_results[group]["models"]:
                print(f"  {model_name}: no hay resultados de CV, saltando")
                continue

            cv_best_params = all_results[group]["models"][model_name]["cv_best_params"]

            # Entrenar en TODO train
            best_model = tr.train_full(X_train, y_train, model_name, base_est, cv_best_params, args.seed)

            # Evaluar en cada split de test
            test_splits_results = {}
            for split_name, test_df in test_dfs.items():
                if group == "textura_suavizada":
                    X_test, y_test, g_test, _ = prepare_xy(test_blur_dfs[split_name], "textura_suavizada")
                else:
                    X_test, y_test, g_test, _ = prepare_xy(test_df, group)

                # Contar mask_empty en este split
                n_mask_empty = int(test_df["mask_empty"].sum()) if "mask_empty" in test_df.columns else 0

                metrics = tr.evaluate_model(best_model, X_test, y_test, g_test)
                metrics["n_samples"] = len(y_test)
                metrics["n_mask_empty"] = n_mask_empty
                test_splits_results[split_name] = metrics

                # Matriz de confusión para reporte
                y_pred = best_model.predict(X_test)
                cm = metrics["confusion_matrix"]
                print(f"    {split_name}: balanced_acc={metrics['balanced_accuracy']:.4f}, "
                      f"recall_diseased={metrics['recall_diseased']:.4f}, "
                      f"mask_empty={n_mask_empty}, CM={cm}")

            group_results["models"][model_name] = {
                "cv_best_params": cv_best_params,
                "test": test_splits_results,
                "is_chosen": (group == chosen_group and model_name == chosen_model),
            }

            if group == chosen_group and model_name == chosen_model:
                chosen_result = {
                    "group": group,
                    "model": model_name,
                    "test_subset": test_splits_results["test_subset"],
                    "test_full": test_splits_results["test_full"],
                }

        all_test_results[group] = group_results

    # 7. Guardar resultados completos
    with open(out_dir / "test_results.pkl", "wb") as f:
        pickle.dump(all_test_results, f)

    # 8. Generar informe legible
    report_lines = [
        f"# Evaluación en Test (congelado: {current_commit[:8]})",
        "",
        f"Configuración elegida en val: **{chosen_group} + {chosen_model}**",
        f"Commit: {current_commit}",
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resultados completos (todas las combinaciones grupo × modelo)",
        "",
    ]

    for group, gres in all_test_results.items():
        report_lines.append(f"### Grupo: {group}")
        report_lines.append(f"Features: {len(gres['features'])}")
        report_lines.append("")
        report_lines.append("| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |")
        report_lines.append("|---|---:|---:|---:|---:|:---:|")
        for mname, mres in gres["models"].items():
            ts = mres["test"]["test_subset"]
            tf = mres["test"]["test_full"]
            chosen_mark = "✓" if mres["is_chosen"] else ""
            report_lines.append(
                f"| {mname} | {ts['balanced_accuracy']:.4f} | {ts['recall_diseased']:.4f} | "
                f"{tf['balanced_accuracy']:.4f} | {tf['recall_diseased']:.4f} | {chosen_mark} |"
            )
        report_lines.append("")

    # Recall por enfermedad en test_full para el modelo elegido
    if chosen_result:
        report_lines.append("## Recall por enfermedad (test_full) - Modelo elegido")
        report_lines.append("")

        # Recalcular por enfermedad para el modelo elegido
        chosen_group = chosen_result["group"]
        chosen_model = chosen_result["model"]
        cv_best_params = all_results[chosen_group]["models"][chosen_model]["cv_best_params"]
        base_est, _ = models_grids[chosen_model]

        if chosen_group == "textura_suavizada":
            X_train, y_train, _, _ = prepare_xy(train_blur, "textura_suavizada")
            X_test, y_test, _, _ = prepare_xy(test_blur_dfs["test_full"], "textura_suavizada")
        else:
            X_train, y_train, _, _ = prepare_xy(train_df, chosen_group)
            X_test, y_test, _, _ = prepare_xy(test_dfs["test_full"], chosen_group)

        final_model = tr.train_full(X_train, y_train, chosen_model, base_est, cv_best_params, args.seed)
        y_pred = final_model.predict(X_test)

        # Añadir disease_name para desglose
        test_full_df = test_dfs["test_full"]
        test_full_df = test_full_df[~test_full_df["mask_empty"]].reset_index(drop=True) if "mask_empty" in test_full_df.columns else test_full_df
        diseases = test_full_df["disease_name"].unique()

        report_lines.append("| Enfermedad | Recall | Soporte |")
        report_lines.append("|---|---:|---:|")
        for dis in sorted(diseases):
            mask = test_full_df["disease_name"] == dis
            if mask.sum() == 0:
                continue
            y_dis = y_test[mask]
            y_pred_dis = y_pred[mask]
            recall = recall_score(y_dis, y_pred_dis, pos_label=1, zero_division=0)
            report_lines.append(f"| {dis} | {recall:.4f} | {int(mask.sum())} |")

        from sklearn.metrics import recall_score
        report_lines.append("")

    # Análisis de sesgo
    report_lines.append("## Análisis del sesgo de nitidez")
    report_lines.append("")

    # control_nitidez vs mejor modelo
    if "control_nitidez" in all_test_results:
        ctrl_ts = all_test_results["control_nitidez"]["models"]["knn"]["test"]["test_subset"]  # kNN como referencia
        ctrl_tf = all_test_results["control_nitidez"]["models"]["knn"]["test"]["test_full"]
        chosen_ts = chosen_result["test_subset"]
        chosen_tf = chosen_result["test_full"]

        report_lines.append("### control_nitidez (kNN) vs modelo elegido")
        report_lines.append("")
        report_lines.append(f"- control_nitidez test_subset: balanced_acc={ctrl_ts['balanced_accuracy']:.4f}, recall_diseased={ctrl_ts['recall_diseased']:.4f}")
        report_lines.append(f"- modelo elegido test_subset: balanced_acc={chosen_ts['balanced_accuracy']:.4f}, recall_diseased={chosen_ts['recall_diseased']:.4f}")
        report_lines.append("")

    # textura vs textura_suavizada
    if "textura" in all_test_results and "textura_suavizada" in all_test_results:
        tex_ts = all_test_results["textura"]["models"]["knn"]["test"]["test_subset"]
        tex_blur_ts = all_test_results["textura_suavizada"]["models"]["knn"]["test"]["test_subset"]
        report_lines.append("### textura vs textura_suavizada (kNN en test_subset)")
        report_lines.append(f"- textura: balanced_acc={tex_ts['balanced_accuracy']:.4f}")
        report_lines.append(f"- textura_suavizada: balanced_acc={tex_blur_ts['balanced_accuracy']:.4f}")
        report_lines.append("")

    # combinadas vs color_manchas
    if "combinadas" in all_test_results and "color_manchas" in all_test_results:
        comb_ts = all_test_results["combinadas"]["models"]["knn"]["test"]["test_subset"]
        cm_ts = all_test_results["color_manchas"]["models"]["knn"]["test"]["test_subset"]
        report_lines.append("### combinadas vs color_manchas (kNN en test_subset)")
        report_lines.append(f"- combinadas: balanced_acc={comb_ts['balanced_accuracy']:.4f}")
        report_lines.append(f"- color_manchas: balanced_acc={cm_ts['balanced_accuracy']:.4f}")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("*Nota: Las puntuaciones no son probabilidades calibradas.*")

    (out_dir / "test_report.md").write_text("\n".join(report_lines))
    print(f"\nInforme guardado en {out_dir / 'test_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())