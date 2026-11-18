"""Juez del bucle de mutación: decide si un experimento (tests nuevos) se queda o se descarta.

Uso:
  python loop/evaluate.py "<qué test añade y qué fallo caza>"

Orden fijo (loop/config.json define rutas, puerta, métrica y límites):
  (a) Si hay cambios en rutas protegidas respecto al tag `base` (ficheros versionados o nuevos
      sin versionar), escribe loop/STOP, imprime PARADA y sale con código 2.
  (b) Pasa la puerta (ruff + pytest). Si falla, descarta.
  (c) Ejecuta la métrica (última línea de su salida = valor).
  (d) Compara con el mejor valor «base» o «keep» de loop/results.tsv: solo se queda si mejora
      más que `min_improvement`.
  (e) keep = `git add` de las rutas editables + commit «loop(mutation): …»;
      discard = `git stash push --include-untracked` de esas mismas rutas.
  (f) Añade una fila a loop/results.tsv e imprime el estado y los descartes seguidos.
La primera ejecución (sin filas) se registra como «base» y no hace commit ni stash.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = REPO_ROOT / "loop"
CONFIG_PATH = LOOP_DIR / "config.json"
RESULTS_PATH = LOOP_DIR / "results.tsv"
STOP_PATH = LOOP_DIR / "STOP"
TSV_FIELDS = ["fecha", "commit", "valor", "estado", "descripcion"]
EXIT_STOP = 2


def load_config() -> dict:
    """Lee loop/config.json."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Ejecuta git en la raíz del repo y devuelve el resultado (sin lanzar excepción)."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def protected_changes(config: dict) -> list[str]:
    """Rutas protegidas que difieren del tag base: versionadas cambiadas o nuevas sin versionar.

    `git diff <base>` compara el tag con el árbol de trabajo (incluye lo que ya está en
    commits posteriores al tag). Los ficheros nuevos sin versionar no salen en `git diff`, por
    eso se añade `git ls-files --others`. Lo ignorado por .gitignore (results.tsv…) no cuenta.
    """
    paths = config["protected"]
    diff = git("diff", "--name-only", config["base"], "--", *paths)
    if diff.returncode != 0:
        raise RuntimeError(f"git diff contra {config['base']!r} falló: {diff.stderr.strip()}")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths)
    return sorted(set(diff.stdout.split()) | set(untracked.stdout.split()))


def run_command(command: str, timeout: float) -> tuple[bool, str]:
    """Ejecuta un comando de la config. Devuelve (ha ido bien, salida completa)."""
    try:
        result = subprocess.run(
            shlex.split(command), cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, f"{command}: superó el límite de {timeout:.0f} s"
    return result.returncode == 0, result.stdout + result.stderr


def run_gate(config: dict) -> str | None:
    """Pasa la puerta en orden. Devuelve None si todo va bien o el motivo del fallo."""
    for command in config["gate"]:
        ok, output = run_command(command, config["timeouts"]["gate_seconds"])
        if not ok:
            tail = "\n".join(output.strip().splitlines()[-10:])
            return f"puerta rota en «{command}»\n{tail}"
    return None


def run_metric(config: dict) -> tuple[float | None, str]:
    """Ejecuta la métrica. Devuelve (valor o None, mensaje de error si lo hay)."""
    ok, output = run_command(config["metric"], config["timeouts"]["metric_seconds"])
    lines = output.strip().splitlines()
    if not ok or not lines:
        return None, "métrica fallida\n" + "\n".join(lines[-10:])
    try:
        return float(lines[-1]), ""
    except ValueError:
        return None, f"la última línea de la métrica no es un número: {lines[-1]!r}"


def read_rows() -> list[dict[str, str]]:
    """Filas de loop/results.tsv (lista vacía si aún no existe)."""
    if not RESULTS_PATH.is_file():
        return []
    with RESULTS_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_row(commit: str, value: float | None, status: str, description: str) -> None:
    """Añade una fila al TSV (crea la cabecera la primera vez)."""
    is_new = not RESULTS_PATH.is_file()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": commit,
                "valor": "-" if value is None else f"{value:.6f}",
                "estado": status,
                "descripcion": " ".join(description.split()),
            }
        )


def best_value(rows: list[dict[str, str]], objective: str) -> float:
    """Mejor valor entre las filas «base» y «keep» (máximo o mínimo según `objective`)."""
    values = [float(r["valor"]) for r in rows if r["estado"] in ("base", "keep")]
    return max(values) if objective == "max" else min(values)


def improves(value: float, best: float, objective: str, min_improvement: float) -> bool:
    """True si `value` mejora a `best` en más de `min_improvement`."""
    gain = value - best if objective == "max" else best - value
    return gain > min_improvement


def consecutive_discards(rows: list[dict[str, str]]) -> int:
    """Número de «discard» seguidos al final del TSV."""
    count = 0
    for row in reversed(rows):
        if row["estado"] != "discard":
            break
        count += 1
    return count


def head_commit() -> str:
    """Hash corto de HEAD."""
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def keep(config: dict, description: str) -> str:
    """Hace commit solo de las rutas editables. Devuelve el hash del commit nuevo."""
    git("add", "--", *config["editable"])
    result = git("commit", "-m", f"loop(mutation): {description}")
    if result.returncode != 0:
        raise RuntimeError(f"git commit falló: {result.stdout}{result.stderr}")
    return head_commit()


def discard(config: dict, description: str) -> None:
    """Guarda en un stash (no borra) los cambios de las rutas editables, incluidos los nuevos."""
    message = f"loop(mutation) descartado: {description}"
    git("stash", "push", "--include-untracked", "-m", message, "--", *config["editable"])


def stop(reason: str) -> int:
    """Escribe loop/STOP con el motivo, imprime PARADA y devuelve el código de salida 2."""
    STOP_PATH.write_text(reason + "\n", encoding="utf-8")
    print(f"PARADA: {reason}")
    return EXIT_STOP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Opciones de la CLI."""
    parser = argparse.ArgumentParser(description="Juez del bucle de mutación (keep/discard).")
    parser.add_argument(
        "description",
        nargs="?",
        default="",
        help="Qué test añade el experimento y qué fallo caza (va al commit y al TSV).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Aplica los pasos (a)–(f) y devuelve 0 (base/keep/discard), 1 (error) o 2 (PARADA)."""
    args = parse_args(argv)
    config = load_config()
    if STOP_PATH.is_file():
        print(f"PARADA: ya existe loop/STOP ({STOP_PATH.read_text(encoding='utf-8').strip()})")
        return EXIT_STOP

    # (a) Rutas protegidas
    try:
        changed = protected_changes(config)
    except RuntimeError as error:
        return stop(str(error))
    if changed:
        return stop("cambios en rutas protegidas: " + ", ".join(changed))

    rows = read_rows()
    is_base = not rows
    description = args.description or ("línea base" if is_base else "sin descripción")

    # (b) Puerta y (c) métrica
    problem = run_gate(config)
    value = None
    if problem is None:
        value, problem = run_metric(config)
        problem = problem or None
    if is_base:
        if problem:
            print(f"ERROR: no se puede fijar la línea base: {problem}")
            return 1
        append_row(head_commit(), value, "base", description)
        print(f"BASE: valor={value:.6f}")
        return 0

    # (d) Comparación y (e) keep/discard
    best = best_value(rows, config["objective"])
    if value is not None and improves(value, best, config["objective"], config["min_improvement"]):
        status, commit = "keep", keep(config, description)
    else:
        status, commit = "discard", head_commit()
        discard(config, description)
        if problem is None:
            problem = f"mejora insuficiente (≤ {config['min_improvement']})"

    # (f) Registro
    append_row(commit, value, status, description)
    shown = "-" if value is None else f"{value:.6f}"
    streak = consecutive_discards(read_rows())
    print(f"{status.upper()}: valor={shown} mejor_anterior={best:.6f} descartes_seguidos={streak}")
    if problem:
        print(f"motivo: {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
