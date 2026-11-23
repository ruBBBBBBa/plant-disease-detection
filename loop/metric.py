"""Métrica del bucle de mutación: fracción de mutantes muertos de UN módulo de `src/`.

Uso:
  python loop/metric.py                          # módulo de loop/config.json (target_module)
  python loop/metric.py --module src/segmentation.py

Qué hace, en orden:
1. Crea una copia temporal en `loop/.staging/` (sin versionar). mutmut 3.8.1 rechaza los
   proyectos que importan `src` como paquete (sus nombres de módulo no pueden empezar por
   `src.`), así que en la copia `src/` pasa a llamarse `pdd/` y se reescriben los
   `from src …` / `import src …`. El `src/` real nunca se toca.
2. Ejecuta los tests en la copia **sin mutar**. Si alguno falla, la medición se anula con un
   error: una cifra calculada sobre tests rotos no significaría nada.
3. Ejecuta mutmut solo sobre el módulo objetivo.
4. Escribe `loop/survivors.txt` con los mutantes vivos y su ruta real (`src/<módulo>.py:línea`).
5. Imprime en la **última línea** la fracción de mutantes muertos (D039):
   (killed + timeout + segfault) / (total − skipped). «no tests» cuenta como vivo.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "loop" / ".staging"
SURVIVORS = REPO_ROOT / "loop" / "survivors.txt"
PACKAGE = "pdd"
COPIED_DIRS = ("tests", "scripts", "configs")
KILLED_STATUSES = {"killed", "timeout", "segfault"}

# Los tests del propio bucle no prueban src/ y necesitan loop/: no se copian a la copia temporal.
IGNORED_IN_STAGING = shutil.ignore_patterns("__pycache__", "test_loop_*.py")

STAGING_PYPROJECT = """\
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.mutmut]
source_paths = ["."]
only_mutate = ["{package}/{module_file}"]
also_copy = ["configs/", "scripts/", "{package}/"]
pytest_add_cli_args_test_selection = ["tests/"]
"""


def rewrite_imports(text: str) -> str:
    """Cambia `from src …` / `import src …` por el paquete de la copia (`pdd`).

    Solo toca imports al principio de línea (con o sin sangría), no cadenas ni comentarios.
    """
    return re.sub(r"^(\s*)(from|import) src\b", rf"\1\2 {PACKAGE}", text, flags=re.MULTILINE)


def build_staging(module_file: str) -> None:
    """Crea `loop/.staging/` desde cero: `src/` → `pdd/`, tests, scripts y configs."""
    shutil.rmtree(STAGING, ignore_errors=True)
    shutil.copytree(REPO_ROOT / "src", STAGING / PACKAGE, ignore=IGNORED_IN_STAGING)
    for name in COPIED_DIRS:
        shutil.copytree(REPO_ROOT / name, STAGING / name, ignore=IGNORED_IN_STAGING)
    for path in STAGING.rglob("*.py"):
        path.write_text(rewrite_imports(path.read_text(encoding="utf-8")), encoding="utf-8")
    (STAGING / "pyproject.toml").write_text(
        STAGING_PYPROJECT.format(package=PACKAGE, module_file=module_file), encoding="utf-8"
    )


def run_in_staging(args: list[str], log_name: str) -> subprocess.CompletedProcess[str]:
    """Ejecuta un comando dentro de la copia y guarda su salida en un log de la copia."""
    result = subprocess.run(args, cwd=STAGING, capture_output=True, text=True)
    (STAGING / log_name).write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def check_clean_tests() -> None:
    """Pasa los tests en la copia sin mutar; si alguno falla, anula la medición."""
    result = run_in_staging(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings"],
        "clean-tests.log",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        sys.exit(
            "Medición anulada: los tests fallan en la copia sin mutar "
            f"(loop/.staging/clean-tests.log).\n{tail}"
        )


def parse_results(text: str) -> dict[str, str]:
    """Convierte la salida de `mutmut results --all true` en {mutante: estado}."""
    statuses = {}
    for line in text.splitlines():
        name, sep, status = line.strip().partition(": ")
        if sep:
            statuses[name] = status
    return statuses


def run_mutmut() -> dict[str, str]:
    """Ejecuta mutmut en la copia y devuelve el estado de cada mutante."""
    run = run_in_staging([sys.executable, "-m", "mutmut", "run"], "mutmut-run.log")
    if run.returncode != 0:
        sys.exit("Medición anulada: mutmut run falló (loop/.staging/mutmut-run.log).")
    results = run_in_staging(
        [sys.executable, "-m", "mutmut", "results", "--all", "true"], "mutmut-results.log"
    )
    statuses = parse_results(results.stdout)
    if not statuses:
        sys.exit("Medición anulada: mutmut no devolvió mutantes (loop/.staging/).")
    return statuses


def mutation_score(statuses: dict[str, str]) -> float:
    """Fracción de mutantes muertos: (killed + timeout + segfault) / (total − skipped).

    Un segfault significa que el mutante hizo caer el test: se ha detectado. «no tests»
    (ningún test ejecuta esa función) cuenta como vivo, porque es justo lo que hay que cubrir.
    """
    tested = [s for s in statuses.values() if s != "skipped"]
    if not tested:
        raise ValueError("No hay mutantes evaluados")
    return sum(s in KILLED_STATUSES for s in tested) / len(tested)


def function_spans(source: str) -> dict[str, tuple[int, int]]:
    """{nombre mutmut de la función: (primera, última línea)} de un fichero de `src/`.

    mutmut llama `x_<función>` a las funciones y `xǁ<Clase>ǁ<método>` a los métodos.
    """
    spans = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            spans[f"x_{node.name}"] = (node.lineno, node.end_lineno or node.lineno)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    key = f"xǁ{node.name}ǁ{item.name}"
                    spans[key] = (item.lineno, item.end_lineno or item.lineno)
    return spans


def changed_lines(diff: str) -> tuple[str, str]:
    """Primera línea quitada (`-`) y primera añadida (`+`) del diff de un mutante."""
    removed = added = ""
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-") and not removed:
            removed = line[1:]
        elif line.startswith("+") and not added:
            added = line[1:]
    return removed, added


def locate(removed: str, span: tuple[int, int], source_lines: list[str]) -> int:
    """Línea real del cambio: la línea de la función que coincide con la línea quitada.

    Si hay varias iguales se devuelve la primera; si no se encuentra, el inicio de la función.
    """
    first, last = span
    for number in range(first, last + 1):
        if source_lines[number - 1].rstrip() == removed.rstrip():
            return number
    return first


def mutant_diff(name: str, module_file: str) -> str:
    """Diff de un mutante con la API de mutmut 3.8.1 (versión fijada en requirements-dev.txt).

    ponytail: usa una función interna de mutmut porque `mutmut show` tarda ~0,5 s por mutante;
    si se actualiza mutmut y falla, cambiar por `subprocess` a `mutmut show <name>`.
    """
    from mutmut.mutation.diff_apply import get_diff_for_mutant

    previous = Path.cwd()
    os.chdir(STAGING)
    try:
        return get_diff_for_mutant(name, path=Path(PACKAGE) / module_file)
    finally:
        os.chdir(previous)


def mutmut_function_key(name: str, module_name: str) -> str:
    """`pdd.features.x_leaf_mask__mutmut_3` → `x_leaf_mask`."""
    return name.removeprefix(f"{PACKAGE}.{module_name}.").rsplit("__mutmut_", 1)[0]


def write_survivors(module: str, statuses: dict[str, str], score: float) -> int:
    """Escribe `loop/survivors.txt` con los mutantes vivos en rutas reales. Devuelve cuántos."""
    module_file = Path(module).name
    source = (REPO_ROOT / module).read_text(encoding="utf-8")
    source_lines = source.splitlines()
    spans = function_spans(source)
    alive = sorted(n for n, s in statuses.items() if s not in KILLED_STATUSES | {"skipped"})

    rows = []
    for name in alive:
        span = spans.get(mutmut_function_key(name, Path(module).stem), (1, len(source_lines)))
        removed, added = changed_lines(mutant_diff(name, module_file))
        line = locate(removed, span, source_lines)
        rows.append(
            f"{module}:{line}\t{statuses[name]}\t{name}\n"
            f"    - {removed.strip()}\n    + {added.strip()}"
        )

    header = (
        f"# Mutantes vivos de {module} ({time.strftime('%Y-%m-%d %H:%M')})\n"
        f"# muertos = {score:.4f}; vivos = {len(alive)} de {len(statuses)}\n"
        "# Formato: ruta:línea<TAB>estado<TAB>mutante; debajo, línea original (-) y mutada (+)\n"
    )
    SURVIVORS.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return len(alive)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Opciones de la CLI; el módulo por defecto sale de `loop/config.json`."""
    config = json.loads((REPO_ROOT / "loop" / "config.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Fracción de mutantes muertos de un módulo.")
    parser.add_argument(
        "--module",
        default=config["target_module"],
        help="Módulo de src/ a mutar (por defecto, target_module de loop/config.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Construye la copia, comprueba los tests, muta el módulo e imprime la fracción."""
    args = parse_args(argv)
    module = Path(args.module).as_posix()
    if not (module.startswith("src/") and module.endswith(".py")):
        sys.exit(f"Error: --module debe ser un fichero src/<módulo>.py, no {module!r}")
    if not (REPO_ROOT / module).is_file():
        sys.exit(f"Error: no existe {module}")

    start = time.perf_counter()
    build_staging(Path(module).name)
    check_clean_tests()
    statuses = run_mutmut()
    score = mutation_score(statuses)
    n_alive = write_survivors(module, statuses, score)

    counts = {s: list(statuses.values()).count(s) for s in sorted(set(statuses.values()))}
    print(f"{module}: {counts}")
    print(f"vivos: {n_alive} (lista en loop/survivors.txt); {time.perf_counter() - start:.0f} s")
    print(f"{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
