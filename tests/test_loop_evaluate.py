"""Tests del juez del bucle de mutación (loop/evaluate.py) y del hook loop/keep_going.py.

Se monta un repo de git sintético en tmp_path con copias de los dos scripts, una puerta falsa
(falla si existe tests/broken.txt) y una métrica falsa (imprime el número de tests/value.txt).
Así se prueban keep, discard y PARADA sin ruff, pytest ni mutmut de verdad.
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = shlex.quote(sys.executable)

FAKE_GATE = """\
import sys
from pathlib import Path
sys.exit(1 if Path("tests/broken.txt").exists() else 0)
"""

FAKE_METRIC = """\
from pathlib import Path
print("ruido de la métrica")
print(Path("tests/value.txt").read_text().strip())
"""


def git(repo: Path, *args: str) -> str:
    """Ejecuta git en el repo sintético y devuelve stdout."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout


@pytest.fixture
def loop_repo(tmp_path: Path) -> Path:
    """Repo con loop/, src/ protegido, tests/ editable y el tag base `loop-base`."""
    repo = tmp_path / "repo"
    (repo / "loop").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    for name in ("evaluate.py", "keep_going.py"):
        shutil.copy(REPO_ROOT / "loop" / name, repo / "loop" / name)
    config = {
        "base": "loop-base",
        "editable": ["tests/"],
        "protected": ["src/", "loop/", ":(exclude)loop/RESUMEN.md"],
        "gate": [f"{PYTHON} gate.py"],
        "metric": f"{PYTHON} metric.py",
        "objective": "max",
        "min_improvement": 0.005,
        "timeouts": {"gate_seconds": 60, "metric_seconds": 60},
        "max_iterations": 3,
        "max_consecutive_discards": 2,
    }
    (repo / "loop" / "config.json").write_text(json.dumps(config))
    (repo / "gate.py").write_text(FAKE_GATE)
    (repo / "metric.py").write_text(FAKE_METRIC)
    (repo / "src" / "module.py").write_text("VALUE = 1\n")
    (repo / "tests" / "value.txt").write_text("0.500\n")
    (repo / ".gitignore").write_text("loop/results.tsv\nloop/STOP\n__pycache__/\n")

    git(repo, "init", "-q")
    git(repo, "config", "user.email", "loop@example.com")
    git(repo, "config", "user.name", "loop")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "inicio")
    git(repo, "tag", "loop-base")
    return repo


def evaluate(repo: Path, description: str = "") -> subprocess.CompletedProcess[str]:
    """Ejecuta el juez del repo sintético."""
    args = [sys.executable, str(repo / "loop" / "evaluate.py")]
    if description:
        args.append(description)
    return subprocess.run(args, cwd=repo, capture_output=True, text=True)


def rows(repo: Path) -> list[dict[str, str]]:
    """Filas de loop/results.tsv."""
    with (repo / "loop" / "results.tsv").open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def keep_going(repo: Path) -> str:
    """Salida del hook Stop en el repo sintético."""
    script = repo / "loop" / "keep_going.py"
    result = subprocess.run([sys.executable, str(script)], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_first_run_is_base_and_then_improvement_is_kept(loop_repo: Path) -> None:
    """La primera ejecución fija la base; una mejora suficiente hace commit solo de tests/."""
    first = evaluate(loop_repo)
    assert first.returncode == 0, first.stdout + first.stderr
    assert [r["estado"] for r in rows(loop_repo)] == ["base"]
    assert rows(loop_repo)[0]["valor"] == "0.500000"

    (loop_repo / "tests" / "value.txt").write_text("0.600\n")
    (loop_repo / "tests" / "test_new.py").write_text("def test_x():\n    assert 1 + 1 == 2\n")
    second = evaluate(loop_repo, "test nuevo que caza un desborde")
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.startswith("KEEP")

    last = rows(loop_repo)[-1]
    assert last["estado"] == "keep" and last["valor"] == "0.600000"
    assert git(loop_repo, "log", "-1", "--format=%s").strip() == (
        "loop(mutation): test nuevo que caza un desborde"
    )
    committed = git(loop_repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == ["tests/test_new.py", "tests/value.txt"]
    assert git(loop_repo, "status", "--porcelain") == ""


def test_insufficient_improvement_is_discarded_into_stash(loop_repo: Path) -> None:
    """Una mejora ≤ min_improvement se descarta: tests/ vuelve al commit y queda en un stash."""
    evaluate(loop_repo)
    head = git(loop_repo, "rev-parse", "HEAD")
    (loop_repo / "tests" / "value.txt").write_text("0.503\n")
    (loop_repo / "tests" / "test_new.py").write_text("def test_x():\n    pass\n")

    result = evaluate(loop_repo, "mejora pequeña")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DISCARD")
    assert "descartes_seguidos=1" in result.stdout

    assert rows(loop_repo)[-1]["estado"] == "discard"
    assert git(loop_repo, "rev-parse", "HEAD") == head
    assert (loop_repo / "tests" / "value.txt").read_text() == "0.500\n"
    assert not (loop_repo / "tests" / "test_new.py").exists()
    assert "mejora pequeña" in git(loop_repo, "stash", "list")


def test_broken_gate_is_discarded_without_running_metric(loop_repo: Path) -> None:
    """Si la puerta falla se descarta aunque la métrica fuera a mejorar; el valor queda «-»."""
    evaluate(loop_repo)
    (loop_repo / "tests" / "value.txt").write_text("0.900\n")
    (loop_repo / "tests" / "broken.txt").write_text("rompe la puerta\n")

    result = evaluate(loop_repo, "test que rompe ruff")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DISCARD")
    assert "puerta rota" in result.stdout

    last = rows(loop_repo)[-1]
    assert last["estado"] == "discard" and last["valor"] == "-"
    assert not (loop_repo / "tests" / "broken.txt").exists()


@pytest.mark.parametrize("change", ["modify_tracked", "add_untracked"])
def test_touching_protected_path_stops_the_loop(loop_repo: Path, change: str) -> None:
    """Cambiar src/ (o crear un fichero nuevo ahí) da PARADA, código 2 y loop/STOP."""
    evaluate(loop_repo)
    if change == "modify_tracked":
        (loop_repo / "src" / "module.py").write_text("VALUE = 2\n")
    else:
        (loop_repo / "src" / "extra.py").write_text("X = 0\n")
    (loop_repo / "tests" / "value.txt").write_text("0.900\n")

    result = evaluate(loop_repo, "trampa")
    assert result.returncode == 2
    assert result.stdout.startswith("PARADA")
    assert (loop_repo / "loop" / "STOP").is_file()
    assert [r["estado"] for r in rows(loop_repo)] == ["base"]


def test_resumen_is_not_protected(loop_repo: Path) -> None:
    """loop/RESUMEN.md es la única excepción de loop/: escribirlo no para el bucle."""
    evaluate(loop_repo)
    (loop_repo / "loop" / "RESUMEN.md").write_text("# Resumen\n")
    result = evaluate(loop_repo, "sin cambios")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (loop_repo / "loop" / "STOP").exists()


def test_keep_going_blocks_until_limits(loop_repo: Path) -> None:
    """El hook bloquea sin base y con trabajo pendiente, y deja parar tras 2 descartes seguidos."""
    blocked = json.loads(keep_going(loop_repo))
    assert blocked["decision"] == "block" and "línea base" in blocked["reason"]

    evaluate(loop_repo)
    assert json.loads(keep_going(loop_repo))["decision"] == "block"

    evaluate(loop_repo, "sin mejora 1")
    evaluate(loop_repo, "sin mejora 2")
    assert keep_going(loop_repo) == ""


def test_keep_going_allows_stop_when_stop_file_exists(loop_repo: Path) -> None:
    """Con loop/STOP el hook deja parar aunque falte la línea base."""
    (loop_repo / "loop" / "STOP").write_text("parada manual\n")
    assert keep_going(loop_repo) == ""
