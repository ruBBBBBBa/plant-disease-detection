"""Hook «Stop» de Claude Code para el bucle de mutación.

Claude Code lo ejecuta cada vez que el agente quiere terminar. Si devuelve
{"decision": "block", "reason": …}, el agente no para y recibe `reason` como instrucción.
Bloquea mientras:
  - no haya línea base en loop/results.tsv,
  - no se haya llegado a `max_iterations` experimentos,
  - no haya `max_consecutive_discards` descartes seguidos,
  - y no exista loop/STOP.
En cualquier otro caso no imprime nada y el agente puede parar.

Se registra en loop/claude-settings.json como:
  python "$CLAUDE_PROJECT_DIR"/loop/keep_going.py
"""

from __future__ import annotations

import json
import sys

# Al ejecutarse como script, la carpeta loop/ está en sys.path y evaluate.py se importa directo.
from evaluate import STOP_PATH, consecutive_discards, load_config, read_rows


def reason_to_continue(rows: list[dict[str, str]], config: dict, stop_exists: bool) -> str | None:
    """Motivo para no dejar parar al agente, o None si ya puede parar."""
    if stop_exists:
        return None
    if not any(row["estado"] == "base" for row in rows):
        return 'Falta la línea base: ejecuta python loop/evaluate.py "línea base".'
    iterations = sum(row["estado"] != "base" for row in rows)
    if iterations >= config["max_iterations"]:
        return None
    if consecutive_discards(rows) >= config["max_consecutive_discards"]:
        return None
    return (
        f"Llevas {iterations} de {config['max_iterations']} experimentos. Elige el siguiente "
        'superviviente de loop/survivors.txt, añade su test y ejecuta python loop/evaluate.py "…".'
    )


def main() -> int:
    """Imprime el bloqueo en JSON si el bucle debe seguir."""
    reason = reason_to_continue(read_rows(), load_config(), STOP_PATH.is_file())
    if reason:
        print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
