# T00b — Guardas de Claude Code e infraestructura del bucle de mutación

- **Estado:** En curso
- **Rama:** `t00b-guardas-claude`
- **Tipo:** infraestructura (fuera de las fases de visión; no cambia `src/`, `configs/` ni `data/`)
- **Referencias:** `docs/WORKFLOW.md` («Bucle de mutación»), D038 y D039.

## Objetivo

1. Poner guardas versionadas a Claude Code para que no pueda editar `data/raw/`, hacer push a
   `main` ni mergear PRs.
2. Preparar (sin lanzarlo) un **bucle de mutación**: un agente añade tests para subir el
   porcentaje de mutantes muertos de **un** módulo de `src/`, con un juez automático que solo
   se queda con los tests que mejoran la métrica y para si se toca algo protegido.

Las pruebas de mutación miden si los tests detectan cambios pequeños en el código (cambiar un
`<` por `<=`, un `0` por `1`…). Un mutante «superviviente» indica que ningún test notaría ese fallo.

## Alcance

- `.claude/settings.json` solo con `permissions.deny` (sin hooks).
- `loop/`: `config.json`, `metric.py`, `evaluate.py`, `keep_going.py`, `claude-settings.json`,
  `program.md`.
- `tests/test_loop_evaluate.py` (juez sobre un repo de git sintético con métrica falsa) y
  `tests/test_loop_metric.py` (piezas puras de la métrica).
- `mutmut` en `requirements-dev.txt`; lo que genera, en `.gitignore`.
- `docs/WORKFLOW.md` (sección «Bucle de mutación») y `docs/DECISIONS.md`.
- Medir la línea base de `src/features.py` y `src/segmentation.py` (la métrica, no el bucle).

## Fuera de alcance

Lanzar el bucle; tocar `src/`, `configs/` o `data/`; añadir tests de visión.

## Criterios de aceptación

- [ ] `.claude/settings.json` con exactamente los tres `deny` pedidos.
- [ ] `loop/metric.py` mide un solo módulo, anula la medición si los tests fallan en la copia sin
      mutar, imprime la fracción en la última línea y escribe `loop/survivors.txt` con rutas reales.
- [ ] `loop/evaluate.py` sigue el orden: protegidos → puerta → métrica → comparación →
      keep/discard → fila en `loop/results.tsv`; la primera ejecución es «base».
- [ ] `loop/keep_going.py` bloquea la parada mientras haya trabajo y deja parar en los límites.
- [ ] `tests/test_loop_evaluate.py` cubre keep, discard por mejora insuficiente, discard por
      puerta rota y PARADA al tocar un archivo protegido.
- [ ] `ruff check .`, `ruff format --check .` y `pytest` en verde.
- [ ] Línea base real de `src/features.py` y `src/segmentation.py` (porcentaje, supervivientes y
      tiempo) en «Entrega».

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**
- **Resultados reales:**
- **Hallazgos:**
- **Decisiones tomadas:**
- **Pendientes:**
