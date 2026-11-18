# T00b — Guardas de Claude Code e infraestructura del bucle de mutación

- **Estado:** En revisión
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

- [x] `.claude/settings.json` con exactamente los tres `deny` pedidos.
- [x] `loop/metric.py` mide un solo módulo, anula la medición si los tests fallan en la copia sin
      mutar, imprime la fracción en la última línea y escribe `loop/survivors.txt` con rutas reales.
- [x] `loop/evaluate.py` sigue el orden: protegidos → puerta → métrica → comparación →
      keep/discard → fila en `loop/results.tsv`; la primera ejecución es «base».
- [x] `loop/keep_going.py` bloquea la parada mientras haya trabajo y deja parar en los límites.
- [x] `tests/test_loop_evaluate.py` cubre keep, discard por mejora insuficiente, discard por
      puerta rota y PARADA al tocar un archivo protegido.
- [x] `ruff check .`, `ruff format --check .` y `pytest` en verde.
- [x] Línea base real de `src/features.py` y `src/segmentation.py` (porcentaje, supervivientes y
      tiempo) en «Entrega».

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados** (Python 3.15.2 local, mutmut 3.8.1):
  ```bash
  pip install -r requirements-dev.txt           # añade mutmut==3.8.1
  python loop/metric.py --module src/features.py
  python loop/metric.py --module src/segmentation.py
  ruff check . && ruff format --check . && pytest -q
  # comprobación manual de la anulación: test que falla a propósito en tests/ (luego borrado)
  python loop/metric.py --module src/segmentation.py   # → «Medición anulada…», código 1
  ```
- **Resultados reales (línea base, solo la métrica; el bucle no se ha lanzado):**

  | Módulo | Mutantes | killed | segfault | survived | no tests | Muertos (D039) | Vivos | Tiempo |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | `src/features.py` | 837 | 448 | 4 | 341 | 44 | **0,540024** | 385 | 77 s |
  | `src/segmentation.py` | 593 | 388 | 3 | 202 | 0 | **0,659359** | 202 | 62 s |

  Puerta completa (`ruff check`, `ruff format --check`, `pytest -q`): 102 tests en verde, 15 s.
  Timeouts fijados en `loop/config.json` con margen sobre lo medido: puerta 120 s, métrica 300 s.

  Supervivientes por función (los que más tienen):
  - `features.py`: `_zero_features` 97, `color_features` 56, `glcm_features` 36,
    `spot_features` 31, `quality_features` 26, `prepare_mask` 25; sin ningún test que las ejecute:
    `leaf_mask` 18, `erode_mask` 13, `dark_fraction` 13.
  - `segmentation.py`: `channel` 43, `segment_kmeans` 39, `segment_grabcut` 27,
    `rim_fraction` 10, `fill_holes` 10.
- **Hallazgos:**
  1. mutmut 3.8.1 no funciona tal cual en este repo: rechaza los módulos que empiezan por `src.`
     (`assert not name.startswith("src.")` en `mutmut/stats.py`). mutmut 2.5.1 no arranca en
     Python 3.15 (pony ORM). Solución aceptada (D039): copia temporal con `src/` → `pdd/`.
  2. La copia temporal pasa los tests sin mutar antes de medir; si alguno falla, la medición se
     anula (comprobado a mano con un test roto a propósito).
  3. `survivors.txt` da la línea real: p. ej. `src/segmentation.py:56`, donde
     `255 - cv2.cvtColor(...)[:, :, 1]` sobrevive mutado a `255 + …` (el canal a* invertido no lo
     detecta ningún test).
  4. Con `min_improvement = 0,005`, un experimento debe matar ≥ 5 mutantes en `features.py`
     (837) y ≥ 3 en `segmentation.py` (593) para quedarse.
  5. La copia normal estaba en `t06-plantdoc-external` con cambios sin commitear; para no
     tocarlos, T00b se hizo en un worktree aparte (`../pdd-t00b`) desde `origin/main`.
- **Decisiones tomadas:**
  1. D038 (Propuesta): el bucle de mutación y sus reglas.
  2. D039 (Aceptada por Rubba): métrica `(killed + timeout + segfault) / (total − skipped)`,
     «no tests» como vivo; mutmut 3.8.1 fijado sobre la copia temporal.
  3. La guarda de protegidos también detecta ficheros nuevos sin versionar
     (`git ls-files --others`), no solo `git diff`.
  4. `results.tsv`, `survivors.txt`, `STOP` y `loop/.staging/` van al `.gitignore`.
  5. Los tests del propio bucle (`tests/test_loop_*.py`) no se copian a la copia temporal:
     no prueban `src/`.
  6. Se usan D038/D039 porque la rama de T06 ya ocupa D031–D037.
- **Pendientes:**
  1. Elegir el módulo del primer bucle (propuesta: `src/features.py`) y lanzarlo.
  2. `metric.py` usa una función interna de mutmut (`get_diff_for_mutant`) para sacar los diffs
     rápido; si se actualiza mutmut, revisar (marcado con `ponytail:` en el código).
  3. CI usa Python 3.12 y no ejecuta mutmut; solo instala la dependencia.
