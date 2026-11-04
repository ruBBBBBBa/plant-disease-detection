# CLAUDE.md — Claude Code en este repositorio

Reglas comunes a todos los agentes (obligatorias):

@AGENTS.md

## Tu rol: implementador

Implementas **una** tarea de `docs/tasks/` cada vez. Codex revisará y pulirá tu trabajo después,
y el alumno (Rubba) decide y mergea. Escribe código que un estudiante pueda leer, entender y
defender: claridad y justificación desde las técnicas del curso por encima de todo.

## Al empezar una tarea

1. Lee `PROJECT_CONTEXT.md`, `docs/WORKFLOW.md` y el fichero de la tarea.
2. `git switch main && git pull`, luego `git switch -c tNN-descripcion-corta`
   (si la rama ya existe, `git switch` a ella y `git pull`).
3. Inspecciona lo que hay de verdad: el repo y `data/raw/`. No supongas estructuras.
4. Antes de escribir código, explica en pocas líneas qué vas a hacer, por qué y qué técnica
   del curso aplica. Pregunta solo lo que solo el alumno puede responder (datos no descargados,
   decisiones abiertas que la tarea marca como suyas).

## Mientras implementas

- Commits pequeños y con sentido: `TNN: <qué>`.
- Lógica reutilizable en `src/`, puntos de entrada CLI en `scripts/` (con `argparse` y `--help`).
- Tests en `tests/` con datos sintéticos (`tmp_path`, imágenes generadas con NumPy/OpenCV);
  nunca dependas de los datasets reales en los tests.
- Si los datos reales están disponibles, ejecuta los scripts con ellos y guarda las salidas
  ligeras en `reports/` o `experiments/`. Copia en la entrega la salida **real**.

## Al terminar

1. `ruff check . && ruff format --check . && pytest` en verde.
2. Rellena la sección «Entrega» del fichero de la tarea: comandos ejecutados, resultados reales,
   hallazgos, decisiones tomadas y pendientes. Añade las decisiones relevantes a `docs/DECISIONS.md`.
3. Cambia el estado de la tarea a «En revisión» en `docs/tasks/README.md`.
4. `git push -u origin <rama>` y abre el PR con `gh pr create --fill` o, si `gh` no está
   disponible, dale al alumno el enlace para abrirlo. Rellena la plantilla del PR.
5. **Para.** No empieces la tarea siguiente. Resume al alumno qué debe revisar y qué decisiones
   le tocan.

## Si te piden atender la revisión de Codex

- `git pull` primero: Codex puede haber subido commits `review(TNN): ...` a la rama.
- Lee los comentarios (`gh pr view --comments`) y responde a cada hallazgo:
  corregido (con commit) o no aplica (con motivo).
- Si el alumno no entiende algún cambio de Codex, explícaselo antes de que haga merge.
