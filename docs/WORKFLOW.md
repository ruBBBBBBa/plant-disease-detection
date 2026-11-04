# Flujo de trabajo: Claude Code implementa, Codex revisa, Rubba decide

## Idea

Cada unidad de trabajo es una **tarea** en `docs/tasks/TNN-*.md` con objetivo, alcance,
criterios de aceptación y una sección «Entrega». Cada tarea pasa por el mismo ciclo y acaba en
un PR que solo el alumno mergea. Los agentes nunca trabajan a la vez sobre la misma rama.

```text
 Rubba            Claude Code                  Codex                     Rubba
 elige TNN  ──►  rama tNN-… → implementa  ──►  revisa diff vs main  ──►  lee, entiende,
                 tests + ejecución real         corrige P0/P1 en la       decide y mergea
                 rellena «Entrega», abre PR     misma rama + comentario   (squash) → TNN+1
                          ▲                              │
                          └──── si hay que rehacer algo ─┘
```

Estados de una tarea (en `docs/tasks/README.md`):
`Pendiente` → `En curso` (Claude Code) → `En revisión` (Codex) → `Lista para merge` (Rubba) → `Hecha`.

## Preparación (una vez)

1. Clonar el repo en el portátil y crear el entorno:
   ```bash
   git clone https://github.com/ruBBBBBBa/plant-disease-detection.git
   cd plant-disease-detection
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   pytest
   ```
2. Descargar los datos de tomate según [`data/README.md`](../data/README.md) (no van a git).
3. **Claude Code**: abrirlo en la raíz del repo (`claude` en la terminal o la extensión de VS Code).
   Lee `CLAUDE.md` solo, que a su vez importa `AGENTS.md`.
4. **Codex**: CLI (`codex`) o extensión de VS Code en la raíz del repo. Lee `AGENTS.md` solo.
5. Opcional, revisión en GitHub: en la configuración de Codex cloud, conectar este repositorio y
   activar *Code review*. Así se puede pedir una revisión comentando `@codex review` en el PR.
6. Recomendado: `gh auth login` para que Claude Code pueda abrir PRs desde la terminal.

## El ciclo de una tarea, paso a paso

### 1. Implementación — Claude Code

Prompt (cambia el número y el nombre de la tarea):

```text
Lee CLAUDE.md, PROJECT_CONTEXT.md y docs/tasks/T01-descarga-y-auditoria.md.
Implementa la tarea T01 siguiendo docs/WORKFLOW.md. Antes de escribir código, explícame el plan
y pregúntame solo lo imprescindible. Ejecuta todo con los datos reales de data/raw/, rellena la
sección «Entrega» de la tarea, abre el PR y detente.
```

Resultado esperado: rama `tNN-…` subida, PR abierto, «Entrega» rellenada con salidas reales.

### 2. Revisión y refinado — Codex

**Opción A (recomendada, en local, porque puede ejecutar con los datos):**

```bash
git switch t01-audit-dataset && git pull
codex
```

Dentro de Codex, escribe `/review` y elige *Review against a base branch* → `main`.
Después, pídele que aplique las correcciones:

```text
Actúa como revisor de este repo (AGENTS.md, «Review guidelines» y «Si eres Codex refinando un PR
en local»). Corrige en esta rama los hallazgos P0 y P1 de la revisión para la tarea T01
(docs/tasks/T01-descarga-y-auditoria.md), con commits `review(T01): ...`. Ejecuta ruff, pytest y
los scripts afectados con los datos reales, haz push y deja en el PR un comentario
«Revisión Codex» con hallazgos, cambios y lo que dejas para mí. No cambies el enfoque de la tarea.
```

Si Codex no puede comentar en GitHub desde tu terminal, que te dé el texto del comentario y lo pegas tú.

**Opción B (en GitHub, sin ejecutar nada):** comenta `@codex review` en el PR. Codex publica una
revisión centrada en P0/P1 siguiendo `AGENTS.md`. Si comentas `@codex` con otra petición (p. ej.
pedirle que corrija un hallazgo), lanza una tarea en la nube con el PR como contexto; ten en
cuenta que en la nube **no tiene los datasets**, así que no puede volver a ejecutar los scripts.

### 3. Cierre — Rubba (+ Claude Code si hace falta)

- Si Codex ha dejado hallazgos sin resolver o no estás de acuerdo con algo, vuelve a Claude Code:
  ```text
  Atiende la revisión de Codex del PR de la tarea T01 (git pull primero). Responde a cada
  hallazgo: corregido o no aplica, con motivo. Explícame cualquier cambio de Codex que afecte
  a cómo funciona el algoritmo.
  ```
- Lee el diff completo. Si no puedes explicar una parte, pide que te la expliquen antes de mergear:
  vas a tener que defenderlo en la memoria.
- Mergea con *Squash and merge*, marca la tarea como `Hecha` y toma las decisiones que la tarea deja abiertas.
- Apunta en `docs/DECISIONS.md` las decisiones que tomes tú.

## Reglas de convivencia entre agentes

- Turnos: nunca Claude Code y Codex a la vez en la misma rama. Siempre `git pull` antes de empezar.
- Commits identificables: `TNN: …` (Claude Code) y `review(TNN): …` (Codex).
- La tarea es el contrato: si un agente quiere cambiar el alcance, lo propone en el PR y decide Rubba.
- Las tareas futuras se detallan cuando la anterior está cerrada, con los resultados reales delante.
