# AGENTS.md — Reglas del repositorio para agentes

Codex lee este archivo automáticamente; Claude Code lo importa desde `CLAUDE.md`.
El contexto completo del proyecto está en [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md): léelo antes de cualquier tarea.

## El proyecto en una frase

Detector `healthy` / `diseased` de hojas de **tomate** con **visión por computador clásica**
(OpenCV + características diseñadas a mano + clasificadores tradicionales). Es un proyecto
**individual** de la asignatura de Visión por Computador (Universidad de Deusto): el alumno
tiene que poder entender, explicar y defender cada línea en la memoria y ante el profesor.

## Roles

| Quién | Rol | Hace | No hace |
|---|---|---|---|
| **Claude Code** | Implementador | Implementa una tarea de `docs/tasks/` en su propia rama, con tests y resultados reales, y abre el PR | Mergear a `main`; empezar la tarea siguiente |
| **Codex** | Revisor y refinador | Revisa el PR contra estas reglas y la tarea; corrige y pule en la misma rama con commits pequeños | Cambiar el enfoque, ampliar el alcance, mergear |
| **Rubba (alumno)** | Decide | Elige tareas, resuelve decisiones abiertas, entiende el código y mergea | — |

El ciclo completo está en [`docs/WORKFLOW.md`](docs/WORKFLOW.md) y el estado de las tareas en
[`docs/tasks/README.md`](docs/tasks/README.md).

## Reglas no negociables (todas las tareas)

1. **Nada de redes neuronales ni modelos preentrenados**, en ninguna etapa. Prohibido:
   `torch`, `tensorflow`, `keras`, `jax`, `transformers`, `timm`, `ultralytics`, `onnxruntime`,
   `cv2.dnn`, `sklearn.neural_network` (`MLPClassifier`), `rembg`, `segment_anything`, `mediapipe`
   y cualquier API externa de visión o diagnóstico. Permitido: OpenCV, NumPy, pandas,
   scikit-image, scikit-learn (SVM, kNN, árboles, Random Forest…), matplotlib.
   `tests/test_no_deep_learning.py` lo comprueba en CI.
2. **No inventar.** No afirmar que se ha descargado, ejecutado o medido algo que no ha ocurrido.
   Cada número que aparezca en documentación o PR tiene que salir de un comando repetible.
   Si algo no se pudo ejecutar (p. ej. faltan datos), se dice explícitamente.
3. **Datos.** `data/raw/` es de solo lectura: nunca modificar, mover ni borrar nada ahí.
   Nada de imágenes de datasets, modelos ni datos pesados en git (ver `.gitignore`; CI rechaza
   ficheros > 2 MB). Los ficheros derivados se regeneran con scripts, no se editan a mano.
4. **Sin fugas de datos.** Particiones `train`/`val`/`test` fijadas antes de ajustar nada;
   agrupadas por hoja (`leaf_group_id`) cuando se conozca; aumentos solo en `train`;
   escaladores, umbrales y selección de características ajustados solo con `train`
   (y elegidos con `val`); el `test` no se usa para decidir nada. PlantDoc es `external_test`
   y nunca se usa para entrenar ni ajustar.
5. **Reproducible.** Semillas fijas y declaradas; rutas relativas a la raíz del repo;
   parámetros por CLI (`argparse`) o en `configs/`. Ninguna ruta absoluta de la máquina del alumno.
6. **Alcance.** Hacer solo la tarea asignada. Lo que se detecte fuera de alcance se anota en el
   PR como «Pendiente», no se implementa.
7. **Legibilidad antes que ingenio.** Funciones pequeñas con docstring (qué hace, por qué y qué
   técnica del curso aplica), type hints, identificadores en inglés, comentarios y documentación
   en español. Sin abstracciones innecesarias.
8. **Honestidad técnica.** No presentar puntuaciones como probabilidades calibradas; no presentar
   una máscara de manchas como «la localización de la enfermedad» que ha usado el clasificador;
   no asumir que verde = sano ni que marrón/amarillo = enfermo.

## Comandos

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
ruff check . && ruff format --check .   # estilo
pytest                                   # tests (no necesitan los datasets)
python scripts/<script>.py --help        # cada script documenta sus opciones
```

## Estructura

```text
data/raw/          datasets originales (no versionado, solo lectura) — ver data/README.md
data/processed/    derivados regenerables (no versionado)
data/metadata.csv  manifiesto generado por scripts
configs/           mapeos de etiquetas y parámetros
scripts/           puntos de entrada CLI (auditoría, preparación, entrenamiento…)
src/               lógica reutilizable (preprocessing, segmentation, features, training, inference)
tests/             pytest; usan imágenes sintéticas, nunca los datasets reales
reports/           informes ligeros versionables (auditorías, figuras para la memoria)
experiments/       resultados de experimentos: parámetros, semilla, métricas, errores
docs/              flujo de trabajo, tareas y registro de decisiones
```

## Convenciones de git

- Una rama por tarea: `tNN-descripcion-corta` (p. ej. `t01-audit-dataset`).
- Commits: `TNN: <qué>` para la implementación; `review(TNN): <qué>` para los cambios de Codex.
- Un PR por tarea, rellenando `.github/pull_request_template.md`.
- **Nunca dos agentes trabajando a la vez en la misma rama.** Antes de tocar una rama: `git pull`.
- `main` solo cambia por merge del alumno.

## Review guidelines

Prioridades para cualquier revisión de este repositorio (Codex en GitHub o local):

- **P0** — uso de redes neuronales o modelos preentrenados (incluidos `cv2.dnn` y `MLPClassifier`).
- **P0** — fuga de datos: la misma hoja, imagen o casi-duplicado en más de un split; escalador,
  umbral o selección de características ajustados con `val`/`test`; hiperparámetros elegidos con
  `test`; PlantDoc usado para entrenar o ajustar.
- **P0** — escritura, movimiento o borrado dentro de `data/raw/`.
- **P0** — imágenes de datasets, modelos, ficheros pesados o secretos (`kaggle.json`, tokens) en el commit.
- **P0** — métricas o resultados en documentación/PR que no se correspondan con un comando ejecutado.
- **P1** — semillas sin fijar o splits no reproducibles; mapeo de etiquetas implícito o que pierde
  `disease_name`; carpetas sin mapear ignoradas en silencio.
- **P1** — métricas engañosas: solo accuracy con clases desequilibradas, falta de matriz de
  confusión/precision/recall/F1, recall de `diseased` sin reportar.
- **P1** — errores de visión por computador: BGR/RGB confundidos, rango de H en OpenCV (0–179)
  tratado como 0–360, tipos `uint8` desbordados, máscaras binarias con valores distintos de 0/255.
- **P1** — rutas absolutas; scripts sin `--help` o que fallan sin un mensaje útil cuando falta un dataset.
- **P1** — lógica nueva no trivial sin tests (mapeo de etiquetas, splits, características, métricas).
- **P1** — código opaco que el alumno no podría explicar cuando hay una alternativa clara;
  funciones públicas de `src/` o `scripts/` sin docstring.
- **P1** — trabajo fuera del alcance de la tarea.
- **P2** (no bloquea) — erratas en documentación, estilo, nombres mejorables.

### Si eres Codex refinando un PR en local

1. Lee la tarea (`docs/tasks/TNN-*.md`), su sección «Entrega» y la descripción del PR.
2. Revisa el diff contra `main` y clasifica los hallazgos con las prioridades de arriba.
3. Corrige P0 y P1 en la misma rama con commits `review(TNN): ...` pequeños y explicados.
   Ejecuta `ruff`, `pytest` y, si los datos están en `data/raw/`, vuelve a ejecutar los scripts
   afectados y actualiza los resultados reales.
4. No cambies el enfoque de la tarea. Si no estás de acuerdo con una decisión de diseño,
   explícalo en el PR en vez de reescribirla.
5. Deja en el PR un comentario «Revisión Codex» con: hallazgos (P0/P1/P2), qué has corregido,
   qué dejas para el alumno y cómo verificarlo.
