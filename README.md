# Detección de posibles enfermedades en hojas de tomate con visión por computador clásica

Proyecto individual de *Introducción a la Visión por Computador* (Universidad de Deusto).

Dada una fotografía de una hoja de tomate, el sistema indicará si visualmente parece **sana**
o con **posibles síntomas de enfermedad**, mostrando las evidencias visuales (color, manchas,
textura, regiones) que han influido en el resultado. No es un diagnóstico fitopatológico.

**Restricción del proyecto:** solo visión por computador tradicional. Sin redes neuronales,
CNN, transformers ni modelos preentrenados en ninguna etapa. Se permiten clasificadores clásicos
(SVM, kNN, árboles, Random Forest) sobre características diseñadas y extraídas explícitamente.

Contexto completo, alcance y criterios: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Estado

**Fase 1 — preparación y auditoría de datos (tomate).** Todavía no hay segmentación,
clasificador ni resultados experimentales. Tablero de tareas: [`docs/tasks/README.md`](docs/tasks/README.md).

## Instalación

Requiere Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Datos

Los datasets no se versionan. Fuentes, estructura comprobada y comandos de descarga (solo tomate):
[`data/README.md`](data/README.md).

- **PlantVillage** (original, GitHub): fuente principal de entrenamiento/validación/test interno.
- **PlantDoc**: reservado como **test externo** con imágenes en condiciones naturales.

## Cómo se desarrolla

El código lo escribe Claude Code, lo revisa y pule Codex, y el alumno decide y mergea.
Ciclo, prompts y reglas: [`docs/WORKFLOW.md`](docs/WORKFLOW.md). Reglas para los agentes:
[`AGENTS.md`](AGENTS.md) y [`CLAUDE.md`](CLAUDE.md). Decisiones tomadas: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Estructura

```text
├── PROJECT_CONTEXT.md     contexto y requisitos del proyecto
├── AGENTS.md / CLAUDE.md  reglas para Codex y Claude Code
├── configs/               mapeos de etiquetas y parámetros
├── data/                  raw/ (no versionado), processed/ (regenerable), metadata.csv
├── docs/                  flujo de trabajo, tareas y decisiones
├── scripts/               scripts CLI (auditoría, preparación de datos, …)
├── src/                   módulos: preprocessing, segmentation, features, training, inference
├── reports/               informes y figuras ligeras para la memoria
├── experiments/           resultados de experimentos
└── tests/                 tests (pytest)
```
