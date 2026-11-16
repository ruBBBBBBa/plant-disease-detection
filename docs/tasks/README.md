# Tablero de tareas

Estados: `Pendiente` → `En curso` (Claude Code) → `En revisión` (Codex) → `Lista para merge` (Rubba) → `Hecha`.
Ciclo y prompts: [`../WORKFLOW.md`](../WORKFLOW.md).

## Fase 1 — Datos de tomate

| ID | Tarea | Estado | Rama / PR |
|---|---|---|---|
| T00 | Setup del repositorio, reglas para agentes y CI | Hecha | `main` |
| [T01](T01-descarga-y-auditoria.md) | Descarga y auditoría de los datos de tomate | Hecha | `t01-audit-dataset` |
| [T02](T02-manifiesto-y-splits.md) | Mapeo de etiquetas, manifiesto y splits sin fugas | Hecha | `t02-prepare-dataset` (PR #2) |

## Fase 2 — Segmentación de la hoja

| ID | Tarea | Estado | Rama / PR |
|---|---|---|---|
| [T03](T03-segmentacion-hoja.md) | Segmentación clásica de la hoja y comparación frente a `segmented` | Hecha | `t03-leaf-segmentation` (PR #3) |

## Fase 3 — Características

| ID | Tarea | Estado | Rama / PR |
|---|---|---|---|
| [T04](T04-caracteristicas.md) | Extracción de características de color, textura y manchas | Hecha | `t04-features` (PR #4) |

## Fase 4 — Clasificación y evaluación

| ID | Tarea | Estado | Rama / PR |
|---|---|---|---|
| [T05](T05-clasificacion-evaluacion.md) | Clasificación clásica y evaluación (con control del sesgo de nitidez) | En revisión | `t05-classification` |

## Hoja de ruta (sin detallar a propósito)

Cada fase se convierte en ficheros de tarea cuando la anterior está cerrada, con sus resultados reales delante.

- **Fase 5 — Test externo y análisis de errores:** PlantDoc, falsos negativos/positivos, «análisis no concluyente».
- **Fase 6 — Explicación visual e inferencia:** CLI de inferencia con máscaras, regiones sospechosas y estadísticas.
- **Ampliaciones condicionadas:** patata/pimiento, enfermedades concretas, vídeo (CamShift/KLT), Streamlit.
