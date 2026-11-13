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
| [T03](T03-segmentacion-hoja.md) | Segmentación clásica de la hoja y comparación frente a `segmented` | En revisión | `t03-leaf-segmentation` |

## Hoja de ruta (sin detallar a propósito)

Cada fase se convierte en ficheros de tarea cuando la anterior está cerrada, con sus resultados reales delante.

- **Fase 3 — Características:** color (HSV/Lab, histogramas, proporción de tonos anómalos),
  textura (GLCM, LBP), regiones/manchas (componentes conexas, área, irregularidad).
- **Fase 4 — Clasificación y evaluación:** baseline por reglas, kNN, SVM, árbol/Random Forest;
  color vs textura vs combinadas; matriz de confusión, precision/recall/F1, tiempo por imagen.
- **Fase 5 — Test externo y análisis de errores:** PlantDoc, falsos negativos/positivos, «análisis no concluyente».
- **Fase 6 — Explicación visual e inferencia:** CLI de inferencia con máscaras, regiones sospechosas y estadísticas.
- **Ampliaciones condicionadas:** patata/pimiento, enfermedades concretas, vídeo (CamShift/KLT), Streamlit.
