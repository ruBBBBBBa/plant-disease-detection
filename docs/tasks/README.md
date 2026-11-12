# Tablero de tareas

Estados: `Pendiente` → `En curso` (Claude Code) → `En revisión` (Codex) → `Lista para merge` (Rubba) → `Hecha`.
Ciclo y prompts: [`../WORKFLOW.md`](../WORKFLOW.md).

## Fase 1 — Datos de tomate

| ID | Tarea | Estado | Rama / PR |
|---|---|---|---|
| T00 | Setup del repositorio, reglas para agentes y CI | Hecha | `main` |
| [T01](T01-descarga-y-auditoria.md) | Descarga y auditoría de los datos de tomate | Hecha | `t01-audit-dataset` |
| [T02](T02-manifiesto-y-splits.md) | Mapeo de etiquetas, manifiesto y splits sin fugas | En revisión | `t02-prepare-dataset` |

**Al cerrar la fase 1 se para**: se revisa lo aprendido de los datos y se detallan las tareas
siguientes con resultados reales delante.

## Hoja de ruta (sin detallar a propósito)

Se convierten en ficheros de tarea cuando la fase anterior esté cerrada.

- **Fase 2 — Segmentación de la hoja:** comparar umbral HSV, Otsu, GrabCut con ROI y watershed;
  medir IoU contra `segmented` de PlantVillage y documentar qué se pierde (zonas marrones/amarillas).
- **Fase 3 — Características:** color (HSV/Lab, histogramas, proporción de tonos anómalos),
  textura (GLCM, LBP), regiones/manchas (componentes conexas, área, irregularidad).
- **Fase 4 — Clasificación y evaluación:** baseline por reglas, kNN, SVM, árbol/Random Forest;
  color vs textura vs combinadas; matriz de confusión, precision/recall/F1, tiempo por imagen.
- **Fase 5 — Test externo y análisis de errores:** PlantDoc, falsos negativos/positivos, «análisis no concluyente».
- **Fase 6 — Explicación visual e inferencia:** CLI de inferencia con máscaras, regiones sospechosas y estadísticas.
- **Ampliaciones condicionadas:** patata/pimiento, enfermedades concretas, vídeo (CamShift/KLT), Streamlit.
