# Registro de decisiones

Cada decisión con fecha, motivo y alternativas descartadas. Sirve de base para la sección de
metodología de la memoria. Las propuestas pendientes se marcan como tales.

| ID | Fecha | Decisión | Motivo | Alternativas / notas | Estado |
|---|---|---|---|---|---|
| D001 | 2026-11-04 | MVP = clasificación binaria `healthy`/`diseased` en hojas de tomate | Alcance realista para un proyecto individual; permite evaluar bien antes de ampliar | Multiclase por enfermedad y otras especies, como ampliación condicionada | Aceptada |
| D002 | 2026-11-04 | Solo visión clásica + características manuales + clasificadores tradicionales | Restricción de la asignatura | Redes neuronales y modelos preentrenados: prohibidos (guarda en CI) | Aceptada |
| D003 | 2026-11-04 | PlantVillage original (GitHub) como fuente principal; PlantDoc solo como `external_test` | Original sin aumentos permite controlar duplicados y agrupación por hoja; PlantDoc aporta condiciones naturales | Variantes de Kaggle (contienen aumentos, posible solapamiento) | Aceptada |
| D004 | 2026-11-04 | Datos fuera de git, descarga parcial (solo tomate) con *sparse checkout* | Evita bajar ~54k imágenes por modalidad; reproducible con comandos documentados | Clonar el repo entero; Hugging Face | Aceptada |
| D005 | 2026-11-04 | Flujo: Claude Code implementa, Codex revisa y pule, el alumno mergea | Separar autoría y revisión; el alumno entiende todo antes de que entre en `main` | — | Aceptada |
| D006 | — | Método de partición por hoja | Depende de lo que revele T01 sobre `leaf_grouping` | T01: el mapa oficial cubre solo el 63 % del tomate (~4 fotos/hoja); Target_Spot y mosaic_virus sin hoja conocida. Ver «Entrega» de T01 | Pendiente (tras T01) |
| D007 | — | Tamaño y reparto del subconjunto inicial | Desequilibrio 1.591 sanas vs 16.569 enfermas | ~500/~500 orientativo | Pendiente (tras T01) |
| D008 | 2026-11-04 | Duplicados exactos entre fuentes con `--compare-with <inventory.csv>` | Cada fuente se audita por separado y se compara por SHA-1 sin volver a leer imágenes | Auditar ambas fuentes a la vez | Propuesta (T01) |
| D009 | 2026-11-04 | Identificador de hoja de PlantVillage = lógica de `plant_village.py` sobre `leaf-map.json`, reimplementada en `src/data_audit.py` | Es la lógica oficial de Hugging Face; reimplementarla evita depender de `datasets` | Usar la parte del nombre tras `___` (descartado: identifica la foto, no la hoja) | Propuesta (T01) |
