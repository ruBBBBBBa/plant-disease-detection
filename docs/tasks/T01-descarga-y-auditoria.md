# T01 — Descarga y auditoría de los datos de tomate

- **Estado:** Pendiente (siguiente)
- **Rama:** `t01-audit-dataset`
- **Depende de:** T00
- **Referencias:** `PROJECT_CONTEXT.md` §5 y §10 (pasos 1–3), `data/README.md`

## Objetivo

Saber exactamente qué datos tenemos antes de tomar ninguna decisión: estructura real, clases,
recuentos, resoluciones, ficheros ilegibles, duplicados exactos, ejemplos visuales y qué
información de **agrupación por hoja** ofrece PlantVillage. Esta tarea **no** crea splits.

## Punto de partida (ya comprobado)

`data/README.md` recoge la estructura y los recuentos obtenidos listando los repositorios sin
descargar imágenes: 18.160 ficheros de tomate en PlantVillage `color` (1.591 sanos) con su par
en `segmented`, y 746 ficheros de tomate en PlantDoc (63 sanos). Lo que falta por comprobar está
al final de la sección «Qué se ha comprobado y qué no».

## Alcance

1. **Datos presentes.** Comprobar si `data/raw/plantvillage` y `data/raw/plantdoc` existen. Si no
   están, pedir al alumno que ejecute los comandos de `data/README.md` (o ejecutarlos con su
   permiso). Verificar que los recuentos coinciden con `data/README.md` o explicar la diferencia.

2. **`scripts/audit_dataset.py`** (lógica reutilizable en `src/data_audit.py` si conviene):
   - CLI: `--source {plantvillage,plantdoc}`, `--root` (por defecto la ruta de `data/README.md`),
     `--species tomato` (filtro por nombre de carpeta, sin distinguir mayúsculas),
     `--out reports/audit/<source>/`, `--samples-per-class N` (por defecto 8), `--seed` (por defecto 42).
   - Suposición explícita y documentada: «carpeta = clase». Para PlantVillage se audita `color`
     y, aparte, se comprueba la correspondencia `color` ↔ `segmented` por nombre base
     (emparejadas / solo en color / solo en segmented).
   - Por fichero: ruta relativa a la raíz del repo, clase (carpeta), split original si lo hay
     (PlantDoc `train`/`test`), extensión, legible (`cv2.imread` no `None`), alto, ancho,
     canales, bytes y SHA-1 del contenido.
   - Salidas:
     - `inventory.csv`: una fila por fichero (regenerable; está en `.gitignore`).
     - `summary.md`: recuentos por clase, extensiones, resoluciones (mín/mediana/máx y las más
       frecuentes), ilegibles, duplicados exactos (dentro de cada clase, entre clases y, si se
       pasan ambas fuentes, entre PlantVillage y PlantDoc) y la tabla de emparejamiento color/segmented.
     - `samples_<clase>.jpg`: rejilla de N ejemplos aleatorios (semilla fija) por clase, reducidos
       para que cada rejilla pese < 300 KB.
   - Mensajes de error útiles si falta la carpeta o no hay imágenes.
   - Para comparar duplicados entre fuentes, el script puede aceptar un segundo inventario
     (`--compare-with reports/audit/<otra>/inventory.csv`) o una opción equivalente; documentar cuál.

3. **Investigar la agrupación por hoja de PlantVillage** (solo leer y documentar, sin implementar
   splits): `leaf_grouping/`, `leaf-map.json`, `data_distribution_for_SVM/*_mapping.txt`,
   `plant_village.py` y `README_HF.md`. Responder en la entrega, citando 2–3 líneas reales de cada
   fichero relevante:
   - ¿Qué formato tienen y cubren las imágenes de tomate?
   - ¿Cómo se obtiene el identificador de hoja de una imagen concreta? ¿Cuántas hojas distintas
     hay por clase y cuántas imágenes por hoja (mín/mediana/máx)?
   - ¿Qué partición ofrecen (`data_distribution_for_SVM`, Hugging Face) y respeta la agrupación?
   - ¿Se confirma o descarta la hipótesis de que la parte del nombre tras `___` identifica la hoja?
   Si un fichero no se entiende o es ambiguo, decirlo; no rellenar huecos con suposiciones.

4. **Tests** en `tests/test_audit_dataset.py` con un mini dataset sintético en `tmp_path`
   (2 clases, alguna imagen generada con NumPy/OpenCV, un fichero corrupto y un duplicado exacto),
   comprobando recuentos, detección de ilegibles y de duplicados.

5. **Ejecutar** el script sobre PlantVillage-tomate y PlantDoc-tomate reales y dejar en
   `reports/audit/` los `summary.md` y las rejillas.

## Fuera de alcance

Mapeo final de etiquetas, manifiesto, splits y subconjunto inicial (T02). Casi-duplicados por
hash perceptual (T02). Segmentación, características y clasificadores.

## Criterios de aceptación

- [ ] `python scripts/audit_dataset.py --help` funciona; error claro si falta un dataset.
- [ ] Ejecutado sobre los datos reales; `reports/audit/plantvillage/summary.md`,
      `reports/audit/plantdoc/summary.md` y las rejillas están versionados (ligeros).
- [ ] Recuentos coherentes con `data/README.md` (o diferencia explicada).
- [ ] Informe de la agrupación por hoja con ejemplos reales y respuestas a las preguntas del punto 3.
- [ ] `ruff check .`, `ruff format --check .` y `pytest` en verde.
- [ ] Nada modificado en `data/raw/` (`git -C data/raw/plantvillage status` limpio).

## Decisiones que quedan para el alumno al cerrar T01 (entrada de T02)

1. Partición: ¿usar la agrupación por hoja oficial / la partición de Hugging Face, o una propia
   agrupada con semilla fija?
2. Subconjunto inicial: tamaño (orientativo ~500 sanas / ~500 enfermas) y reparto de `diseased`
   entre las 9 enfermedades (¿proporcional o equilibrado por enfermedad?).
3. Qué hacer con clases de PlantDoc con muy pocas imágenes (p. ej. *spider mites*: 2) y con
   *Tomato leaf yellow virus* si su correspondencia no es clara.
4. Confirmar que `segmented` se usa solo como referencia de evaluación de la segmentación.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**
- **Resultados reales (resumen):**
- **Agrupación por hoja — hallazgos:**
- **Sorpresas / problemas en los datos:**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
