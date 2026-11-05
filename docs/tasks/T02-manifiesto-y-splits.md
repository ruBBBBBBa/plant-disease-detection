# T02 — Mapeo de etiquetas, manifiesto y splits sin fugas

- **Estado:** En curso
- **Rama:** `t02-prepare-dataset`
- **Depende de:** T01
- **Referencias:** `PROJECT_CONTEXT.md` §5 y §10 (pasos 4–7)

## Decisiones del alumno tras T01 (mandan sobre el resto del fichero)

1. **Partición (D006).** `train`/`val`/`test` solo con imágenes de PlantVillage con hoja conocida
   según `leaf-map.json` (lógica de D009). Las que no la tienen van al manifiesto con
   `split = excluded` y el motivo (`exclusion_reason`), y se cuentan en el resumen.
   - *Target_Spot*: si los números de `Com.G_TgS_FL nnnn` coinciden 1:1 con los `Com.G_FL nnnn`
     de su CSV, se usan para asignar la hoja y se genera una rejilla con las fotos de 3 hojas al
     azar para comprobarlo visualmente; si no, *Target_Spot* queda excluida.
   - No se usa la numeración del nombre (p. ej. `GH_HL Leaf 259.1`): solo se anota como posible ampliación.
   - Agrupada por `leaf_group_id`, estratificada por enfermedad, 70/15/15, semilla 42.
   - Duplicados exactos y casi-duplicados se unen en el mismo grupo.
   - Comprobación: contar y reportar imágenes cuyo `leaf_id` es de una clase distinta a su carpeta
     (`resolve_leaf_id` acepta un candidato único sin mirar la clase).
2. **Subconjunto (D007).** Después de la partición, `in_subset` marca 500 sanas y 500 enfermas;
   las enfermas a partes iguales entre las enfermedades disponibles, muestreando hojas completas
   dentro de cada split para mantener el 70/15/15. Semilla 42.
3. **PlantDoc (D010).** Todo el tomate es `external_test`. Se quitan los duplicados exactos
   (se conserva una copia). `tomato_V8.jpg`: `binary_label = diseased` y `disease_name` marcado como
   conflictivo. *Spider mites* y *yellow virus* se mantienen (para el binario son `diseased`).
4. **`segmented` (D011).** Solo referencia para evaluar la segmentación; no entra en el manifiesto.

## Objetivo

Generar de forma reproducible el manifiesto `data/metadata.csv` con etiquetas trazables y
particiones `train` / `val` / `test` / `external_test` sin fugas, y el subconjunto inicial de
tomate sano/enfermo para los primeros experimentos.

## Alcance

1. **Mapeo explícito** en `configs/label_mapping.csv` (una fila por carpeta y fuente):
   `source, folder, species, disease_name, binary_label`. Una carpeta de tomate que no esté en el
   mapeo es un **error**, no se ignora en silencio.

2. **`scripts/prepare_dataset.py`** (lógica en `src/` donde tenga sentido):
   - Parte del inventario de T01 (o vuelve a recorrer `data/raw/`); descarta ilegibles y registra cuántos.
   - Asigna `leaf_group_id` con el método decidido tras T01. Si para alguna fuente no hay
     información de hoja, grupo = imagen, y se documenta el riesgo residual.
   - Detecta **casi-duplicados** con un hash perceptual implementado con OpenCV/NumPy (aHash o
     dHash + distancia de Hamming con umbral por CLI) y une sus grupos (union-find) para que
     caigan en el mismo split. Reportar cuántos se encontraron con el umbral usado.
   - Particiona por grupos y estratificado por enfermedad (p. ej. `StratifiedGroupKFold` o
     `GroupShuffleSplit` de scikit-learn), con proporciones y semilla por CLI
     (orientativo 70/15/15, semilla 42). Primero la partición, luego el subconjunto.
   - Subconjunto inicial según la decisión del alumno (orientativo ~500/~500), respetando grupos
     y conservando `disease_name`.
   - PlantDoc → `split = external_test` siempre.
   - Salidas: `data/metadata.csv` (columnas de `PROJECT_CONTEXT.md` §5 más `modality`, `sha1`,
     `in_subset`) y `reports/data/split_summary.md` con la distribución original y la usada por
     split × `binary_label` × `disease_name`.

3. **Comprobación de fugas** (`scripts/check_leakage.py` o función en `src/` con su test):
   falla si un `leaf_group_id`, un SHA-1 o un par de casi-duplicados aparece en más de un split,
   o si alguna imagen de PlantDoc está fuera de `external_test`. Se ejecuta al final de
   `prepare_dataset.py`.

4. **Tests** con datos sintéticos: mapeo (incluida la carpeta sin mapear), splits deterministas
   con la misma semilla, grupos nunca repartidos entre splits, detección de casi-duplicados.

## Fuera de alcance

Segmentación, características, clasificadores, aumentos de datos (se aplicarán solo a `train` en fases posteriores).

## Criterios de aceptación

- [ ] Ejecutar dos veces `prepare_dataset.py` con la misma semilla produce el mismo `metadata.csv`.
- [ ] `check_leakage` pasa sobre el manifiesto real y su test demuestra que detecta fugas.
- [ ] `split_summary.md` con distribuciones reales versionado; `metadata.csv` versionado si pesa < 2 MB.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto.
- [ ] La entrega explica el riesgo residual de fugas que quede.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Decisiones del alumno aplicadas:**
- **Comandos ejecutados:**
- **Resultados reales (distribución por split, casi-duplicados encontrados):**
- **Riesgo residual de fugas:**
- **Pendientes antes de empezar la segmentación:**
