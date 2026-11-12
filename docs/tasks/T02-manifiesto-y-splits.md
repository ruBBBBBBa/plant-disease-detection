# T02 — Mapeo de etiquetas, manifiesto y splits sin fugas

- **Estado:** En revisión
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
2. **Subconjunto (D007, revisada al cerrar T02).** Después de la partición, `in_subset` marca
   **todas** las sanas con hoja conocida y, en cada split, el mismo número de enfermas, a partes
   iguales entre las enfermedades disponibles y por hojas completas. Semilla 42. Es el comportamiento
   por defecto de `prepare_dataset.py` sin argumentos. (Primera versión: 500 sanas / 500 enfermas.)
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
- [ ] `split_summary.md` con distribuciones reales versionado. `metadata.csv` **no se versiona** (pesa > 2 MB);
      `split_summary.md` registra su SHA-1 y el commit de `data/raw/plantvillage` (y de `plantdoc`).
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto.
- [ ] La entrega explica el riesgo residual de fugas que quede.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Decisiones del alumno aplicadas:** D006, D007, D010 y D011 tal como están arriba; D008 y D009
  pasan a «Aceptada». Antes de implementar, dos choques con los datos reales que el alumno resolvió:
  - **Subconjunto:** hay 8 enfermedades con hoja conocida y 500/8 = 62,5. Además, las hojas tienen
    3–4 fotos. Decisión: objetivo por (enfermedad, split) redondeado, con el resto repartido por
    semilla (4 enfermedades con 63 y 4 con 62), y se añaden hojas completas **sin pasarse**.
  - **Umbral de casi-duplicados = 2** (configurable con `--near-dup-threshold`). Motivo en
    «Resultados».
  - Target_Spot: la correspondencia 1:1 se cumple, así que se usa (ver abajo). El alumno revisó
    `reports/data/target_spot_leaves.jpg` (3 hojas al azar, una fila por hoja): cada fila es la
    misma hoja → **Target_Spot queda incluida** (D013).
  - **Cierre de T02 (revisión del alumno):** subconjunto con todas las sanas disponibles y el mismo
    número de enfermas (D007 revisada); protocolo de evaluación documentado (D014); `metadata.csv`
    no versionado, con su SHA-1 y los commits de los datos en `split_summary.md`. La partición no cambia.
  - Clase del `leaf_id` distinta de la carpeta de la imagen: **0 casos**.
- **Comandos ejecutados:**

  ```bash
  python scripts/prepare_dataset.py --help
  python scripts/prepare_dataset.py            # seed 42, 70/15/15, umbral 2, subconjunto por defecto (~90 s)
  python scripts/prepare_dataset.py            # 2.ª ejecución tras copiar la 1.ª salida aparte
  cmp data/metadata.csv <copia> && cmp reports/data/split_summary.md <copia>   # idénticos
  sha1sum data/metadata.csv                    # 4f1e7c0bd2eb31df136e0e7e1f76333038ee985b
  python scripts/prepare_dataset.py --near-dup-threshold 4 --out <tmp>/metadata.csv --reports <tmp>   # comparación de umbral
  python scripts/audit_dataset.py --source plantvillage && python scripts/audit_dataset.py --source plantdoc --compare-with reports/audit/plantvillage/inventory.csv   # informes de T01 sin cambios tras mover md_table
  ruff check . && ruff format --check . && pytest   # 36 passed (tras la revisión de Codex y el cierre)
  git -C data/raw/plantvillage status --short | wc -l   # 0  (igual en plantdoc)
  ```

- **Resultados reales (distribución por split, casi-duplicados encontrados):** detalle en
  `reports/data/split_summary.md`.
  - Manifiesto: 18.906 filas (18.160 PlantVillage + 746 PlantDoc). Usadas 13.558, excluidas 5.348.
  - Excluidas: 5.345 de PlantVillage por `no_leaf_id` (healthy 592, late_blight 989,
    septoria 771, YLCV 2.619, mosaic_virus 373 (entera) y early_blight 1). Además, 3 copias exactas
    de PlantDoc (`exact_duplicate_of:<image_id>`). Ilegibles: 0.
  - **Target_Spot:** 1.404 ficheros, 1.404 números distintos y 1.404 números en el CSV, que coinciden
    1:1. Las 1.404 imágenes entran con su hoja del CSV.
  - Splits de PlantVillage (imágenes / hojas): **train 8.987 / 2.248, val 1.928 / 482,
    test 1.900 / 475**. Cada enfermedad queda ≈ 70/15/15 (p. ej. healthy 703/148/148,
    YLCV 1.918/412/408). Las sanas usadas son 999 de 12.815 (≈ 1:12).
  - `external_test` (PlantDoc): 743 imágenes. Entre ellas, `tomato_V8.jpg` con
    `disease_name = conflict:bacterial_spot|septoria_leaf_spot` y `binary_label = diseased`.
  - **Subconjunto (`in_subset`), versión final: 1.998 imágenes = 999 sanas + 999 enfermas.**
    Por split (sanas / enfermas): train 703 / 703, val 148 / 148, test 148 / 148. Enfermas por
    enfermedad: 116–131 (bacterial_spot 120, early_blight 128, late_blight 128, leaf_mold 116,
    septoria 128, spider_mites 128, target_spot 120, YLCV 131); en val y en test, 16 o 20 por
    enfermedad. Como las hojas tienen 3–4 fotos, lo que una enfermedad no llena sin pasarse pasa a
    la siguiente del mismo split, así que cada split cuadra con sus sanas.
    (Primera versión, sustituida: 959 = 492 + 467 con objetivo 500/500.)
  - **Casi-duplicados:** dHash de 64 bits con distancia de Hamming ≤ 2 → **0 parejas** entre las
    12.815 imágenes usadas de PlantVillage. También 0 duplicados exactos entre las usadas (las 14
    parejas de T01 caen en lotes excluidos). `check_leakage` también compara PlantDoc con
    PlantVillage: sin fugas.
    Con `--near-dup-threshold 4`: 38 parejas, 3.205 → 3.169 grupos, y **14 grupos mezclan
    enfermedades distintas**. Las parejas que revisé con distancia 3–4 eran hojas distintas sobre el
    mismo fondo gris (falsos positivos): con un umbral alto se unirían hojas que no tienen nada que ver.
  - Determinismo: dos ejecuciones seguidas con la versión final → `metadata.csv` y `split_summary.md`
    idénticos byte a byte (`cmp`). SHA-1 del CSV `4f1e7c0bd2eb31df136e0e7e1f76333038ee985b`, registrado
    en `split_summary.md` junto con los commits de `data/raw/plantvillage` (`7f7ecc7e…`) y `data/raw/plantdoc` (`5467f601…`).
  - `metadata.csv` pesa 6,0 MB (> 2 MB): **no se versiona** (añadido a `.gitignore`). Se regenera
    con el comando de arriba y se verifica con el SHA-1 de `split_summary.md`.

- **Riesgo residual de fugas:**
  - La agrupación depende de las hojas de cálculo originales de PlantVillage. Si la misma hoja física
    tiene dos números distintos (p. ej. en dos sesiones de captura), quedaría repartida y no se puede
    detectar con estos datos. Un mismo número de hoja usado para dos hojas distintas solo une de más
    (dirección segura).
  - Fotos de hojas **distintas** de la misma planta o sesión (mismo fondo, luz y cámara) pueden caer en
    splits distintos. No es una fuga por definición, pero hace el test interno optimista: por eso
    existe PlantDoc como `external_test`.
  - dHash con umbral 2 solo detecta casi-copias (recompresión, brillo). No detecta recortes, giros ni
    cambios de escala; PlantVillage original no trae aumentos, pero no está verificado imagen a imagen.
  - PlantDoc: solo se quitan duplicados exactos. Versiones redimensionadas de la misma imagen web
    seguirían contando dos veces dentro de `external_test` (no es fuga, pero infla el recuento).

- **Pendientes antes de empezar la segmentación:**
  - Protocolo de evaluación (D014): métricas principales en el `test` del subconjunto; análisis por
    enfermedad en el `test` completo (en el subconjunto solo hay 16–20 por enfermedad); PlantDoc como test externo.
  - **Ampliación pendiente (no implementada):** usar el número de hoja del nombre
    (`GH_HL Leaf 259.1`, `GHLB_PS Leaf 24 Day 16`) para recuperar parte de las 5.345 excluidas. Solo
    el lote `GH_HL Leaf …` recuperaría **unas 590 sanas** (590 según la auditoría de T01), que hoy
    limitan el subconjunto.
  - Casi-duplicados dentro de PlantDoc (redimensionados) si se quiere un `external_test` más limpio.
  - Hecho al cerrar T02: `PROJECT_CONTEXT.md` §5 ya dice que `metadata.csv` no se versiona.

## Revisión Codex

- **P0 corregido (protección de originales):** `--out` y `--reports` podían apuntar a
  `data/raw/` o a las raíces de los datasets. Ahora se rechazan antes de leer los datos,
  resolviendo también enlaces simbólicos. No se ha escrito en los originales.
- **P1 corregido (parámetros CLI):** un umbral negativo desactivaba silenciosamente la
  detección de casi-duplicados. Se exige un umbral entre 0 y 64, fracciones finitas no
  negativas que sumen 1 y tamaños de subconjunto no negativos. Pruebas de regresión incluidas.
- **Verificación ejecutada:** `ruff check .`, `ruff format --check .` y `pytest`
  (35 passed). Dos ejecuciones de `prepare_dataset.py --seed 42`, comparadas con `cmp`,
  generan CSV idénticos byte a byte: SHA-1 `ad75a61155e45a37fd8703ff844c09ffa93301db`.
  Ambas pasan `check_leakage`: 18.906 filas, 959 imágenes en el subconjunto,
  train/val/test/external_test = 8.987/1.928/1.900/743; excluidas 5.348.
  Se repitieron las auditorías de PlantVillage y PlantDoc con comparación entre fuentes:
  18.160 y 746 ficheros, respectivamente; informes versionados sin cambios.
  `git -C data/raw/plantvillage status --short` y su equivalente de PlantDoc sin cambios.
- **Para el alumno:** revisar visualmente Target_Spot y aceptar o ajustar el subconjunto
  de 959 imágenes. Se mantienen los riesgos residuales documentados: dHash no garantiza
  detectar recortes o giros. Sin cambio del enfoque ni tareas posteriores implementadas.
- **Referencia:** la tarea real de esta rama es `T02-manifiesto-y-splits.md`;
  `T02-descarga-y-auditoria.md` no existe.
