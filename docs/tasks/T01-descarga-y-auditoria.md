# T01 — Descarga y auditoría de los datos de tomate

- **Estado:** En revisión
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

  ```bash
  python scripts/audit_dataset.py --help
  python scripts/audit_dataset.py --source plantdoc --root nope   # → "Error: no existe nope. Descarga los datos siguiendo data/README.md (sección «Descarga»)."
  python scripts/audit_dataset.py --source plantvillage           # 18160 ficheros auditados (~5 s)
  python scripts/audit_dataset.py --source plantdoc --compare-with reports/audit/plantvillage/inventory.csv   # 746 ficheros
  ruff check . && ruff format --check . && pytest                 # 19 passed
  git -C data/raw/plantvillage status --short | wc -l              # 0
  git -C data/raw/plantdoc status --short | wc -l                  # 0
  ```

  Duplicados entre fuentes: opción `--compare-with <inventory.csv de la otra fuente>` (se ejecuta
  primero PlantVillage y después PlantDoc comparando con su inventario).

- **Resultados reales (resumen):** detalle en `reports/audit/plantvillage/summary.md` y
  `reports/audit/plantdoc/summary.md`; rejillas `samples_<clase>.jpg` (8 por clase, semilla 42,
  todas < 60 KB).
  - **PlantVillage `color` tomate:** 18.160 ficheros en 10 clases, **iguales** a `data/README.md`
    clase a clase (1.591 sanos / 16.569 enfermos). 0 ilegibles. Todos 256×256 y 3 canales.
    Extensiones: 18.159 `.jpg`/`.JPG` y 1 `.jpeg`.
  - Duplicados exactos: **14 parejas**, todas dentro de la misma clase (6 en `healthy`, 8 en
    `Late_blight`), 0 entre clases. En cada pareja los dos ficheros tienen distinto UUID pero el
    mismo nombre original (p. ej. `…___GH_HL Leaf 220.JPG` dos veces).
  - `color` ↔ `segmented`: 18.159 emparejados por nombre base. La única excepción
    (`Spider_mites`) es la misma foto: en `color` es `0db2da38-…___Com.G_SpM_FL 1378.JPG` y en
    `segmented` le falta el prefijo UUID (`Com.G_SpM_FL 1378_final_masked.jpg`). En la práctica, 1:1.
  - **PlantDoc tomate:** 746 ficheros en 9 clases (8 con `train`+`test`; *spider mites* solo 2 en
    `train`), iguales a `data/README.md`. 0 ilegibles. 745 `.jpg` y 1 `.png` (4 canales, BGRA).
    Resoluciones muy variables: de 115×69 a 6000×6000, mediana 640×562.
  - Duplicados exactos en PlantDoc: 3 parejas. 2 son la misma imagen en `train` y en `test` de la
    misma clase (`early-blight-septoria-ls-fig-3.jpg`, `tylcv-seminar-1-638.jpg`). La otra, `tomato_V8.jpg`,
    está en `train/Tomato Septoria leaf spot` **y** en `test/Tomato leaf bacterial spot`: la misma
    imagen tiene dos etiquetas distintas.
  - Entre PlantVillage y PlantDoc: **0** duplicados exactos.

- **Agrupación por hoja — hallazgos:**
  - **Formato.** `leaf_grouping/leaf_maps/*.csv` y `filtered_leafmaps/<clase>.csv` son hojas de
    cálculo exportadas (`Tomato Leaf Counts.xlsx` → CSV con `create_map.py`). Cada fila es una foto
    original y su número de hoja:
    ```text
    "File Name","Leaf #"            (filtered_leafmaps/Tomato___healthy.csv)
    "RS_HL 0003.JPG","1.0"
    "RS_HL 0004.JPG","1.0"
    ```
    `aggregate_map.py` las junta en `leaf-map.json` (idéntico al de la raíz, comprobado con `cmp`).
    Cada nombre en minúsculas y sin extensión se asocia a una lista de candidatos `"<clase>:::<nº hoja>"`:
    ```text
    {"rutg._hl 3733": ["Peach___healthy:::54.0"], "com.g_fl 8310": ["Tomato___Target_Spot:::243.0"], …
    ```
  - **Cómo se obtiene la hoja de una imagen.** `plant_village.py` (cargador de Hugging Face) hace:
    ```python
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    ```
    Después pasa a minúsculas, quita la extensión y lo busca en `leaf-map.json`. Si no lo
    encuentra, asigna `leaf_id = f"fallback_{image_identifier}"`, es decir, la imagen queda como su propia «hoja».
    Lo reproduce `src/data_audit.py` (`leaf_lookup_key`, `resolve_leaf_id`).
  - **Cobertura en tomate: 11.411 de 18.160 imágenes (63 %) tienen hoja conocida**, en 2.854 hojas.
    Allí donde hay mapa: **mín 3 / mediana 4 / máx 4 imágenes por hoja** en todas las clases.

    | Clase | Imágenes | Con hoja | Hojas |
    |---|---:|---:|---:|
    | Bacterial_spot | 2.127 | 2.127 | 532 |
    | Early_blight | 1.000 | 999 | 250 |
    | Late_blight | 1.909 | 920 | 230 |
    | Leaf_Mold | 952 | 952 | 238 |
    | Septoria_leaf_spot | 1.771 | 1.000 | 250 |
    | Spider_mites | 1.676 | 1.676 | 419 |
    | Target_Spot | 1.404 | **0** | 0 |
    | Yellow_Leaf_Curl_Virus | 5.357 | 2.738 | 685 |
    | mosaic_virus | 373 | **0** | 0 |
    | healthy | 1.591 | 999 | 250 |

    Las imágenes sin hoja se concentran en ciertos lotes de captura (prefijo del nombre):
    `healthy` `GH_HL Leaf …` (590); `Late_blight` `GHLB…` (987); `Septoria` `JR_Sept…` y `Keller…`
    (771); `YLCV` `UF…` (2.619); `mosaic_virus` `PSU_CG…` (373; no hay CSV de mosaic). En
    `Target_Spot` el CSV sí existe (1.404 filas), pero lista `Com.G_FL nnnn` y los ficheros se llaman
    `Com.G_TgS_FL nnnn`, así que la búsqueda falla. Que sean las mismas fotos es posible, pero
    **no está verificado**.
  - **Hipótesis «la parte tras `___` identifica la hoja»: descartada tal cual.** Hay tantos valores
    distintos como imágenes, salvo los 14 duplicados exactos. Esa parte identifica la **foto
    original**, no la hoja. La hoja (~4 fotos) viene del CSV. Algunos lotes sin mapa llevan un
    número de hoja en el propio nombre (`GH_HL Leaf 259.1`, `GHLB_PS Leaf 24 Day 16`). Es una pista
    para T02, **sin verificar**.
  - **Particiones.**
    - `data_distribution_for_SVM` solo incluye 6.435 imágenes de tomate (2.853 train, 3.582 test),
      no todo el dataset. Entre las que tienen hoja conocida (1.818 + 2.263), **0 hojas aparecen en
      ambos splits**. El resto (sin hoja) no se puede comprobar.
      ```text
      raw/color/Tomato___Bacterial_spot/dc3f2b4a-…___GCREC_Bact.Sp 3760.JPG	SVM/test/28/79b70cf4-….JPG
      ```
    - Hugging Face: `README_HF.md` dice *"pre-defined 80/20 train/test splits that preserve the leaf
      grouping logic"* y *"Multiple images often capture the same physical leaf"*. Las listas de esa
      partición (`splits/color_train.txt`, `color_test.txt`) se descargan de Hugging Face y **no están
      en el repo de GitHub**, así que no se han verificado. En las imágenes sin mapa (37 % del tomate),
      `plant_village.py` usa `fallback_<foto>`. Eso agrupa por foto, no por hoja: en esas imágenes la
      partición de Hugging Face no puede garantizar la agrupación por hoja.

- **Sorpresas / problemas en los datos:**
  - Los duplicados exactos de PlantVillage (14 parejas) deben ir siempre al mismo split.
  - El 37 % del tomate (incluidas Target_Spot y mosaic_virus enteras y 592 sanas) no tiene hoja
    conocida. Separarlas imagen a imagen puede repartir fotos de la misma hoja entre splits.
  - PlantDoc: el mismo fichero está en su `train` y en su `test`, y `tomato_V8.jpg` tiene dos
    etiquetas distintas. Si se usa todo PlantDoc como `external_test`, hay que deduplicarlo y
    decidir qué hacer con la imagen de doble etiqueta.
  - En PlantDoc hay plantas enteras, varias hojas por imagen, fondos blancos y marcas de agua
    (ver `reports/audit/plantdoc/samples_tomato_leaf.jpg`). Un PNG tiene 4 canales: habrá que pasarlo a BGR.

- **Decisiones tomadas durante la tarea:**
  - Duplicados entre fuentes con `--compare-with` (inventario de la otra fuente) (D008).
  - Hoja de cada imagen con la misma lógica que `plant_village.py`, reimplementada para no depender
    de la librería `datasets` (D009).
  - Cada fichero se lee una sola vez: con sus bytes se calcula el SHA-1 y se decodifica con
    `cv2.imdecode(..., IMREAD_UNCHANGED)`. Es el mismo decodificador que `cv2.imread` y así se
    conserva el número real de canales.
  - `inventory.csv` no se versiona (`.gitignore`; el de PlantVillage ocupa 3,8 MB). Se regenera con los comandos de arriba.

- **Pendientes:**
  - Decisiones para el alumno (arriba): partición y qué hacer con el 37 % sin hoja conocida
    (¿agrupar por lote o por número de hoja del nombre?), subconjunto inicial, clases pequeñas o
    dudosas de PlantDoc y deduplicación de PlantDoc, uso de `segmented` solo como referencia.
  - Comprobar si `Com.G_TgS_FL nnnn` (Target_Spot) se corresponde con `Com.G_FL nnnn` del CSV (T02).
  - Casi-duplicados por hash perceptual (T02, fuera de alcance).
