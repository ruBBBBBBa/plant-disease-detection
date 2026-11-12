# T03 — Segmentación clásica de la hoja (PlantVillage)

- **Estado:** Pendiente (siguiente)
- **Rama:** `t03-leaf-segmentation`
- **Depende de:** T02
- **Referencias:** `PROJECT_CONTEXT.md` §6 (pipeline y consideraciones sobre segmentación);
  Tema 5 de la asignatura (umbralización global y Otsu, crecimiento de regiones, watershed,
  clustering con K-means); práctica de segmentación del manual (contornos y momentos, GrabCut con
  rectángulo inicial, Watershed con Otsu + dilatación/erosión); decisiones D011 y D014.

## Objetivo

Separar la **hoja** del **fondo** en las imágenes de PlantVillage con técnicas clásicas, comparar
varios métodos de forma cuantitativa frente a la referencia `segmented` de PlantVillage y proponer
uno, justificado, para las fases siguientes.

Punto clave: el método **no puede perder las zonas enfermas** (marrones, amarillas, necróticas),
porque son justo la evidencia que necesitará el clasificador. Una segmentación por umbral de verde
probablemente las pierde: hay que **medirlo**, no suponerlo.

Esta tarea segmenta la hoja completa. **No** detecta lesiones dentro de la hoja (eso es fase 3).

## Datos y protocolo (sin fugas)

- Imágenes: `data/metadata.csv` con `source == "plantvillage"` e `in_subset == True`
  (1.998 imágenes: 703 + 703 en `train`, 148 + 148 en `val` y 148 + 148 en `test`).
  Si `data/metadata.csv` no existe, se regenera con `python scripts/prepare_dataset.py` y se
  comprueba que su SHA-1 coincide con el de `reports/data/split_summary.md`.
- **Desarrollo y ajuste de parámetros:** solo con `train` (si conviene, una muestra fija por clase
  con semilla declarada).
- **Elección del método y de sus parámetros:** con `val`.
- **`test`:** se ejecuta **una sola vez**, al final, con todo congelado (métodos, parámetros y
  elección). Solo para informar; no se cambia nada después de verlo.
- **PlantDoc queda fuera** de esta tarea (fase 5): no se ejecuta, no se mira y no se ajusta nada con él.
- **Referencia:** máscara obtenida de `raw/segmented/<clase>/<nombre>_final_masked.jpg`, que es la
  hoja sobre fondo negro. El emparejamiento con `color` es como en T01 (`pair_color_segmented`),
  incluido el caso de `Spider_mites` cuyo fichero segmentado no lleva prefijo UUID.
  Ojo: es la salida de un procesado automático del dataset, **no una anotación manual garantizada**.
  Se llama «referencia», no *ground truth*. Hay que revisar visualmente una muestra y documentar
  sus defectos (bordes por la compresión JPEG, sombras incluidas o recortes, etc.).

## Alcance

1. **`src/segmentation.py`**. Cada método es una función
   `segment_<metodo>(image_bgr: np.ndarray, **params) -> np.ndarray` que devuelve una máscara
   `uint8` con valores 0/255 y el mismo alto y ancho que la imagen. Como mínimo, 4 métodos:
   - **M1 — Umbral de verde en HSV** (`cv2.inRange`). Es el baseline. Recordar que en OpenCV
     H va de 0 a 179.
   - **M2 — Otsu sobre un canal** que separe hoja y fondo sea cual sea el color de la hoja
     (candidatos: S de HSV, a* o L* de Lab, exceso de verde ExG = 2G − R − B…). El canal se elige
     con datos de `train` y se justifica, por ejemplo con histogramas de hoja frente a fondo
     calculados con la referencia.
   - **M3 — K-means** (k = 2 o 3) en un espacio de color (Lab o HSV) con una regla explícita y
     documentada para decidir qué cluster es la hoja.
   - **M4 — GrabCut** inicializado con la máscara de M2 (o con un rectángulo, como en la práctica).
   - **M5 (opcional) — Watershed con marcadores**: fondo seguro por dilatación y hoja segura por
     erosión, como en la práctica.
   - **Postprocesado común**, aplicable a cualquier método y medido con y sin él: apertura/cierre
     morfológico, quedarse con la mayor componente conexa y **rellenar huecos** (las lesiones
     interiores pueden quedar como agujeros en la máscara).
   - `reference_mask(segmented_bgr, threshold)`: máscara de referencia a partir de la imagen
     segmentada (fondo negro → 0).
   - Métricas: **IoU**, **Dice** y **cobertura de zonas no verdes**. Esta última se calcula sobre
     los píxeles de la referencia cuyo tono no es verde (rango de H documentado) y mide qué fracción
     de ellos incluye la máscara del método. Mide directamente si el método pierde la parte enferma.

2. **`scripts/evaluate_segmentation.py`**:
   - CLI: `--split {train,val,test}`, `--methods`, `--postprocess`, `--sample-per-class N`
     (para `train`), `--seed`, `--config` y `--out experiments/segmentation/<nombre>/`.
   - Salidas:
     - `per_image.csv`: métricas por imagen y método.
     - `summary.md`: media, mediana y percentil 10 de IoU, Dice y cobertura no verde por método,
       separados por `binary_label` y por `disease_name`. También el tiempo medio por imagen en ms,
       indicando la máquina.
     - `overlay_<metodo>.jpg`: rejillas con el contorno de la máscara dibujado sobre la imagen,
       con casos típicos (cerca de la mediana) y los 8 peores.
   - Parámetros finales en `configs/segmentation.json`, para que las fases siguientes los lean de ahí.

3. **Elección.** Con `val`, proponer el método por defecto y justificarlo. No basta con el IoU:
   cuenta también la cobertura de zonas no verdes, el comportamiento en hojas enfermas y el tipo de
   fallos. Después, ejecutar `test` una vez con todo congelado e informar de todos los métodos.
   La elección hecha con `val` no cambia.

4. **Tests** (`tests/test_segmentation.py`) con imágenes sintéticas, por ejemplo una elipse verde con
   una mancha marrón interior sobre fondo gris:
   - cada método devuelve una máscara 0/255 del tamaño correcto;
   - el relleno de huecos recupera la mancha interior;
   - IoU y Dice dan 1 con máscaras idénticas y 0 con disjuntas;
   - `reference_mask` funciona sobre fondo negro;
   - la cobertura no verde se calcula bien en un caso conocido.

## Fuera de alcance

Detección de lesiones dentro de la hoja, características y clasificadores (fases 3 y 4). PlantDoc,
fotos propias, vídeo e interfaz. Probar técnicas no vistas en el curso sin justificarlo antes en el PR.

## Criterios de aceptación

- [ ] ≥ 4 métodos más el postprocesado. Cada función tiene un docstring que dice qué técnica del
      Tema 5 o de la práctica aplica y por qué encaja.
- [ ] Resultados de `val` para todos los métodos y resultados de `test`, ejecutado una sola vez con
      todo congelado. En ambos casos, separados por sano/enfermo y por enfermedad.
- [ ] Cobertura de zonas no verdes medida y comentada: ¿el umbral de verde pierde lesiones?, ¿cuánto?
- [ ] Rejillas con los peores casos y una explicación de los tipos de fallo.
- [ ] Defectos observados de la referencia `segmented` documentados con ejemplos.
- [ ] Parámetros finales en `configs/segmentation.json` y decisión propuesta en `docs/DECISIONS.md`.
- [ ] `ruff check .`, `ruff format --check .` y `pytest` en verde; `data/raw/` intacto; ficheros de
      `experiments/` y `reports/` ligeros (< 2 MB cada uno).

## Decisiones que quedan para el alumno al cerrar T03

1. Método de segmentación por defecto para el pipeline.
2. Si las características de la fase 3 se calculan solo dentro de la máscara de la hoja, y qué hacer
   cuando la máscara sale mal (p. ej. si el área cae por debajo de un umbral fijado con `val`, el
   análisis se declara «no concluyente»).
3. Si merece la pena probar otro método del Tema 5 (crecimiento de regiones, Mean Shift) o es suficiente.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**
- **Resultados reales en `val` (tabla por método):**
- **Resultados reales en `test` (ejecución única, todo congelado):**
- **Cobertura de zonas no verdes — hallazgos:**
- **Fallos típicos y defectos de la referencia:**
- **Método propuesto y por qué:**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
