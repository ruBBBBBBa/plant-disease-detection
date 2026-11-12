# T03 — Segmentación clásica de la hoja (PlantVillage)

- **Estado:** En revisión
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

  ```bash
  sha1sum data/metadata.csv        # 4f1e7c0b… = reports/data/split_summary.md (no hizo falta regenerarlo)
  python scripts/tune_segmentation.py --sample-per-class 30 --seed 42      # solo train (270 imágenes), ~2 min
  python scripts/evaluate_segmentation.py --split val --out experiments/segmentation/val     # 296 imágenes, ~30 s
  # elección con val → configs/segmentation.json: default_method=kmeans, default_postprocess=true
  # commit 7eb054c con todo congelado; después, una sola vez:
  python scripts/evaluate_segmentation.py --split test --out experiments/segmentation/test
  ruff check . && ruff format --check . && pytest        # 54 passed
  git -C data/raw/plantvillage status --short | wc -l    # 0
  ```

  `tune_segmentation.py` es un script más de los que pedía la tarea: separa el ajuste (`train`)
  de la evaluación. `evaluate_segmentation.py --split test` se niega a ejecutarse si la
  configuración no tiene `default_method`, para no poder mirar `test` antes de elegir. Volver a
  ejecutar el ajuste y `val` con el código final da la misma configuración y métricas idénticas por
  imagen (comprobado). Tiempos medidos en Intel i7-1355U, Linux, un hilo de Python.

- **Ajuste con `train`** (`experiments/segmentation/tuning_train/summary.md`, 30 imágenes por
  enfermedad, semilla 42):
  - **Criterio:** media por imagen de (IoU + cobertura no verde) / 2. Con solo IoU, los métodos
    «de verde» ganan aunque pierdan lesiones, porque la mayoría de los píxeles de hoja son verdes.
  - Umbral de la referencia: **22** (valle del histograma de `max(B,G,R)` en `segmented`).
  - Otsu (M2): canal **ExG**, con score 0,68 (S 0,55, a* 0,63, L* 0,41).
  - M1: H 20–95, S ≥ 30, V ≥ 20.
  - M3: **k = 3 sobre (a*, b*)** con la regla del borde (`border_share` 0,3; 0,3–0,5 dan lo mismo).
  - M4: 3 iteraciones. M5: erosión 5, dilatación 15. Núcleo del postprocesado: 3.
  - **Dos correcciones de K-means durante el ajuste, ambas con `train`:**
    - La regla inicial «fondo = clúster de menor croma» fallaba cuando el fondo se partía en dos
      clústeres (iluminado y sombra), con IoU 0,54. Se sustituyó por «fondo = clústeres que ocupan
      el borde».
    - Con Lab completo la luminosidad domina la distancia (IoU 0,52); con solo (a*, b*), 0,93.
      Ambas variantes quedan en `tuning.csv`.

- **Resultados reales en `val`** (296 imágenes; detalle en `experiments/segmentation/val/summary.md`):

  | Variante | IoU media / p10 | IoU enfermas media / p10 | Cob. no verde media | Cob. no verde enfermas media / p10 | ms/img |
  |---|---|---|---|---|---:|
  | hsv_green | 0,904 / 0,810 | 0,917 / 0,835 | 0,645 | 0,670 / 0,458 | 0,1 |
  | hsv_green + post | 0,927 / 0,848 | 0,933 / 0,850 | 0,743 | 0,754 / 0,470 | 0,5 |
  | otsu (ExG) | 0,935 / 0,844 | 0,909 / 0,805 | 0,642 | 0,426 / 0,085 | 0,2 |
  | otsu + post | 0,940 / 0,871 | 0,919 / 0,813 | 0,687 | 0,511 / 0,131 | 0,5 |
  | kmeans | 0,938 / 0,858 | 0,945 / 0,893 | 0,824 | 0,704 / 0,349 | 5,5 |
  | **kmeans + post** | **0,943 / 0,872** | **0,954 / 0,914** | **0,854** | **0,762** / 0,440 | 5,9 |
  | grabcut | 0,911 / 0,837 | 0,900 / 0,790 | 0,741 | 0,621 / 0,214 | 76,4 |
  | grabcut + post | 0,910 / 0,837 | 0,898 / 0,788 | 0,741 | 0,622 / 0,208 | 76,8 |
  | watershed | 0,922 / 0,864 | 0,910 / 0,816 | 0,692 | 0,546 / 0,205 | 0,9 |
  | watershed + post | 0,924 / 0,875 | 0,912 / 0,835 | 0,703 | 0,570 / 0,214 | 1,3 |

  En *late_blight* (la enfermedad más difícil), kmeans + post saca IoU 0,912 y cobertura 0,892;
  el siguiente es hsv_green + post con 0,844 y 0,635. En sanas, el mejor es otsu (IoU 0,962).

- **Resultados reales en `test`** (ejecución única tras el commit `7eb054c`, todo congelado;
  `experiments/segmentation/test/summary.md`):

  | Variante | IoU media / p10 | IoU enfermas media / p10 | Cob. no verde media | Cob. no verde enfermas media / p10 |
  |---|---|---|---|---|
  | hsv_green | 0,900 / 0,820 | 0,894 / 0,772 | 0,636 | 0,621 / 0,381 |
  | hsv_green + post | 0,923 / 0,855 | 0,911 / 0,820 | 0,728 | 0,694 / 0,424 |
  | otsu | 0,922 / 0,858 | 0,883 / 0,736 | 0,620 | 0,370 / 0,059 |
  | otsu + post | 0,931 / 0,883 | 0,900 / 0,796 | 0,672 | 0,468 / 0,064 |
  | kmeans | 0,926 / 0,834 | 0,922 / 0,843 | 0,824 | 0,702 / 0,273 |
  | **kmeans + post** (elegido) | **0,932** / 0,834 | **0,932 / 0,843** | **0,859** | **0,768 / 0,429** |
  | grabcut | 0,902 / 0,826 | 0,883 / 0,772 | 0,728 | 0,598 / 0,158 |
  | grabcut + post | 0,903 / 0,823 | 0,885 / 0,771 | 0,734 | 0,610 / 0,155 |
  | watershed | 0,911 / 0,872 | 0,887 / 0,751 | 0,679 | 0,510 / 0,118 |
  | watershed + post | 0,914 / 0,873 | 0,892 / 0,773 | 0,689 | 0,531 / 0,117 |

  En `test` se confirma el orden de `val` en lo que importa: kmeans + post tiene la mayor IoU
  media en enfermas y la mayor cobertura no verde. Otsu + post tiene mejor p10 global
  (0,883 frente a 0,834), gracias a las sanas, pero en enfermas cubre menos de la mitad de lo no
  verde (0,468). *Late_blight* en test: kmeans + post IoU 0,834 y cobertura 0,892; hsv + post 0,788 y 0,577.

- **Cobertura de zonas no verdes — hallazgos:**
  - **Sí: el umbral de verde (M1) pierde lesiones.** En `val`, sin postprocesado, 35 imágenes
    quedan con cobertura < 0,5, y en ellas solo el 25 % de los píxeles no verdes está en el borde
    de la referencia: el resto son **lesiones interiores** fuera de la máscara. El relleno de huecos
    recupera las interiores (20 imágenes, ahora 81 % en el borde). Sube la cobertura en enfermas de
    0,670 a 0,754 en `val` y de 0,621 a 0,694 en `test`. Las lesiones que tocan el borde de la hoja
    no se recuperan.
  - Otsu sobre ExG (un índice de verdor) es el que más pierde: cobertura en enfermas de 0,43 (val)
    y 0,37 (test) sin postprocesado. En *late_blight* corta la hoja por la necrosis (ver
    `overlay_otsu.jpg`).
  - K-means con (a*, b*) no exige que la hoja sea verde y es el que más cubre.
  - **Límite de la métrica:** cuando quedan pocos píxeles no verdes, muchos están en el halo de la
    referencia (sombra o fondo pegados al borde), no en lesiones. En kmeans + post, de las 19
    imágenes de `val` con cobertura < 0,5, una mediana del 95 % de lo no cubierto está a ≤ 3 px del
    borde (87 % en `test`). Así se explica la cobertura baja de *yellow_leaf_curl_virus* en `test`
    (0,514): IoU 0,979, mediana de 401 píxeles no verdes y un 97 % en el borde en los casos bajos.
    La columna `non_green_rim_frac` y la tabla «Diagnóstico» de cada `summary.md` lo reproducen.

- **Fallos típicos y defectos de la referencia:**
  - Defectos de `segmented`: `experiments/segmentation/tuning_train/reference_examples.jpg` (de
    `train`; columnas color | segmented | referencia).
    - **Pierde necrosis oscura**: en *late_blight* muy oscura quedan solo fragmentos de la hoja.
    - **Recorta lesiones del borde.**
    - **Trunca alguna hoja sana.**
    - **Incluye sombra.**
    - **Sobre fondo azul incluye gran parte del fondo.**
    - Además, `segmented` está **aclarado** respecto a `color` (×1,14–1,25 por canal) pero
      alineado píxel a píxel (desplazamiento óptimo (0,0) en 80 de 80 imágenes). Por eso solo se usa
      su geometría, y los colores salen siempre de `color`.
  - Consecuencia: parte de los peores casos de los métodos son errores de la referencia. Por
    ejemplo, el peor de kmeans en `val` (*late_blight*, IoU 0,52) es una hoja que el método segmenta
    entera y la referencia recorta.
  - Fallos de los métodos (rejillas `overlay_*.jpg` de `val` y `test`, fila 2 = 8 peores):
    - **K-means:** incluye sombras marcadas del fondo (sanas con IoU 0,70–0,75). Ojo: una sombra
      dentro de la máscara es una zona oscura que la fase 3 podría confundir con necrosis.
    - **Otsu (ExG) y watershed:** pierden zonas marrones o necróticas.
    - **HSV:** deja huecos y canales conectados con el exterior en hojas con brillos.
    - **GrabCut:** hereda los cortes de Otsu y es 13 veces más lento (76 ms/img).

- **Método propuesto y por qué:** **K-means (k = 3, cromaticidad a*b*, fondo = clústeres del borde)
  + postprocesado (apertura/cierre 3×3, mayor componente, relleno de huecos)**, elegido con `val`
  (D015):
  - Mejor IoU media y p10 en enfermas.
  - Mayor cobertura no verde: no supone que la hoja sea verde, que es el punto clave de la tarea.
  - Gana con claridad en *late_blight*.
  - Rápido (≈ 6 ms/img).
  - Su fallo principal (sombras) está documentado. hsv_green + post es la alternativa más cercana
    en cobertura y la mejor en *yellow_leaf_curl_virus*.

- **Decisiones tomadas durante la tarea:**
  - Criterio de ajuste combinado IoU/cobertura.
  - Regla del borde y (a*, b*) en K-means.
  - Referencia = umbral 22 + relleno de huecos.
  - Definición de «verde»: H 30–90, S ≥ 40, fijada antes de evaluar.
  - Guarda que impide ejecutar `test` sin `default_method`.
  - D015 y D016 en `docs/DECISIONS.md` (propuestas).

- **Pendientes (para el alumno):**
  - Aceptar el método por defecto (D015) o cambiarlo.
  - Fase 3: calcular características solo dentro de la máscara y fijar con `val` un umbral de
    área para declarar «no concluyente».
  - **Métrica:** decidir si añadir una cobertura no verde «interior» (referencia erosionada unos
    3 px) que no castigue el halo de la referencia. No se ha cambiado aquí porque el problema se vio
    en `val`, y cambiar la métrica después habría sido ajustar con `val`.
  - Sombras en K-means: posible mejora (p. ej. combinarlo con otra máscara o probar k = 4) en una
    tarea futura, ajustando con `train`.
  - Otros métodos del Tema 5 (crecimiento de regiones, Mean Shift): no parecen necesarios con estos
    resultados.
