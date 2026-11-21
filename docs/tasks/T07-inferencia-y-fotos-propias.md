# T07 — Herramienta de inferencia, ROI manual y test con fotos propias

- **Estado:** En curso (fase A congelada; fase B a la espera de las fotos y los CSV)
- **Rama:** `t07-inference-own-photos`
- **Depende de:** T06 (D035 límite de validez, D036 variante por defecto B, D037 hipótesis del fondo blanco)
- **Referencias:** `PROJECT_CONTEXT.md` §2 (evidencias visuales, «no concluyente») y §4 (fotos
  propias); práctica de segmentación (GrabCut con rectángulo inicial); Tema 6.

## Objetivo

1. Construir la **herramienta de inferencia**: dada una foto, mostrar la máscara de la hoja, las
   regiones candidatas, la predicción con su puntuación (no es una probabilidad), las
   características que más han pesado y el aviso de «no concluyente» con su motivo.
2. Añadir un **modo con ROI manual**: el usuario marca un rectángulo alrededor de la hoja y la
   segmentación se hace con GrabCut inicializado con ese rectángulo.
3. Hacer un **test de un solo uso con fotos propias de hojas sanas** de una tomatera, en tres fondos.
   Mide los **falsos positivos** en condiciones reales y contrasta la hipótesis D037.

Las fotos propias solo contienen hojas sanas (más algunas «alteradas, no enfermas»), así que este
test **no mide el recall de enfermas**: eso ya está cubierto por PlantVillage y PlantDoc.

## Fotos propias: protocolo y formato (lo hace el alumno)

- **Hojas:** 20–30 hojas sanas distintas de la tomatera, sin arrancarlas. Cada hoja se fotografía en
  tres fondos: **gris** (cartulina gris o marrón claro, la condición válida tipo PlantVillage),
  **blanco** (cartulina blanca) y **natural** (sin cartulina). Luz de día indirecta, el móvil a
  20–30 cm, la hoja ocupando buena parte del encuadre.
- **Opcional, «alteradas»:** hojas viejas amarillentas, rotas o comidas por insectos, fotografiadas
  igual. **No son enfermas:** se informan aparte y nunca entran en las métricas de sanas ni de enfermas.
- **Ficheros:** `data/raw/own/<id_hoja>_<fondo>.jpg`, por ejemplo `h01_gris.jpg`, `h01_blanco.jpg`,
  `h01_natural.jpg`, `a01_gris.jpg` (`a` = alterada). Las imágenes **no se versionan**.
- **Etiquetas:** `data/annotations/own_photos.csv` (versionado), con las columnas
  `file, leaf_id, background, label, notes`; `label` ∈ {`healthy`, `altered`}. Las fotos de la misma
  hoja comparten `leaf_id`: son un grupo, como en PlantVillage.
- **Condiciones reales de la sesión de fotos** (se declaran tal cual; no se corrigen):
  - Muchas fotos muestran una **hoja compuesta** (varios folíolos y el raquis), no un folíolo suelto.
    `leaf_id` identifica la hoja compuesta.
  - Las hojas tienen **residuo blanco de un tratamiento** y gotas de agua. Siguen siendo `healthy`;
    el residuo es un factor de confusión declarado.
  - La cartulina «gris» es casi negra (`background = gris_oscuro`) y en algunas fotos no tapa todo
    el fondo. No todas las hojas tienen los tres fondos.
  - En `notes` se marcan, con palabras clave fijas separadas por `;`: `residuo`, `gotas`,
    `compuesta`, `fondo_incompleto`. Sirven para estratificar el informe; no se ajusta nada con ellas.
- **ROI, una por folíolo:** `data/annotations/own_rois.csv` (versionado), con las columnas
  `file, leaflet_id, x, y, w, h`. Una foto puede tener **varias filas**, una por folíolo visible
  entero. `leaflet_id` (`f1`, `f2`…) identifica el mismo folíolo en las fotos de la misma hoja con
  distinto fondo, cuando se pueda reconocer. Regla fija para dibujarlas: **rectángulo pegado al
  folíolo, sin margen: el margen lo añade el recorte** (D041); el folíolo entero, sin elegir
  folíolos según su aspecto. Las coordenadas son sobre la imagen original
  y se dibujan con `scripts/annotate_roi.py`, que solo muestra la foto y guarda los rectángulos; no
  segmenta ni predice.
- Los agentes **no abren ni procesan las fotos propias** hasta que exista el commit congelado (fase B).
  Dibujar las ROI es trabajo del alumno y está permitido antes de congelar.

## Fase A — herramienta, ROI y protocolo (sin fotos propias)

1. **Inferencia** (`src/inference.py` y `scripts/predict.py`):
   - CLI: `--image`, `--roi x,y,w,h` o `--select-roi` (interactivo, con `cv2.selectROI`),
     `--variant B` (por defecto, D036), `--out`.
   - Pipeline: normalización de T06, segmentación (K-means de T03 o GrabCut con ROI), comprobación
     de «no concluyente» (D031/D035), características `v1` y modelo de `models/exp2`.
   - Salida: una figura con
     - original, máscara de la hoja y regiones candidatas superpuestas;
     - la predicción y la **puntuación de decisión** del SVM (`decision_function`), etiquetada
       como «puntuación, no probabilidad»;
     - las **5 contribuciones más grandes** (coeficiente × valor estandarizado), que se pueden
       calcular porque B es lineal;
     - si es no concluyente, el motivo (qué característica de calidad se sale del rango) y **sin
       predicción**.
   - Además, un JSON con los mismos datos.
   - Texto fijo en la figura: las regiones candidatas son zonas de color anómalo, no «dónde ve la
     enfermedad el modelo» (regla 8 de `AGENTS.md`), y el sistema solo es fiable con una hoja sola
     sobre fondo uniforme (D035).

2. **Modo ROI** (preregistrado; no se ajusta con fotos propias):
   - recortar la ROI con el margen de `configs/own.json` (10 % de w y h por lado, elegido solo con
     PlantVillage, D041), normalizar el recorte con la regla de T06 (lado mayor a 256 px) y aplicar GrabCut inicializado con el rectángulo interior, con las iteraciones de
     `configs/segmentation.json`;
   - después, el mismo postprocesado de T03.

   **Comprobación de sanidad, solo con PlantVillage `val`** (`scripts/check_roi_mode.py`): simular
   la ROI como la caja **ajustada** que envuelve la máscara de referencia, ampliar la imagen solo en
   la simulación con `BORDER_REPLICATE` para que el recorte no choque con el borde, y medir el IoU
   frente a la referencia, comparado con K-means. Con la misma simulación se elige el margen de
   recorte (decisión del alumno, D041): el menor de {10, 20, 30, 40, 50 %} cuya mediana de
   `leaf_area_frac` en el recorte quede en [p25, p75] de train+val de PlantVillage. Limitaciones: el
   fondo replicado es una aproximación (rayas si hay sombra u hoja en el borde) y este IoU no es el
   rendimiento real del modo ROI.

3. **Script de evaluación** `scripts/eval_own.py`, con la misma guarda que `eval_external.py`:
   commit exacto, árbol limpio y se niega si existe cualquier `execution_*.json` en
   `experiments/own/v1/`. Evalúa cada foto con B en dos modos (automático y ROI), y también con A
   como referencia. Guarda las predicciones por foto (comprimidas, < 2 MB). Las tablas y rejillas se
   regeneran desde ellas.

4. **Métricas e hipótesis, fijadas antes de congelar.**
   - **Unidades:** en modo automático se evalúa cada **foto**; en modo ROI, cada **folíolo** (fila de
     `own_rois.csv`). Las tasas se agregan primero por `leaf_id` y fondo, y el bootstrap remuestrea
     **hojas compuestas** (`leaf_id`), nunca folíolos ni fotos (10 000 réplicas, semilla 42).
   - Para las sanas: **tasa de fallo** = (FP + no concluyentes) / todas las unidades (D040), y como
     descriptivas la **tasa de FP sobre concluyentes** y la **tasa de no concluyentes**, por fondo y
     por modo; lectura «solo concluyentes» siempre con su cobertura. Cada tabla indica cuántas hojas y
     cuántos folíolos entran.
   - **Estratificación descriptiva** por las palabras clave de `notes` (residuo, gotas, compuesta,
     fondo incompleto). Sin veredicto.

   Las hipótesis usan la variante B. Las comparaciones entre fondos se emparejan por `leaf_id` y solo
   usan las hojas que tienen fotos en los dos fondos comparados. Regla: «apoyada» si el IC 95 % de la
   diferencia emparejada excluye 0 con el signo esperado; «no apoyada (efecto contrario)» si lo
   excluye con el signo opuesto; en otro caso, «no concluyente».
   - **H4:** en modo ROI, la tasa de fallo con fondo **gris oscuro** es menor que con fondo
     **natural**.
   - **H5 (contrasta D037):** en modo ROI, la tasa de fallo con fondo **blanco** es mayor que con
     fondo **gris oscuro**.

   H4 y H5 usan la misma tasa de fallo que H6 para que un fondo no salga ganando por abstenerse
   (D040).
   - **H6:** juntando todos los fondos, el modo ROI reduce la tasa de «no concluyente + FP» frente al
     modo automático (por hoja: tasa del modo ROI sobre sus folíolos frente a tasa del modo
     automático sobre sus fotos).

   H4 y H5 se miden en modo ROI porque, con hojas compuestas, el modo automático marcará casi todo
   como no concluyente por la forma de la máscara, y no por el fondo. El modo automático se informa
   igualmente, de forma descriptiva.

   Con 20–30 hojas los IC serán anchos: un «no concluyente» es un resultado legítimo y se informa así.
   Las hojas «alteradas» se describen aparte: qué predice el sistema y por qué, sin métricas de acierto.

5. **Tests** (`tests/test_inference.py`), con imágenes sintéticas:
   - la figura y el JSON contienen todos los campos;
   - una imagen no concluyente no lleva predicción;
   - las contribuciones lineales suman la puntuación de decisión (menos el término independiente);
   - el modo ROI recorta, reescala y transforma las coordenadas correctamente;
   - `annotate_roi.py` guarda varias ROI por foto con su `leaflet_id`, en coordenadas de la imagen original;
   - el bootstrap remuestrea hojas compuestas (no folíolos ni fotos);
   - la guarda de `eval_own.py` impide una segunda ejecución.

**Commit congelado:** todo lo anterior, con `ruff` y `pytest` en verde, sin haber procesado las
fotos propias. Después, el implementador **se detiene** y espera a que el alumno confirme que las
fotos, `own_photos.csv` y `own_rois.csv` están listos y versionados.

## Fase B — ejecución única (con las fotos del alumno)

1. Comprobar que cada fichero del CSV existe en `data/raw/own/`, que no hay fotos sin anotar, que
   cada foto sana tiene al menos una ROI y que los `leaflet_id` son coherentes. Informar de cuántas
   hojas tienen cada combinación de fondos (no se exige tener los tres). Si algo falla, parar y
   avisar; no corregir a mano.
2. Ejecutar `eval_own.py` **una vez**.
3. Informe `experiments/own/v1/summary.md`:
   - tablas por fondo × modo;
   - veredicto de H4–H6;
   - rejillas con las fotos y sus máscaras (miniaturas de fotos propias: el alumno decide si se versionan);
   - descripción de las «alteradas»;
   - 3–4 figuras de ejemplo de `predict.py` para la memoria.

## Fuera de alcance

Reajustar umbrales, segmentación o modelos con las fotos propias. Interfaz web (Streamlit, como
ampliación posterior si queda tiempo). Vídeo.

## Criterios de aceptación

- [ ] `predict.py` funciona en modo automático y en modo ROI, y produce la figura y el JSON descritos.
- [ ] Comprobación de sanidad del modo ROI en PlantVillage `val`.
- [ ] Protocolo, hipótesis y guarda en un commit congelado **anterior** a procesar las fotos propias.
- [ ] Ejecución única registrada; predicciones guardadas; informe regenerable.
- [ ] Veredicto de H4–H6 con IC agrupados por hoja y sin forzar conclusiones.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; imágenes propias fuera de git salvo decisión del alumno.

## Decisiones que quedan para el alumno al cerrar T07

1. Modo por defecto de la herramienta (automático o ROI).
2. Si se versionan miniaturas de las fotos propias.
3. Qué figuras de ejemplo van a la memoria.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Fase A — comandos y comprobación de sanidad del ROI:**
  - `python scripts/check_roi_mode.py` (296 imágenes de `val` del subconjunto; ~3,5 min). Salida en
    `experiments/segmentation/roi_sanity_val/` (`summary.json`, `per_image.csv.gz`); escribe
    `roi` en `configs/own.json`.
  - Primer intento con la ROI simulada = caja + 10 % y sin ampliar la imagen: el recorte chocaba
    con el borde en el 100 % de las imágenes (la hoja de PlantVillage ocupa ~80 × 91 % del
    encuadre), la mediana de `leaf_area_frac` salía 0,481 con cualquier margen (> p75 = 0,474) y no
    se podía elegir. El alumno eligió la simulación con caja ajustada y `BORDER_REPLICATE` (D041).
  - Resultado final, [p25, p75] de train+val = [0,376, 0,474]:

    | margen | mediana `leaf_area_frac` | no concluyentes | IoU mediano |
    |---|---|---|---|
    | **10 %** | **0,429** | 2,4 % | 0,898 |
    | 20 % | 0,317 | 1,4 % | 0,897 |
    | 30 % | 0,245 | 4,1 % | 0,894 |
    | 40 % | 0,194 | 24,3 % | 0,888 |
    | 50 % | 0,158 | 95,9 % | 0,892 |

    Margen elegido: **10 %**. IoU frente a la referencia (mediana / media): K-means 0,971 / 0,954
    en enfermas y 0,959 / 0,933 en sanas; modo ROI al 10 % 0,912 / 0,886 en enfermas y
    0,889 / 0,885 en sanas. Con hojas de PlantVillage sobre fondo uniforme, K-means sigue siendo
    mejor; el modo ROI funciona pero no es una mejora aquí. Limitaciones: el fondo replicado es una
    aproximación y este IoU no es el rendimiento real del modo ROI con fotos propias.
  - `python scripts/predict.py --image <imagen de val> --out <scratchpad>` y con `--roi`: figura y
    JSON correctos con el modelo real (`models/exp2/color_manchas__svm_lineal.joblib`).
  - Prueba de principio a fin de `eval_own.py` + `report_own.py` con imágenes de PlantVillage
    haciendo de «fotos propias» en un directorio temporal (guarda sustituida solo en esa prueba;
    nada se guardó en el repo).
  - `ruff check . && ruff format --check . && pytest`: en verde (110 tests).
  - No se ha abierto, listado ni procesado nada de `data/raw/own/`.
- **Commit congelado:** el commit `T07: fase A congelada …` de esta rama (hash en el PR y en el
  registro de `eval_own.py`).
- **Fase B — fotos recibidas (hojas, fondos, alteradas):**
- **Resultados por fondo × modo:**
- **Veredicto H4–H6:**
- **Hojas alteradas:**
- **Decisiones tomadas durante la tarea:** D040 (tasa de fallo para H4–H6), D041 (simulación de la
  ROI, margen del 10 % y regla de dibujo), D042 (modo ROI: GrabCut y «no concluyente» sobre el
  recorte), D043 (agregación por hoja y bootstrap).
- **Pendientes:** fase B (cuando el alumno confirme fotos, `own_photos.csv` y `own_rois.csv`).
