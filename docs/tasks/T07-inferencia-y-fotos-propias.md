# T07 — Herramienta de inferencia, ROI manual y test con fotos propias

- **Estado:** Pendiente (siguiente)
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
  distinto fondo, cuando se pueda reconocer. Regla fija para dibujarlas: el folíolo entero con un
  margen pequeño, sin elegir folíolos según su aspecto. Las coordenadas son sobre la imagen original
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
   - recortar la ROI con un margen del 10 %, normalizar el recorte con la regla de T06 (lado mayor
     a 256 px) y aplicar GrabCut inicializado con el rectángulo interior, con las iteraciones de
     `configs/segmentation.json`;
   - después, el mismo postprocesado de T03.

   **Comprobación de sanidad, solo con PlantVillage `val`:** simular la ROI como la caja que
   envuelve la máscara de referencia más un margen del 10 % y medir el IoU frente a la referencia.
   Comparar con K-means. Es solo una comprobación de que el modo funciona; no se elige nada con ella.

3. **Script de evaluación** `scripts/eval_own.py`, con la misma guarda que `eval_external.py`:
   commit exacto, árbol limpio y se niega si existe cualquier `execution_*.json` en
   `experiments/own/v1/`. Evalúa cada foto con B en dos modos (automático y ROI), y también con A
   como referencia. Guarda las predicciones por foto (comprimidas, < 2 MB). Las tablas y rejillas se
   regeneran desde ellas.

4. **Métricas e hipótesis, fijadas antes de congelar.**
   - **Unidades:** en modo automático se evalúa cada **foto**; en modo ROI, cada **folíolo** (fila de
     `own_rois.csv`). Las tasas se agregan primero por `leaf_id` y fondo, y el bootstrap remuestrea
     **hojas compuestas** (`leaf_id`), nunca folíolos ni fotos (10 000 réplicas, semilla 42).
   - Para las sanas: **tasa de falsos positivos (FP)** y **tasa de no concluyentes**, por fondo y por
     modo; lectura «solo concluyentes» siempre con su cobertura. Cada tabla indica cuántas hojas y
     cuántos folíolos entran.
   - **Estratificación descriptiva** por las palabras clave de `notes` (residuo, gotas, compuesta,
     fondo incompleto). Sin veredicto.

   Las hipótesis usan la variante B. Las comparaciones entre fondos se emparejan por `leaf_id` y solo
   usan las hojas que tienen fotos en los dos fondos comparados. Regla: «apoyada» si el IC 95 % de la
   diferencia emparejada excluye 0 con el signo esperado; «no apoyada (efecto contrario)» si lo
   excluye con el signo opuesto; en otro caso, «no concluyente».
   - **H4:** en modo ROI, la tasa de FP con fondo **gris oscuro** es menor que con fondo **natural**.
   - **H5 (contrasta D037):** en modo ROI, la tasa de FP con fondo **blanco** es mayor que con fondo
     **gris oscuro**.
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
- **Commit congelado:**
- **Fase B — fotos recibidas (hojas, fondos, alteradas):**
- **Resultados por fondo × modo:**
- **Veredicto H4–H6:**
- **Hojas alteradas:**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
