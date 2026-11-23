# T08 — Segundo test externo: dataset de hojas de tomate de Taiwán

- **Estado:** Pendiente (después de T07, o en paralelo si T07 está bloqueada esperando fotos)
- **Rama:** `t08-taiwan-external`
- **Depende de:** T06 (`eval_external.py`, criterio de «no concluyente», variantes A/B y límite de
  validez D035) y T07 (resultado de las fotos propias, D044, del que sale la hipótesis H9)
- **Referencias:** Huang, M.-L. y Chang, Y.-H. (2020). *Dataset of Tomato Leaves*. Mendeley Data, V1.
  DOI: 10.17632/ngdgg79rzb.1.

## Objetivo

Contrastar el **límite de validez** declarado en D035 (el sistema solo es fiable con una hoja sola
sobre fondo uniforme) con un dataset nuevo e independiente. Según su descripción, el subconjunto de
Taiwán tiene seis clases (cinco enfermedades y sanas) e imágenes con **una o varias hojas** y con
**fondo simple o complejo**. Así se puede comparar el rendimiento entre condiciones dentro de la
misma fuente, cosa que PlantDoc no permitía.

Además, contrasta una **hipótesis surgida de T07**. Con las fotos propias, el modo ROI segmentó los
folíolos de forma aparentemente adecuada, pero la variante B llamó «enferma» a casi todas las hojas
sanas (93 de 97 folíolos; 2 sanas y 2 no concluyentes), con contribuciones dominadas por el
histograma de b\*. La explicación probable, no demostrada, es que las características de color
dependen de la iluminación y del balance de blancos de la cámara. Si es así, **normalizar el color**
antes de extraer las características debería reducir el fallo en datos nuevos. Las fotos propias ya
están usadas, así que la hipótesis se comprueba aquí, con datos que nadie ha mirado.

Igual que PlantDoc, este dataset **se usa una sola vez** y nada se ajusta con él.

## Ya comprobado (21-nov-2026, solo la lista de ficheros del archivo del alumno)

- **Licencia:** CC BY 4.0, según la página del dataset en Mendeley Data.
- **El archivo descargado (`taiwan.7z`) solo contiene la parte de Taiwán**, no la de PlantVillage:
  5.598 JPG en dos carpetas.
  - `Preprocessed data/{Train,Test}/<clase>/`: **622 imágenes originales**, redimensionadas a
    227×227. Por clase: Bacterial spot 110, Black mold 67, Gray spot 84, Late blight 98, health 106 y
    powdery mildew 157.
  - `data augmentation/{Train,Test}/<clase>/`: 4.976 versiones aumentadas de las mismas imágenes
    (rotaciones, espejos y brillo; sufijos como `_mirror` o `_change_90`). **Se excluyen por
    completo**: son copias transformadas de las originales.
- El reparto Train/Test del propio dataset **no se usa**: las 622 originales son todas test externo
  (no se entrena con nada de aquí). Se conserva como columna `original_split`.
- **Enfermedades vistas en el entrenamiento:** Bacterial spot y Late blight. **No vistas:** Black mold,
  Gray spot y powdery mildew. El oídio (powdery mildew) produce manchas blancas, como el residuo de las
  fotos propias de T07: anotarlo como posible confusión al comparar.

Sin comprobar: dimensiones reales de cada imagen, ilegibles y duplicados (auditoría del punto 3).

## Fase A — descarga, auditoría y anotación de condiciones (antes de congelar)

1. **Descarga (la hace el alumno).** Desde Mendeley Data (DOI de arriba), a `data/raw/taiwan/`, que
   no se versiona. **Revisar la licencia** en la página del dataset y anotarla en `data/README.md`.
   Si la licencia no permite este uso, parar y avisar.
2. **Usar solo `Preprocessed data` (622 originales).** Se excluyen `data augmentation` y, si apareciera, el subconjunto de
   PlantVillage: ese se **excluye por completo**, porque solaparía con el entrenamiento.
3. **Auditoría** con `scripts/audit_dataset.py` (estructura real, clases, recuentos, tamaños,
   ilegibles, duplicados). Además, buscar **casi-duplicados frente a PlantVillage** con el dHash de
   T02, que tolera el cambio de tamaño de 256 a 227 px. Cualquier imagen con un casi-duplicado en
   PlantVillage se excluye y se informa. Ninguno de estos pasos ejecuta la segmentación ni los modelos.
4. **Mapeo de etiquetas** en `configs/label_mapping.csv` (fuente `taiwan`): sana → `healthy`; el resto
   → `diseased`, conservando el nombre original. Las enfermedades **no vistas** en el entrenamiento
   (las que no existen en PlantVillage) se marcan como tales para informarlas aparte.
5. **Anotación de condiciones (la hace el alumno, sin ver predicciones).**
   `data/annotations/taiwan_conditions.csv`, versionado, con las columnas `file, leaves, background`;
   `leaves` ∈ {`single`, `multiple`} y `background` ∈ {`simple`, `complex`}. Regla fija:
   - `simple` = fondo liso de un solo color (papel, mesa, cartulina);
   - `complex` = cualquier otra cosa (planta, suelo, objetos, varias texturas);
   - si hay dudas, `complex`.

   Una herramienta mínima que muestre cada imagen y guarde las dos etiquetas con dos teclas es
   suficiente. No muestra máscaras ni predicciones.
6. **Variante B′ con normalización de color (solo PlantVillage).**
   - **Normalización «gray world»** (supone que la media de la escena es gris neutro): se escala cada
     canal B, G y R para que su media en la imagen completa sea igual a la media de los tres canales,
     y se recorta a [0, 255] en `uint8`. Es una técnica clásica de constancia de color y no se aprende
     de los datos. Se documenta su punto débil: en imágenes donde la hoja ocupa casi todo, la «escena»
     es verde y la corrección también altera el color de la hoja.
   - **Dónde se aplica:** solo para **extraer las características**. La segmentación (K-means de
     T03) y el criterio de «no concluyente» se calculan sobre la imagen sin normalizar, igual que en
     B. Así B y B′ comparten máscara y cobertura, y solo cambia el color que ven las características.
   - **Entrenamiento:** mismo grupo (`color_manchas`) y mismo modelo (SVM lineal) que B, con la misma
     búsqueda de C por CV agrupada por hoja en `train` de PlantVillage. Se informa de la BA de `val` de
     B′ junto a la de B. **No se ejecuta el test interno de PlantVillage** con B′ (como C en T05).
   - Si la BA de `val` de B′ cae más de 5 puntos respecto a B, se informa como coste de la
     normalización, pero B′ se evalúa igualmente en Taiwán: la hipótesis es sobre generalización.
   - B′ **no se evalúa con las fotos propias**: ya están usadas.
   - Todo esto entra en el commit congelado de la fase B y se registra como decisión, citando que la
     hipótesis nace del resultado de T07.

## Fase B — protocolo congelado y ejecución única

1. **Reutilizar `eval_external.py`** (parametrizado por fuente) con la misma normalización, el mismo
   criterio de «no concluyente» (D031) y la misma guarda: commit exacto, árbol limpio y se niega si
   existe cualquier `execution_*.json` en `experiments/external/taiwan_v1/`. Evalúa B (por defecto,
   D036), B′ (normalización de color) y A como referencia, en modo automático. Sin ROI: el dataset no trae rectángulos y
   dibujarlos aquí sería otro proyecto.
2. **Métricas** (igual que en T06): lecturas «todas» y «solo concluyentes» con su cobertura;
   balanced accuracy, recall por clase, matriz de confusión; IC 95 % por bootstrap sobre imágenes.
   Se informan para el total y para cada combinación `leaves × background`, con el número de imágenes
   de cada celda. Las enfermedades no vistas van aparte.
3. **Hipótesis (variante B, fijadas antes de ejecutar).** Las comparaciones son entre grupos distintos
   de imágenes, así que se usa bootstrap no emparejado de la diferencia (10 000 réplicas, semilla 42).
   Se aplica la regla de siempre: apoyada / no apoyada (efecto contrario) / no concluyente.
   - **H7:** la cobertura (porcentaje de imágenes concluyentes) es mayor en `single + simple` que en
     el resto.
   - **H8:** en la lectura «todas», la balanced accuracy es mayor en `single + simple` que en el resto.
   - **H9 (surge de T07):** en el conjunto completo de Taiwán, lectura «todas», la balanced accuracy
     de B′ es mayor que la de B. Como las dos variantes ven las mismas imágenes con la misma máscara,
     esta comparación usa **bootstrap emparejado** sobre imágenes. Se informa también la diferencia en
     el recall de sanas, que es donde falló B en T07, de forma descriptiva y sin veredicto.
   - **Descriptivo, sin veredicto:** comparación con los resultados de PlantDoc (T06) y H9 desglosada
     por condición `leaves × background`.

   Si alguna celda tiene menos de 20 imágenes de alguna clase, se informa igual, pero el veredicto
   de la hipótesis afectada se declara «no concluyente por tamaño».
4. **Informe** `experiments/external/taiwan_v1/summary.md` con las tablas, los veredictos, las
   enfermedades no vistas y una rejilla de aciertos y errores por condición. Atribución y licencia
   del dataset junto a las rejillas.
5. **Tests** con datos sintéticos: el mapeo de etiquetas de `taiwan`, la exclusión por casi-duplicado, la normalización gray world (una imagen con dominante de color queda con medias de canal iguales y en `uint8`; la máscara de B′ es idéntica a la de B),
   el bootstrap no emparejado y la guarda.

## Fuera de alcance

Ajustar umbrales, segmentación o modelos con este dataset; entrenar con él; elegir el método de normalización mirando Taiwán o las fotos propias; ROI manual; los otros
datasets encontrados (Bangladesh, Pakistán), que quedan como trabajo futuro.

## Criterios de aceptación

- [ ] Licencia comprobada; subconjunto de PlantVillage excluido; casi-duplicados frente a PlantVillage
      buscados y excluidos.
- [ ] Condiciones anotadas por el alumno y versionadas **antes** del commit congelado.
- [ ] Ejecución única registrada; predicciones guardadas; informe regenerable sin volver a predecir.
- [ ] B′ entrenada solo con PlantVillage, con su BA de `val` junto a la de B, y congelada antes de procesar Taiwán.
- [ ] Veredictos de H7, H8 y H9 con IC, sin forzar conclusiones.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; nada pesado en git.

## Decisiones que quedan para el alumno al cerrar T08

1. Cómo se redacta el límite de validez en la memoria a la vista de PlantDoc, de Taiwán y de las fotos
   propias.
2. Qué trabajo futuro se propone.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Fase A — comandos ejecutados:**
  `python scripts/audit_dataset.py --source taiwan --compare-with reports/audit/plantvillage/inventory.csv`;
  `python scripts/prepare_taiwan.py --near-dup-threshold 2`;
  `python scripts/extract_features.py --subset-only --splits train val --gray-world`;
  `python scripts/train_classifiers.py cv-val-variant --features data/processed/features_v1_grayworld_subset.csv --out experiments/classification/exp2_grayworld`.
  Sobre Taiwán no se ha ejecutado segmentación, características ni modelos (solo lectura de ficheros
  para el inventario y el dHash).
- **Licencia y estructura real del dataset:** CC BY 4.0 (en `data/README.md`). `Preprocessed data`:
  622 JPG de 227×227, 3 canales, todos legibles; sin duplicados exactos dentro de Taiwán ni con
  PlantVillage (`reports/audit/taiwan/summary.md`). `data augmentation` no se lee.
- **Imágenes excluidas (PlantVillage, duplicados) y motivo:** ninguna. dHash, Hamming ≤ 2 contra las
  18 160 imágenes de PlantVillage: 0 casi-duplicados; distancia mínima 7 (22 imágenes entre 6 y 10,
  600 por encima de 10). 0 pares casi duplicados dentro de Taiwán (`reports/data/taiwan_summary.md`).
- **Recuento por clase y por condición `leaves × background`:** por clase, 622 `external_test`:
  health 106, Bacterial spot 110, Late blight 98 (vistas); Black mold 67, Gray spot 84,
  powdery mildew 157 (no vistas). Por condición: pendiente de la anotación del alumno.
- **Commit congelado y registro de la ejecución:**
- **Resultados (todas / solo concluyentes + cobertura), total y por condición:**
- **BA de `val` de B′ frente a B:** B′ 0,962838 (IC 95 % [0,926, 0,990]; C = 0,1; CV 0,9815) frente a
  B 0,945946 ([0,902, 0,976]). El procedimiento de `cv-val-variant` reproduce B exactamente con su CSV
  original (C = 1, 0,945946). Recall de sanas 0,986 frente a 0,946. Sin coste (D052).
- **Veredicto H7–H9:**
- **Enfermedades no vistas:**
- **Decisiones tomadas durante la tarea:** D050 (exploratorio a posteriori de T07), D051 (manifiesto y casi-duplicados), D052 (B′ y su limitación), D053 (anotación ciega a la clase).
- **Pendientes:**
