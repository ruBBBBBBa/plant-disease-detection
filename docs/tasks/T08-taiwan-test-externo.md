# T08 — Segundo test externo: dataset de hojas de tomate de Taiwán

- **Estado:** Hecha (PR #9)
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
   `leaves` ∈ {`leaflet`, `compound`, `multiple`} y `background` ∈ {`simple`, `complex`}.
   Reglas fijas (D053). Hojas, con tres niveles:
   - `leaflet` = un solo folíolo, sin otras hojas en primer plano (condición tipo PlantVillage);
   - `compound` = una sola hoja compuesta (varios folíolos con su raquis), sin otras hojas enteras;
   - `multiple` = se ven otras hojas enteras o casi enteras, o la planta;
   - si hay dudas, `multiple`.

   Fondo:
   - `simple` = fondo liso de un solo color (papel, mesa, cartulina);
   - `complex` = cualquier otra cosa (planta, suelo, objetos, varias texturas);
   - si hay dudas, `complex`.

   Herramienta: `scripts/annotate_conditions.py` (teclas `1`/`2`/`3` para hojas y `s`/`c` para fondo).
   Muestra cada imagen sin nombre ni clase, en orden aleatorio fijo, y no muestra máscaras ni
   predicciones.
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
   - **H7 (solo imágenes enfermas, D054):** la cobertura (porcentaje de imágenes concluyentes) es
     mayor en `leaflet + simple` que en el resto.
   - **H8 (solo imágenes enfermas, D054):** en la lectura «todas», el recall de enfermas es mayor en
     `leaflet + simple` que en el resto.
   - Por qué solo enfermas: en la anotación no hay ninguna imagen sana con fondo `simple` (las 106
     sanas son `complex`), así que la BA de `leaflet + simple` no existe y, con ambas clases, la
     condición se confundiría con la clase. Se ve en los recuentos de la anotación, antes de procesar
     ninguna imagen. Sigue habiendo confusión con la enfermedad (la mezcla de enfermedades cambia
     entre condiciones); se informa el desglose por enfermedad, solo descriptivo.
   - `compound + simple` se informa aparte, solo descriptivo (sin veredicto); en H7 y H8 forma parte
     del «resto».
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

- [x] Licencia comprobada; subconjunto de PlantVillage excluido; casi-duplicados frente a PlantVillage
      buscados y excluidos.
- [x] Condiciones anotadas por el alumno y versionadas **antes** del commit congelado.
- [x] Ejecución única registrada; predicciones guardadas; informe regenerable sin volver a predecir.
- [x] B′ entrenada solo con PlantVillage, con su BA de `val` junto a la de B, y congelada antes de procesar Taiwán.
- [x] Veredictos de H7, H8 y H9 con IC, sin forzar conclusiones.
- [x] `ruff` y `pytest` en verde; `data/raw/` intacto; nada pesado en git.

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
  powdery mildew 157 (no vistas). Por condición (anotación del alumno, `c8afde9`): `leaflet` 442,
  `compound` 116, `multiple` 64; `simple` 226, `complex` 396. Ninguna sana con fondo `simple`
  (0 de 106) → H7 y H8 solo con enfermas (D054).
- **Fase B — comandos ejecutados:**
  `python scripts/eval_external.py --source taiwan --frozen-commit c673c1e67292d0b894c255ba3413d66bed6ee61d`
  (una sola vez); `python scripts/report_taiwan.py`; `python scripts/plot_taiwan_grids.py --variant B`.
  Antes, prueba en seco fuera del repo con 8 imágenes de `val` de PlantVillage: A, B y B′ reproducen
  las predicciones guardadas de `val` (misma clase; puntuación con diferencia ≤ 2·10⁻⁵).
- **Commit congelado y registro de la ejecución:** `c673c1e`;
  `experiments/external/taiwan_v1/execution_c673c1e67292d0b894c255ba3413d66bed6ee61d.json` (sha1 de
  configuración, anotaciones, modelos y predicciones). `report_taiwan.py` se amplió después (motivos
  de «no concluyente»); no predice ni cambia nada de lo congelado.
- **Resultados (todas / solo concluyentes + cobertura), total y por condición:** cobertura 36,0 %
  (398 de 622 no concluyentes: 15 máscaras vacías, el resto por calidad de máscara). BA «todas» /
  «solo concluyentes»: A 0,129 / 0,349, B 0,151 / 0,414, B′ 0,122 / 0,374. Entre las concluyentes,
  B acierta 3 de 26 sanas. Por condición (B): cobertura 62,6 % en `leaflet + simple`, 68,2 % en
  `compound + simple`, 17–24 % en las `complex`; BA solo existe en las `complex` (≤ 0,116). Detalle en
  `experiments/external/taiwan_v1/summary.md` y `tables.md`.
- **BA de `val` de B′ frente a B:** B′ 0,962838 (IC 95 % [0,926, 0,990]; C = 0,1; CV 0,9815) frente a
  B 0,945946 ([0,902, 0,976]). El procedimiento de `cv-val-variant` reproduce B exactamente con su CSV
  original (C = 1, 0,945946). Recall de sanas 0,986 frente a 0,946. Sin coste (D052).
- **Veredicto H7–H9:** H7 **apoyada** (cobertura de enfermas, `leaflet + simple` − resto = 0,394
  [0,310; 0,477]); H8 **apoyada** (recall de enfermas «todas», 0,237 [0,156; 0,319]; no es
  independiente de H7); H9 **no concluyente** (BA B′ − B = −0,029 [−0,057; 0,002]; recall de sanas
  +0,047 [0,000; 0,094], descriptivo).
- **Lectura de los resultados (sin cambiar ningún número):**
  - **H7** indica que la condición `leaflet + simple` aumenta la **analizabilidad** (cobertura), no que
    la clasificación sea correcta.
  - **H8 depende de H7:** con un sistema sesgado hacia «diseased», el recall de enfermas sube con la
    cobertura. No es evidencia independiente de buena clasificación.
  - **Limitación principal de T08:** ninguna sana tiene fondo `simple`, así que la especificidad en la
    condición válida (`leaflet + simple`) no se ha podido medir.
  - **B en «solo concluyentes» (BA 0,414) está por debajo del azar** por el recall de sanas (3/26);
    el sesgo de T07 se repite, pero depende de la condición: en `leaflet + simple`, B clasifica como
    sanas 41 de 124 enfermas concluyentes; con fondo complejo, casi todas las sanas salen enfermas
    (23 de 26 sanas concluyentes). Lectura descriptiva (la mezcla de enfermedades cambia entre
    condiciones): las decisiones siguen las condiciones de captura más que los síntomas.
  - **H9:** no concluyente, con punto estimado negativo. Gray world con los umbrales de T04 no corrige
    el sesgo.
  - **Autoría de la anotación:** las etiquetas de `data/annotations/taiwan_conditions.csv` son de
    Rubba; el commit `c8afde9` (hecho por el implementador) fue solo mecánico.
- **Enfermedades no vistas:** recall de B «todas» / «solo concluyentes»: black mold 0,358 / 0,706,
  gray spot 0,262 / 0,786, powdery mildew 0,191 / 0,698; parecido a las vistas, pero con sanas casi
  siempre llamadas enfermas no indica que se reconozcan.
- **Decisiones tomadas durante la tarea:** D050 (exploratorio a posteriori de T07), D051 (manifiesto y casi-duplicados), D052 (B′ y su limitación), D053 (anotación ciega a la clase), D054 (H7/H8 solo con enfermas), D055 (protocolo congelado de la fase B), D056 (límite de validez) y D057 (trabajo futuro), decididas por el alumno al cerrar.
- **Pendientes:** ninguno de T08; las dos decisiones del alumno quedan registradas en D056 (límite
  de validez) y D057 (trabajo futuro). Fuera de alcance y sin hacer: reajustar umbrales para color normalizado; segmentación
  para fondos complejos o negros; ROI manual.


### Revisión Codex — informe por condición

- P1 corregido: para A, B y B′ se incluyen ambas lecturas por celda `leaves × background`,
  cada una con matriz [[TN, FP], [FN, TP]], recall por clase, BA, soportes por clase de la
  lectura y de la celda completa, y cobertura (concluyentes / total de la celda).
- Sin una clase, su recall y la BA se muestran como «no medible». Sin concluyentes,
  matriz nula y métricas no medibles. Test sintético con una celda sin sanas y otra lectura vacía.
- `python scripts/report_taiwan.py` regeneró tablas y métricas solo desde datos guardados.
  El bloque por condición de `summary.md` procede de las nuevas tablas.
- Comparación estructural con el JSON previo: totales, métricas globales, por enfermedad,
  lecturas «todas» por celda e hipótesis H7–H9 idénticos, incluidos IC y veredictos.
- Ruff y formato correctos; pytest: 137 passed. No se ejecutó `eval_external.py`, ni se
  procesaron imágenes, ni se cambiaron modelos, umbrales o decisiones.
