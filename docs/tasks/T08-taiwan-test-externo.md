# T08 — Segundo test externo: dataset de hojas de tomate de Taiwán

- **Estado:** Pendiente (después de T07, o en paralelo si T07 está bloqueada esperando fotos)
- **Rama:** `t08-taiwan-external`
- **Depende de:** T06 (`eval_external.py`, criterio de «no concluyente», variantes A/B y límite de
  validez D035)
- **Referencias:** Huang, M.-L. y Chang, Y.-H. (2020). *Dataset of Tomato Leaves*. Mendeley Data, V1.
  DOI: 10.17632/ngdgg79rzb.1.

## Objetivo

Contrastar el **límite de validez** declarado en D035 (el sistema solo es fiable con una hoja sola
sobre fondo uniforme) con un dataset nuevo e independiente. Según su descripción, el subconjunto de
Taiwán tiene seis clases (cinco enfermedades y sanas) e imágenes con **una o varias hojas** y con
**fondo simple o complejo**. Así se puede comparar el rendimiento entre condiciones dentro de la
misma fuente, cosa que PlantDoc no permitía.

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

## Fase B — protocolo congelado y ejecución única

1. **Reutilizar `eval_external.py`** (parametrizado por fuente) con la misma normalización, el mismo
   criterio de «no concluyente» (D031) y la misma guarda: commit exacto, árbol limpio y se niega si
   existe cualquier `execution_*.json` en `experiments/external/taiwan_v1/`. Evalúa B (por defecto,
   D036) y A como referencia, en modo automático. Sin ROI: el dataset no trae rectángulos y
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
   - **Descriptivo, sin veredicto:** comparación con los resultados de PlantDoc (T06).

   Si alguna celda tiene menos de 20 imágenes de alguna clase, se informa igual, pero el veredicto
   de la hipótesis afectada se declara «no concluyente por tamaño».
4. **Informe** `experiments/external/taiwan_v1/summary.md` con las tablas, los veredictos, las
   enfermedades no vistas y una rejilla de aciertos y errores por condición. Atribución y licencia
   del dataset junto a las rejillas.
5. **Tests** con datos sintéticos: el mapeo de etiquetas de `taiwan`, la exclusión por casi-duplicado,
   el bootstrap no emparejado y la guarda.

## Fuera de alcance

Ajustar umbrales, segmentación o modelos con este dataset; entrenar con él; ROI manual; los otros
datasets encontrados (Bangladesh, Pakistán), que quedan como trabajo futuro.

## Criterios de aceptación

- [ ] Licencia comprobada; subconjunto de PlantVillage excluido; casi-duplicados frente a PlantVillage
      buscados y excluidos.
- [ ] Condiciones anotadas por el alumno y versionadas **antes** del commit congelado.
- [ ] Ejecución única registrada; predicciones guardadas; informe regenerable sin volver a predecir.
- [ ] Veredictos de H7 y H8 con IC, sin forzar conclusiones.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; nada pesado en git.

## Decisiones que quedan para el alumno al cerrar T08

1. Cómo se redacta el límite de validez en la memoria a la vista de PlantDoc, de Taiwán y de las fotos
   propias.
2. Qué trabajo futuro se propone.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Licencia y estructura real del dataset:**
- **Imágenes excluidas (PlantVillage, duplicados) y motivo:**
- **Recuento por clase y por condición `leaves × background`:**
- **Commit congelado y registro de la ejecución:**
- **Resultados (todas / solo concluyentes + cobertura), total y por condición:**
- **Veredicto H7–H8:**
- **Enfermedades no vistas:**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
