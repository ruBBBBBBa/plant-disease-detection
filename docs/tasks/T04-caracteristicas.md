# T04 — Extracción de características de color, textura y manchas

- **Estado:** En revisión
- **Rama:** `t04-features`
- **Depende de:** T03 (segmentación K-means + postprocesado, `configs/segmentation.json`)
- **Referencias:** `PROJECT_CONTEXT.md` §6; Tema 2 (espacios de color e histogramas), Tema 4
  (extracción de características: color, textura, forma), Tema 6 (descripción de objetos y
  regiones); decisiones D015–D019.

## Objetivo

Convertir cada hoja en un vector de **características explícitas e interpretables** de color,
textura y manchas, calculadas **solo dentro de la máscara de la hoja** (D017). Esa tabla será la
entrada de los clasificadores de la fase 4. Además, hay que resolver dos pendientes de T03:

- **D019 — sombras.** K-means incluye a veces la sombra proyectada de las hojas sanas, y esa
  sombra se puede confundir con necrosis.
- **D017 — «no concluyente».** Falta el criterio para declarar que una imagen no se puede analizar
  porque la máscara es inadecuada.

Esta tarea **no entrena clasificadores** ni da métricas de clasificación.

## Datos y protocolo (sin fugas)

- Imágenes: `data/metadata.csv`, PlantVillage, `in_subset == True` (train / val / test del
  subconjunto) y también el `test` completo, que se necesita para el análisis por enfermedad (D014).
- **Todo el diseño y el ajuste se hace con `train`**: umbrales de color, tratamiento de sombras,
  criterio de «no concluyente», elección de parámetros de GLCM/LBP y análisis exploratorio.
- Las características de `val` y `test` **se calculan pero no se analizan**: no se hacen gráficos
  ni estadísticas con ellas. Se usarán en la fase 4.
- La segmentación de T03 está congelada: se usa `configs/segmentation.json` tal cual.
- PlantDoc queda fuera (fase 5).
- No hay escalado ni normalización aprendida aquí; eso se ajusta con `train` en la fase 4.

## Alcance

1. **Preparación de la máscara (D019)** en `src/features.py` o `src/preprocessing.py`. Se comparan
   con `train` al menos tres variantes:
   - (a) la máscara tal cual;
   - (b) la máscara erosionada *n* píxeles;
   - (c) la máscara sin los píxeles de sombra, definidos como oscuros, poco saturados y conectados
     al borde de la máscara.

   **Criterio de comparación**, que se fija antes de mirar los resultados y se documenta: la
   proporción de píxeles «oscuros / tipo necrosis» dentro de la máscara debería ser baja en las
   hojas sanas y alta en las enfermas que tienen necrosis (late blight, early blight). Se mide la
   separación sano/enfermo de esa proporción en `train` (por ejemplo con el AUC univariante o con
   la diferencia de medianas) y se comprueba con rejillas de casos sanos con sombra que la variante
   elegida no borra lesiones reales del borde.
   La variante elegida se guarda en `configs/features.json`.

2. **Características** (`src/features.py`). Cada función devuelve un diccionario
   `{nombre: valor}` con nombres estables, y su docstring explica la técnica y por qué podría
   relacionarse con síntomas:
   - **Color:** media y desviación típica de H, S y V y de L\*, a\* y b\* (atención a los rangos de
     OpenCV: H va de 0 a 179 y a\*/b\* están desplazados +128); histogramas gruesos y normalizados
     de H y de a\*/b\*; proporciones de tonos anómalos (amarillento, marrón, oscuro) con umbrales
     explícitos elegidos con `train`.
   - **Textura:** GLCM (`skimage.feature.graycomatrix` / `graycoprops`: contraste, homogeneidad,
     energía, correlación, promediando ángulos) y un histograma LBP uniforme
     (`skimage.feature.local_binary_pattern`), ambos **solo sobre píxeles de la hoja**. Documentar
     cómo se excluye el fondo (por ejemplo, un nivel reservado en la GLCM o recorte más máscara).
   - **Manchas / regiones:** máscara de **regiones candidatas** = píxeles de la hoja con color
     anómalo, más limpieza morfológica. Sobre ella: número de componentes conexas, fracción de área
     de la hoja, área media y máxima, y forma media (solidez, excentricidad, circularidad).
     Esta máscara es **evidencia visual de zonas sospechosas**; no es «dónde ve la enfermedad el
     clasificador» (regla 8 de `AGENTS.md`).
   - **Calidad de la máscara:** fracción de la imagen que ocupa la hoja, si la máscara toca el
     borde de la imagen y su solidez. Se usa para el criterio de «no concluyente».

3. **Criterio «no concluyente» (D017).** Se define con la distribución de `train`, por ejemplo
   «área de la hoja por debajo del percentil 1 de train» o «máscara que toca más de X % del borde
   de la imagen». Se informa de cuántas imágenes de `train` caen fuera. El criterio queda en
   `configs/features.json`; aquí no se usa `val` para ajustarlo.

4. **`scripts/extract_features.py`**:
   - CLI: `--splits`, `--subset-only` / `--full-test`, `--config` y `--out`.
   - Escribe `data/processed/features_<versión>.csv`: una fila por imagen con `image_id`, `split`,
     `binary_label`, `disease_name`, `leaf_group_id`, `inconclusive` y todas las características.
     No se versiona.
   - Registra el tiempo medio por imagen.

5. **Análisis exploratorio solo con `train`** (`reports/features/`):
   - `features_train.md`: tabla con el AUC univariante de cada característica (sano/enfermo),
     ordenada, y las correlaciones más altas entre características.
   - Boxplots de las 8–10 características más separadoras.
   - Rejillas de 2–3 hojas por enfermedad con su máscara de regiones candidatas.
   - Rejilla de las variantes de sombra.
   - Conclusión en texto: qué familias (color, textura, manchas) parecen separar más y cuáles
     parecen redundantes, **sin anticipar resultados de clasificación**.

6. **`docs/FEATURES.md`**: diccionario de características (nombre, familia, cálculo, rango y por qué
   podría relacionarse con síntomas). Servirá directamente para la memoria.

7. **Tests** (`tests/test_features.py`) con imágenes sintéticas:
   - una hoja verde uniforme da una fracción de tonos anómalos casi nula y 0 componentes;
   - una mancha marrón de área conocida da la fracción y el número de componentes esperados;
   - el fondo no influye en las características (misma hoja sobre dos fondos distintos → mismo vector);
   - los nombres de las características y su orden son estables;
   - la variante de sombra no elimina una lesión interior.

## Fuera de alcance

Clasificadores, escalado y selección de características basada en un modelo (fase 4). PlantDoc
(fase 5). Inferencia e interfaz (fase 6). Cambiar la segmentación de T03.

## Criterios de aceptación

- [ ] Variantes de sombra comparadas con un criterio fijado de antemano y elección justificada con `train`.
- [ ] Criterio «no concluyente» definido con `train` y documentado.
- [ ] `features_*.csv` generado para el subconjunto (train/val/test) y para el test completo, de
      forma reproducible (dos ejecuciones → mismo SHA-1).
- [ ] Informe exploratorio y figuras **solo de `train`**.
- [ ] `docs/FEATURES.md` completo.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; ficheros de `reports/` < 2 MB.

## Decisiones que quedan para el alumno al cerrar T04

1. Variante de máscara para las características (tal cual, erosionada o sin sombras).
2. Criterio de «no concluyente».
3. Qué grupos de características se comparan en la fase 4 (color, textura, manchas, combinadas;
   `PROJECT_CONTEXT.md` §7).

## Criterio de las variantes de sombra (fijado antes de ver resultados, 2026-11-13)

- Píxel «oscuro» = V < 60 en HSV de OpenCV (0–255). Medida: fracción de píxeles oscuros dentro
  de la máscara.
- **Criterio principal:** AUC univariante de esa fracción entre `healthy` y las enfermedades con
  necrosis (`late_blight` + `early_blight`), solo con `train`. **Secundario:** AUC `healthy` frente
  a todas las enfermas.
- Los parámetros de cada variante (*n* de la erosión; umbrales V y S de la sombra) se eligen con el
  mismo criterio, también en `train`.
- **Desempate:** se elige la variante más simple (a < b < c) salvo que otra mejore el criterio
  principal en ≥ 0,01.
- **Control:** la fracción de píxeles oscuros que cada variante elimina en hojas de *blight*
  (pérdida de lesión) y rejillas de sanas con sombra y de *blight* con lesión en el borde.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**

  ```bash
  git merge origin/main                          # T03 (PR #3) con configs/segmentation.json
  python scripts/tune_features.py --seed 42      # solo train (1.406 imágenes), ~1,5 min
  python scripts/extract_features.py --subset-only   # 1.998 imágenes (train/val/test del subconjunto)
  python scripts/extract_features.py --full-test     # 1.900 imágenes (test completo de PlantVillage)
  sha1sum data/processed/features_v1_*.csv       # repetida la extracción → mismos SHA-1:
  #   04020f706325ec6bcfd1319dd4de754306be947f  features_v1_subset.csv
  #   560c1f1e82de33a706eaa40f0307579e5054d453  features_v1_test_full.csv
  python scripts/analyze_features.py --seed 42   # solo filas train
  ruff check . && ruff format --check . && pytest   # 70 passed
  git -C data/raw/plantvillage status --short | wc -l   # 0
  ```

  El criterio de sombras se registró en este fichero (commit `9a4d2be`) **antes** de calcular
  nada. Los análisis y figuras solo usan `train`. `analyze_features.py` descarta las filas de `val`
  y `test` al cargar el CSV, y `extract_features.py` solo escribe recuentos por split. PlantDoc no
  se usa y la segmentación de T03 no se toca.

- **Comparación de variantes de sombra (train):** detalle en `reports/features/tuning_train.md` y
  `shadow_variants_train.jpg`.
  - **El criterio fijado de antemano no discrimina.** Con V < 60, el 18 % de los píxeles de las
    hojas **sanas** ya son «oscuros» (mediana 0,179; *blight* 0,173): es el verde oscuro normal. El
    AUC sanas frente a *blight* queda entre 0,473 y 0,489 en las 14 combinaciones: (a) 0,480; mejor
    (b) n = 8, 0,484; mejor (c) V < 50 y S < 80, 0,489.
  - Ninguna mejora en ≥ 0,01, así que la regla de desempate elige **(a) la máscara tal cual**.
  - **Pérdida de lesión:** la erosión quita entre el 16 % (n = 2) y el 37 % (n = 8) de los píxeles
    oscuros de las hojas de *blight*. La variante de sombra, entre el 0,4 % y el 4,8 %.
  - Las rejillas muestran que ninguna variante quita de verdad la sombra de las sanas. En algunas
    *blight* es la máscara de T03 la que ya excluye la necrosis oscura.
  - **Sensibilidad posterior** (columnas `posthoc_*`, solo informan): con V < 30 o V < 15 el AUC
    **baja** a 0,30–0,39. En `train`, las sanas tienen **más** píxeles oscuros que las hojas con
    *blight*. El riesgo de D019 («sombra = falsa necrosis») apunta en el sentido contrario: la
    oscuridad es más frecuente en las sanas por cómo están fotografiadas.

- **Criterio «no concluyente» y cuántas imágenes de train afecta:**
  - Se calcularon tres reglas candidatas con `train`: área de la hoja < p1 (0,2154), solidez de la
    máscara < p1 (0,6504) y fracción de borde > p99 (0,1627). Juntas marcan 41 imágenes.
  - Revisión visual (`inconclusive_train.jpg`, un bloque por regla):
    - **Área:** marca sobre todo hojas pequeñas bien segmentadas.
    - **Borde:** marca primeros planos bien segmentados.
    - **Solidez:** recoge la mayoría de las máscaras fragmentadas (necrosis perdida, fondo
      incluido), más 4 hojas de YLCV rizadas que están bien.
  - **Regla activa propuesta: `mask_solidity < 0,6504` → 14 de 1.406 imágenes de train**, todas
    enfermas. Las otras dos quedan en `inconclusive_candidates` (`configs/features.json`).
  - Ojo: con cualquier regla, todas las marcadas en `train` son enfermas. Excluirlas en la
    evaluación quitaría enfermas difíciles; la fase 4 debe contarlas aparte, no descartarlas.

- **Características: número por familia y tiempo por imagen:**
  - **75 en total:** calidad 3, color 43, textura 22 (GLCM d = 1 con 32 niveles; LBP uniforme
    P = 16, R = 2) y manchas 7. Diccionario completo en `docs/FEATURES.md`.
  - **Tiempo:** 27–44 ms por imagen, incluida la segmentación, en un i7-1355U con un hilo; varía
    con la carga entre ejecuciones (`reports/features/extraction.md`).
  - Los CSV pesan unos 1,5 MB y no se versionan; se regeneran con los comandos de arriba y se
    verifican con su SHA-1.

- **Hallazgos del análisis exploratorio (train)** (`reports/features/features_train.md`, 1.392
  imágenes, sin las 14 no concluyentes):
  - **Separación mediana |AUC − 0,5| por familia:** textura 0,397, manchas 0,214, color 0,193 y
    calidad 0,070.
  - **Las más separadoras:** `glcm_contrast` (AUC 0,013; sanas 35,5 frente a enfermas 10,3),
    `glcm_homogeneity` (0,96), varios bins LBP (0,91–0,95), `l_std` y `v_std` (0,05).
  - **Señal de alarma — sesgo de captura.** La nitidez (varianza del laplaciano dentro de la hoja,
    calculada solo como diagnóstico) separa sanas de enfermas con AUC 0,014. La mediana es 20.112 en
    sanas y de 952 a 9.311 en enfermas. La textura separa casi igual que la nitidez, así que buena
    parte de esa separación puede venir de que el lote de sanas está fotografiado más nítido, no de
    los síntomas. No se puede separar ambas cosas dentro de PlantVillage: es una pregunta para la
    fase 4 (comparar familias) y la fase 5 (PlantDoc).
  - **Color:** los histogramas de a\* (`a_hist_04` 0,12; `a_hist_01` 0,86), `s_mean` (0,80) y
    `a_mean` (0,20) separan de forma moderada. Las enfermas son más saturadas (S media 0,43 frente
    a 0,32) y tienen a\* **más negativo** (−14,2 frente a −10,3, verde más puro). Otra vez puede
    ser la captura, no un síntoma. `frac_yellow` (0,73) y `frac_brown` (0,67) separan menos de lo
    esperable por sí solas.
  - **Manchas:** separación moderada (`spot_area_frac` 0,75). Capta bien lesiones marrones y casi
    nada en sanas, pero apenas la clorosis suave (YLCV, *leaf mold*) ni a *spider mites*
    (`spots_train.jpg`).
  - **Redundancia:** 104 pares con |r| ≥ 0,9, casi todos dentro de la textura (93 entre bins LBP,
    más LBP–GLCM). Además, `l_std`/`v_std` (r 0,998) y `l_mean`/`v_mean` (0,995) son prácticamente
    la misma medida. La textura
    es la familia más redundante.
  - Separación univariante no es capacidad de clasificación: no se anticipan resultados de la fase 4.

- **Decisiones tomadas durante la tarea:**
  - **Umbral de tono:** H solo cuenta con V ≥ 50 y S ≥ 40. En `train`, los píxeles oscuros de
    hojas sanas daban picos espurios de H y un p1 de 15, que habría dejado vacía la clase
    «amarillento»; con píxeles fiables el p1 es 29.
  - **Media circular de H:** H es un ángulo.
  - **Error corregido gracias a un test:** el histograma LBP erosiona R + 1 y no R (con R = 2 los
    vecinos interpolados tocaban el fondo). Corregido antes de la extracción final.
  - **Regla «no concluyente» activa:** solo la solidez; las otras dos quedan como candidatas.
  - **Scripts:** tres (`tune_features`, `extract_features` y `analyze_features`) en lugar de uno,
    para separar el ajuste (`train`), la extracción (todos los splits) y el análisis (`train`).
  - **Propuestas registradas:** D020–D022 en `docs/DECISIONS.md`.

- **Pendientes (para el alumno):**
  1. **Variante de máscara (D020):** aceptar «tal cual», que es lo que da el criterio fijado. El
     problema de sombras de D019 no se resuelve con estas variantes; en `train` la oscuridad es más
     frecuente en sanas.
  2. **«No concluyente» (D021):** aceptar solo la solidez (14 de train), activar también área o
     borde, o no usar ninguna. En todos los casos solo afecta a enfermas.
  3. **Grupos para la fase 4:** color, textura, manchas y combinadas, como dice `PROJECT_CONTEXT.md`
     §7. Dado el sesgo de nitidez, conviene comparar color + manchas **sin** textura y valorar en
     PlantDoc si la textura generaliza. Otra opción para la fase 4 sería normalizar la nitidez
     (p. ej. un suavizado común antes de la textura), ajustado con `train`.
  4. **Excluir `l_std`/`v_std`:** también reflejan el sesgo de captura; decidir si se excluyen de
     la familia «color».
