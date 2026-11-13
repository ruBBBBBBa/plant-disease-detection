# T04 — Extracción de características de color, textura y manchas

- **Estado:** Pendiente (siguiente, tras mergear T03)
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

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**
- **Comparación de variantes de sombra (train):**
- **Criterio «no concluyente» y cuántas imágenes de train afecta:**
- **Características: número por familia y tiempo por imagen:**
- **Hallazgos del análisis exploratorio (train):**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
