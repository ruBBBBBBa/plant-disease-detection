# T06 — Test externo en PlantDoc y análisis de errores

- **Estado:** En revisión
- **Rama:** `t06-plantdoc-external`
- **Depende de:** T05 (modelos `exp2`, variantes preregistradas en D030)
- **Referencias:** `PROJECT_CONTEXT.md` §2 («análisis no concluyente») y §7 (test externo);
  D010 (PlantDoc), D014 (protocolo), D021 (criterio pendiente), D023 (sesgo de nitidez), D030.

## Objetivo

Medir cómo generaliza el sistema a fotos en **condiciones naturales**, con fondos, iluminación,
resoluciones y encuadres distintos. Esta tarea contrasta las hipótesis preregistradas en D030 y
analiza los fallos. PlantDoc **se usa una sola vez**: es la prueba que no se puede repetir.

Se espera que el rendimiento baje respecto a PlantVillage. El objetivo no es «sacar buen número»,
sino **medir y explicar** cuánto baja, por qué y qué variante aguanta mejor.

## Regla de oro: nada se decide mirando PlantDoc

Todo lo que el pipeline necesite para procesar PlantDoc se fija **antes** de ejecutarlo, en un
commit congelado, **sin abrir ni procesar ninguna imagen de PlantDoc**:

- **Normalización de tamaño** (preregistrada): reescalar cada imagen para que su lado mayor mida
  256 px (`cv2.INTER_AREA`, manteniendo la proporción), como las de PlantVillage. Las imágenes con
  4 canales se convierten a BGR al leerlas.
- **Segmentación, características y modelos:** los de T03, T04 y T05 (`exp2`), tal cual.
- **Criterio de «no concluyente» (D021)** a partir de **PlantVillage**, no de PlantDoc: una imagen
  es no concluyente si su máscara está vacía, o si alguna de las características de calidad
  (fracción de área de la hoja, toque del borde de la imagen, solidez) cae fuera del rango
  [p0,5 ; p99,5] de `train` + `val` de PlantVillage. Los umbrales se calculan y se guardan en
  `configs/external.json` antes de ejecutar.
- **Métricas, comparaciones e hipótesis**, escritas antes de ejecutar (ver abajo).

## Hipótesis preregistradas (D030)

Se mide la **caída** de cada variante: `Δ = BA(val de PlantVillage) − BA(PlantDoc)`. Se usa val y
no test porque C y C_lineal solo se evaluaron en val; así se compara igual para todas.

- **H1:** B (`color_manchas` + SVM lineal) cae menos que A (`combinadas` + SVM lineal).
- **H2:** C_lineal (`color_manchas_sin_l_v` + SVM lineal) cae menos que B.
- **H3 (control):** el modelo `control_nitidez` cae hasta cerca del azar en PlantDoc. Si se cumple,
  apoya que la nitidez era un atajo propio de la captura de PlantVillage.

Cada hipótesis se resuelve con un **bootstrap emparejado sobre las imágenes de PlantDoc** de la
diferencia de BA entre las dos variantes (IC 95 %). Una hipótesis solo se da por apoyada si el IC
de la diferencia no incluye el 0; si lo incluye, el resultado es **no concluyente** y se dice así.

## Alcance

1. **`scripts/eval_external.py`.** Reutiliza la guarda de `eval_test.py`: exige el commit exacto, un
   árbol limpio y un registro exclusivo por commit, y se niega a ejecutarse dos veces. Procesa las
   743 imágenes `external_test` de `data/metadata.csv` (normalización → segmentación → características
   → modelos) y guarda las predicciones y puntuaciones por imagen y variante (comprimidas, < 2 MB).
   Evalúa:
   - las cuatro variantes preregistradas y el control de nitidez (principal);
   - el resto de combinaciones grupo × modelo de `exp2`, como **apéndice descriptivo** (no se usan
     para elegir nada).

2. **Métricas** (por variante), en dos lecturas, ambas obligatorias:
   - **Todas las imágenes:** las no concluyentes cuentan como error.
   - **Solo concluyentes:** con la **cobertura** (porcentaje de imágenes analizadas) al lado. Una
     métrica sin su cobertura no se presenta.

   Para cada lectura: matriz de confusión; recall de sanas y de enfermas; balanced accuracy; F1;
   ROC-AUC con las puntuaciones; IC 95 % por bootstrap sobre imágenes; recall por enfermedad.
   PlantDoc está muy desequilibrado (≈ 63 sanas frente a ≈ 680 enfermas): no se informa de la
   accuracy sin la balanced accuracy al lado, y se avisa de que los IC de las sanas serán anchos.

3. **Casos especiales que se informan aparte:**
   - `mosaic_virus`: enfermedad **no vista** en el entrenamiento (D006);
   - `yellow_virus`: correspondencia dudosa con Yellow Leaf Curl (D010);
   - `spider_mites`: solo 2 imágenes;
   - `tomato_V8.jpg`: etiqueta en conflicto, pero `diseased` en el binario.

4. **Diagnóstico del cambio de dominio** (descriptivo): comparar las distribuciones de PlantDoc con
   las de `train` de PlantVillage para nitidez, `l_std`, `v_std`, `glcm_contrast`, fracción de área
   de la hoja y toque del borde (histogramas superpuestos y medianas). Sirve para explicar las
   caídas, no para ajustar nada.

5. **Análisis de errores:**
   - rejillas de falsos negativos, falsos positivos y no concluyentes de la variante A y de la mejor
     de las otras tres en PlantDoc, con la máscara de la hoja superpuesta;
   - clasificación manual de los fallos en categorías fijadas antes de mirarlos: segmentación fallida,
     varias hojas o planta entera, fondo complejo, iluminación o desenfoque, síntoma poco visible,
     etiqueta dudosa y otros;
   - recuento por categoría.

6. **Informe** `experiments/external/plantdoc_v1/summary.md`: tabla de variantes con las dos
   lecturas, caídas Δ con IC, veredicto de H1–H3 (apoyada / no apoyada / no concluyente), casos
   especiales, diagnóstico de dominio y análisis de errores. Más `docs/DECISIONS.md` con lo decidido.

7. **Tests** (`tests/test_external.py`), con imágenes sintéticas:
   - la normalización deja el lado mayor en 256 y conserva la proporción;
   - una imagen de 4 canales se lee como BGR;
   - el criterio de «no concluyente» marca máscaras vacías y valores fuera de rango;
   - las dos lecturas de métricas dan lo esperado en un caso pequeño;
   - el bootstrap emparejado usa las mismas imágenes para las dos variantes;
   - la guarda impide una segunda ejecución.

## Fuera de alcance

Reentrenar, reajustar o cambiar cualquier parámetro después de ver PlantDoc. Usar PlantDoc para
entrenar. Fotos propias (posible tarea aparte, con etiquetas fiables). Interfaz e inferencia (fase 6).
Las mejoras que se ocurran al ver los errores van a «Pendientes» como trabajo futuro.

## Criterios de aceptación

- [ ] Normalización, criterio de «no concluyente», métricas e hipótesis fijados en un commit
      **anterior** a la ejecución, sin haber procesado PlantDoc (comprobable con git).
- [ ] Una sola ejecución registrada; predicciones guardadas; tablas y rejillas regenerables sin
      volver a predecir.
- [ ] Las dos lecturas (todas / solo concluyentes con su cobertura) para cada variante.
- [ ] Veredicto de H1–H3 con IC del bootstrap emparejado y sin forzar conclusiones.
- [ ] Casos especiales, diagnóstico de dominio y análisis de errores por categorías.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; nada pesado en git.

## Decisiones que quedan para el alumno al cerrar T06

1. Qué variante será el modelo por defecto de la herramienta de inferencia (fase 6), a la vista de
   PlantVillage **y** PlantDoc. La elección se justifica, pero los números de PlantDoc no se reajustan.
2. Si el criterio de «no concluyente» se mantiene en la herramienta final.
3. Qué limitaciones y trabajo futuro se llevan a la memoria.

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Commit congelado y registro de la ejecución:** `5a2c39ca90e1bcb3729f705a042103144254b36c`
  (protocolo y umbrales en `9bd8b7e`; guarda y tests en `5a2c39c`; ninguno procesó PlantDoc).
  Comando: `python scripts/eval_external.py --frozen-commit 5a2c39ca90e1bcb3729f705a042103144254b36c`
  → «743 imágenes, 575 no concluyentes». Registro: `experiments/external/plantdoc_v1/execution_5a2c39ca….json`
  (`status: complete`, SHA-1 de predicciones y pipelines). Una sola ejecución.
  Tras ella: `python scripts/report_external.py` y `python scripts/plot_external_errors.py`
  (leen las predicciones; no predicen). `ruff` y `pytest` en verde (99 tests; 10 nuevos en `tests/test_external.py`).
- **Umbrales de «no concluyente» (desde PlantVillage train+val, 1 702 imágenes):**
  `leaf_area_frac` [0,180; 0,616], `border_touch_frac` [0,000; 0,213], `mask_solidity` [0,593; 0,961]
  (`configs/external.json`). La regla marca el 2,3 % de PlantVillage train+val como referencia.
  **En PlantDoc marca el 77,4 %** (área 49,7 %, solidez 39,9 %, borde 20,5 %, máscara vacía 0,8 %).
- **Resultados por variante (todas / solo concluyentes + cobertura 22,6 %):** BA «todas» ≈ 0,11 en las cinco
  variantes (A 0,114, B 0,115, C 0,111, C_lineal 0,114, control 0,112); BA «solo concluyentes» ≈ 0,50
  (A 0,505, B 0,508, C 0,497, C_lineal 0,505, control 0,500; n = 168, solo 16 sanas). Tabla completa con IC,
  recalls, F1 y ROC-AUC en `experiments/external/plantdoc_v1/summary.md`. La AUC «todas» < 0,5 es un artefacto
  de la convención de no concluyentes.
- **Caídas Δ y veredicto de H1–H3:** Δ ≈ 0,83–0,85 en todas. H1 (A−B) 0,014 [0,010; 0,019] apoyada;
  H2 (B−C_lineal) 0,003 [0,001; 0,003] apoyada; H3 (control−A) −0,015 [−0,026; 0,004] no concluyente.
  H1 y H2 son diferencias de una o pocas imágenes sobre 168 concluyentes: no son una ventaja real. La BA de val
  se toma como fija (los IC no recogen su incertidumbre). BA del control 0,112 [0,096; 0,128], no incluye 0,5.
- **Casos especiales:** `mosaic_virus` 40/54 no concluyentes (recall 0,93 en las 14 concluyentes);
  `yellow_virus` 64/75 (0,91 en 11); `spider_mites` 2/2 y `tomato_V8.jpg` 1/1 no concluyentes.
- **Diagnóstico de cambio de dominio:** nitidez ×0,07, `glcm_contrast` ×0,19, `l_std` ×0,75, `v_std` ×0,80,
  área de la hoja ×0,60, toque del borde 0 → 0,069 (medianas PlantDoc/PlantVillage train). `domain_shift.png`.
- **Errores por categoría (clasificación manual, una persona):** no concluyentes (muestra de 48): segmentación
  fallida 28, varias hojas 8, fondo complejo 5, otros 7. FN de A (8): segmentación 2, varias hojas 3, fondo 1,
  síntoma poco visible 1, etiqueta dudosa 1. FP de A (15): otros 8 (hojas sueltas sobre fondo blanco), varias hojas 3,
  iluminación 2, segmentación 1, etiqueta dudosa 1. Rejillas y `error_categories.csv` en `experiments/external/plantdoc_v1/`.
- **Decisiones tomadas durante la tarea:** D031–D034 (H3 redefinida por el alumno, guarda reforzada, control = SVM
  lineal, BA de val fija). Los scripts de informe y rejillas se escribieron después de la ejecución; la lógica de
  métricas ya estaba congelada en `src/external.py`.
- **Pendientes / trabajo futuro:** revisar el criterio de «no concluyente» (decisión del alumno, no tocado);
  segmentación más robusta; fotos propias con más sanas; entender por qué las sanas sobre fondo blanco salen enfermas.
