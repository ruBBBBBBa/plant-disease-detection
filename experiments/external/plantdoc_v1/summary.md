# T06 — Test externo en PlantDoc (resumen)

Ejecución única con el commit congelado `5a2c39ca90e1bcb3729f705a042103144254b36c` (743 imágenes
`external_test`: 680 enfermas, 63 sanas). Todas las cifras salen de `metrics.json` y `tables.md`
(`python scripts/report_external.py`), calculadas sobre las predicciones guardadas en
`predictions/`. Nada se reajustó tras ver los resultados.

## Hallazgo principal: el criterio de «no concluyente» descarta el 77 % de PlantDoc

Con los umbrales de PlantVillage (`configs/external.json`), **575 de 743 imágenes (77,4 %) son no
concluyentes**; la cobertura es del **22,6 %** (168 imágenes). En PlantVillage train+val la misma
regla marca el 2,3 %. Motivos (se solapan): área de la hoja fuera de rango 49,7 %, solidez 39,9 %,
toque del borde 20,5 %, máscara vacía 0,8 % (PlantVillage: 1,1 / 1,1 / 0,5 / 0 %).

Con esa cobertura, lo que mide el sistema en PlantDoc es sobre todo **si logra encontrar una hoja
bien encuadrada**, no si clasifica bien. Los números de la lectura «todas» están dominados por 575
errores forzados idénticos en todas las variantes.

## Variantes preregistradas (D030) — dos lecturas

Cobertura 22,6 % en ambas lecturas. IC 95 % por bootstrap sobre imágenes (2 000 réplicas).

| Variante | Lectura | n | BA [IC95] | Recall sanas | Recall enfermas | ROC-AUC [IC95] |
|---|---|---:|---|---:|---:|---|
| A | todas | 743 | 0,114 [0,095; 0,138] | 0,016 | 0,212 | 0,036 [0,019; 0,054] |
| A | solo concluyentes | 168 | 0,505 [0,459; 0,578] | 0,062 | 0,947 | 0,629 [0,495; 0,763] |
| B | todas | 743 | 0,115 [0,095; 0,139] | 0,016 | 0,213 | 0,035 [0,019; 0,054] |
| B | solo concluyentes | 168 | 0,508 [0,462; 0,580] | 0,062 | 0,954 | 0,622 [0,500; 0,742] |
| C | todas | 743 | 0,111 [0,096; 0,127] | 0,000 | 0,222 | 0,041 [0,023; 0,061] |
| C | solo concluyentes | 168 | 0,497 [0,490; 0,500] | 0,000 | 0,993 | 0,719 [0,614; 0,810] |
| C_lineal | todas | 743 | 0,114 [0,095; 0,138] | 0,016 | 0,212 | 0,034 [0,018; 0,052] |
| C_lineal | solo concluyentes | 168 | 0,505 [0,458; 0,577] | 0,062 | 0,947 | 0,600 [0,469; 0,730] |
| control | todas | 743 | 0,112 [0,096; 0,128] | 0,000 | 0,224 | 0,030 [0,016; 0,047] |
| control | solo concluyentes | 168 | 0,500 [0,500; 0,500] | 0,000 | 1,000 | 0,532 [0,377; 0,675] |

Matrices de confusión (filas = real sana/enferma; columnas = predicho sana/enferma):
A «todas» `[[1, 62], [536, 144]]`, A «solo concluyentes» `[[1, 15], [8, 144]]`.
El resto está en `metrics.json`.

Lectura de las cifras:

- **Sin sanas no hay clasificador.** Entre las 168 concluyentes solo hay 16 sanas, y A acierta 1.
  Predice «enferma» casi siempre: recall de enfermas 0,947, de sanas 0,062; BA 0,505 ≈ azar.
  Los IC de las sanas son anchísimos ([0,000; 0,200]).
- **ROC-AUC «todas» < 0,5 es un artefacto**: las no concluyentes se colocan en el peor extremo del
  ranking (convención de `predict_with_empty`). Solo la AUC «concluyentes» (0,53–0,72) es
  interpretable, y con IC amplios que incluyen 0,5 en A, B, C_lineal y control.
- C y control colapsan a «todo enfermo» entre las concluyentes (recall de sanas 0).

## Caídas Δ y veredicto de hipótesis

Δ = BA(val PlantVillage) − BA(PlantDoc, lectura «todas»). La BA de val se toma como **fija**: los IC
solo recogen la incertidumbre de PlantDoc (limitación).

| Variante | BA val | BA PlantDoc | Δ [IC95] |
|---|---:|---:|---|
| A | 0,959 | 0,114 | 0,846 [0,822; 0,865] |
| B | 0,946 | 0,115 | 0,831 [0,807; 0,851] |
| C | 0,953 | 0,111 | 0,842 [0,826; 0,857] |
| C_lineal | 0,943 | 0,114 | 0,829 [0,805; 0,848] |
| control | 0,943 | 0,112 | 0,831 [0,815; 0,846] |

| H | Contraste (Δ_peor − Δ_mejor > 0) | Diferencia [IC95] | Veredicto |
|---|---|---|---|
| H1 | Δ_A − Δ_B | 0,014 [0,010; 0,019] | apoyada |
| H2 | Δ_B − Δ_C_lineal | 0,003 [0,001; 0,003] | apoyada |
| H3 | Δ_control − Δ_A | −0,015 [−0,026; 0,004] | no concluyente |

Aplicada la regla preregistrada, H1 y H2 salen «apoyadas». **No deben leerse como una ventaja
real**: las diferencias son de 1,4 y 0,3 puntos de BA con todas las variantes en torno a 0,11, es
decir, todas caen ≈ 0,83–0,85. La lectura «todas» está dominada por el 77 % de imágenes no
concluyentes, comunes a todas; la diferencia A–B sale de **una sola imagen** concluyente (144 frente a
145 enfermas acertadas). Las diferencias de BA entre variantes en la lectura «solo concluyentes» (0,497–0,508) están
dentro del ruido de los IC. H3 es **no concluyente**.

Descriptivo, sin veredicto: BA del control en PlantDoc 0,112 IC [0,096; 0,128]; **no** incluye 0,5.
Está muy por debajo de 0,5 porque la lectura «todas» penaliza las no concluyentes. En «solo
concluyentes» su BA es exactamente 0,500 (predice «enferma» siempre).

## Casos especiales (variante A)

| Caso | n | No concluyentes | Recall «todas» | Recall «solo concluyentes» |
|---|---:|---:|---:|---:|
| `mosaic_virus` (no visto en entrenamiento, D006) | 54 | 40 | 0,241 | 0,929 (n = 14) |
| `yellow_virus` (correspondencia dudosa, D010) | 75 | 64 | 0,133 | 0,909 (n = 11) |
| `spider_mites` | 2 | 2 | 0,000 | sin datos |
| `tomato_V8.jpg` (etiqueta en conflicto; `diseased`) | 1 | 1 | 0,000 | sin datos |

Con n = 14 y 11 no se puede concluir que un virus no visto se detecte: los pocos concluyentes sí se
marcan como enfermos, pero el 74–85 % de estas imágenes ni se analiza.

## Diagnóstico de cambio de dominio (medianas; PlantVillage = train; `domain_shift.png`)

| Característica | PlantVillage | PlantDoc | Cociente |
|---|---:|---:|---:|
| nitidez (varianza del laplaciano) | 12 820 | 921 | 0,07 |
| `l_std` | 18,2 | 13,6 | 0,75 |
| `v_std` | 0,177 | 0,142 | 0,80 |
| `glcm_contrast` | 23,8 | 4,6 | 0,19 |
| fracción de área de la hoja | 0,428 | 0,255 | 0,60 |
| toque del borde | 0 | 0,069 | — |

La nitidez y el contraste GLCM caen 14 y 5 veces: coherente con D023 (la textura de PlantVillage
reflejaba en parte una captura nítida). La hoja ocupa menos imagen y toca más el borde. Es una
explicación descriptiva, no una prueba causal.

## Análisis de errores (rejillas `grid_*.jpg`, categorías en `error_categories.csv`)

Categorías fijadas antes de mirar: segmentación fallida, varias hojas o planta entera, fondo
complejo, iluminación o desenfoque, síntoma poco visible, etiqueta dudosa y otros. Clasificación
**manual y subjetiva** (una persona, sin segunda opinión). La variante «mejor de las otras tres» es B
(BA «todas» 0,115 frente a 0,114 y 0,111: una diferencia que no sostiene un «mejor»). Las no
concluyentes son una **muestra aleatoria de 48** (semilla 42) de las 575, no todas.

| Categoría | A FN (8) | A FP (15) | NC muestra (48) | B FN (7) | B FP (15) |
|---|---:|---:|---:|---:|---:|
| segmentación fallida | 2 | 1 | 28 | 2 | 1 |
| varias hojas o planta entera | 3 | 3 | 8 | 1 | 3 |
| fondo complejo | 1 | 0 | 5 | 0 | 0 |
| iluminación o desenfoque | 0 | 2 | 0 | 0 | 2 |
| síntoma poco visible | 1 | 0 | 0 | 2 | 0 |
| etiqueta dudosa | 1 | 1 | 0 | 1 | 1 |
| otros | 0 | 8 | 7 | 1 | 8 |

- **No concluyentes:** 28 de 48 (58 %) son segmentación fallida (máscara parcial o fragmentada); el
  resto, varias hojas, fondo complejo y «otros» (hoja que llena el encuadre, diapositivas, collages).
- **Falsos positivos (todas las sanas concluyentes):** 8 de 15 son hojas sueltas sobre fondo blanco
  de fotos de archivo (con marca de agua) que la segmentación recorta bien y aun así se clasifican
  como enfermas. No hay una explicación que se pueda afirmar con estos datos.
- **Falsos negativos:** 8 en A, de los que 6 son segmentación fallida, varias hojas o fondo complejo.

## Limitaciones

- La BA de val se toma como fija: los IC de Δ no recogen su incertidumbre.
- Solo 63 sanas (16 concluyentes): IC muy anchos y recall de sanas casi inestimable.
- La clasificación de errores es manual y por una sola persona; las no concluyentes se muestrean.
- H1/H2 sacan «apoyada» con diferencias prácticamente nulas: el contraste depende de la lectura
  «todas», dominada por el criterio de «no concluyente».
- Los scripts `report_external.py` y `plot_external_errors.py` se escribieron **después** de la
  ejecución única. No contienen lógica nueva de métricas (viven en `src/external.py`, congelada) y
  no vuelven a predecir.

## Pendientes (trabajo futuro, no implementado)

- Revisar el criterio de «no concluyente» con la distribución de PlantDoc: es una decisión del
  alumno y no se ha tocado tras ver los resultados.
- Segmentación más robusta a fondos complejos, varias hojas y hojas que llenan el encuadre.
- Un test con fotos propias y etiquetas fiables, con más sanas.
- Estudiar por qué las hojas sanas sobre fondo blanco se clasifican como enfermas.
