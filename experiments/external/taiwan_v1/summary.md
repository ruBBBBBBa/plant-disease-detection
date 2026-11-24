# T08 — Segundo test externo: Taiwán (ejecución única)

Dataset: Huang, M.-L. y Chang, Y.-H. (2020), *Dataset of Tomato Leaves*, Mendeley Data, V1,
DOI 10.17632/ngdgg79rzb.1, licencia CC BY 4.0. Solo `Preprocessed data` (622 imágenes, todas
`external_test`; 0 casi-duplicados con PlantVillage, D051). Atribución de las rejillas en
`ATTRIBUTION.md`.

- **Commit congelado:** `c673c1e` (`execution_c673c1e67292d0b894c255ba3413d66bed6ee61d.json`, con los
  sha1 de configuración, anotaciones, modelos y predicciones). Protocolo: `configs/taiwan.json` (D055),
  normalización y «no concluyente» de `configs/external.json` (T06), sin cambios.
- **Anotación** (ciega a la clase, D053; commit `c8afde9`): 442 `leaflet`, 116 `compound`,
  64 `multiple`; 226 `simple` y 396 `complex`. **Ninguna sana tiene fondo `simple`**, de ahí que H7
  y H8 usen solo enfermas (D054).
- **Regenerar sin volver a predecir:** `python scripts/report_taiwan.py` (→ `tables.md`,
  `metrics.json`) y `python scripts/plot_taiwan_grids.py --variant B` (→ `grid_B_*.jpg`,
  `grid_index.csv`).

Tablas completas en [`tables.md`](tables.md). Lo esencial:

## Resultado global

| Variante | Cobertura | BA «todas» [IC95] | BA «solo concluyentes» [IC95] | Recall sanas «concl.» | Recall enfermas «concl.» |
|---|---:|---|---|---:|---:|
| A | 36,0 % | 0,129 [0,108; 0,153] | 0,349 [0,293; 0,418] | 0,077 | 0,621 |
| B | 36,0 % | 0,151 [0,127; 0,177] | 0,414 [0,350; 0,491] | 0,115 | 0,712 |
| B′ | 36,0 % | 0,122 [0,093; 0,154] | 0,374 [0,286; 0,476] | 0,308 | 0,439 |

- **398 de 622 imágenes son «no concluyentes»**: 15 con máscara vacía y el resto con la calidad de la
  máscara fuera de los rangos de PlantVillage (motivos solapables: área de hoja 282, contacto con el
  borde 142, solidez 119). B y B′ tienen la misma cobertura porque comparten máscara.
- Incluso entre las concluyentes, la BA de las tres variantes está **por debajo de 0,5**: de las 26
  sanas concluyentes, B solo acierta 3 (recall de sanas 0,115). Matiz:
  El sesgo de T07 se repite, pero depende de la condición: en `leaflet + simple`, B clasifica como
  sanas 41 de 124 enfermas concluyentes; con fondo complejo, casi todas las sanas salen enfermas (23
  de 26 sanas concluyentes). Lectura descriptiva (la mezcla de enfermedades cambia entre
  condiciones): las decisiones siguen las condiciones de captura más que los síntomas.
- BA de `val` en PlantVillage (referencia): A 0,959, B 0,946, B′ 0,963.

## Por condición `leaves × background` (ambas lecturas)

| Variante | Hojas | Fondo | Lectura | Total celda (sanas / enfermas) | n lectura (sanas / enfermas) | Cobertura | BA [IC95] | Recall sanas | Recall enfermas [IC95] | Matriz [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---:|---|---|---|---|
| A | compound | complex | todas | 94 (37 / 57) | 94 (37 / 57) | 17.0% | 0.061 [0.023; 0.107] | 0.000 | 0.123 [0.047; 0.214] | [[0, 37], [50, 7]] |
| A | compound | complex | solo concluyentes | 94 (37 / 57) | 16 (7 / 9) | 17.0% | 0.389 [0.250; 0.500] | 0.000 | 0.778 [0.500; 1.000] | [[0, 7], [2, 7]] |
| A | compound | simple | todas | 22 (0 / 22) | 22 (0 / 22) | 68.2% | no medible | no medible | 0.500 [0.317; 0.727] | [[0, 0], [11, 11]] |
| A | compound | simple | solo concluyentes | 22 (0 / 22) | 15 (0 / 15) | 68.2% | no medible | no medible | 0.733 [0.533; 0.933] | [[0, 0], [4, 11]] |
| A | leaflet | complex | todas | 244 (37 / 207) | 244 (37 / 207) | 24.2% | 0.096 [0.062; 0.135] | 0.027 | 0.164 [0.114; 0.215] | [[1, 36], [173, 34]] |
| A | leaflet | complex | solo concluyentes | 244 (37 / 207) | 59 (11 / 48) | 24.2% | 0.400 [0.302; 0.525] | 0.091 | 0.708 [0.574; 0.830] | [[1, 10], [14, 34]] |
| A | leaflet | simple | todas | 198 (0 / 198) | 198 (0 / 198) | 62.6% | no medible | no medible | 0.348 [0.288; 0.414] | [[0, 0], [129, 69]] |
| A | leaflet | simple | solo concluyentes | 198 (0 / 198) | 124 (0 / 124) | 62.6% | no medible | no medible | 0.556 [0.468; 0.645] | [[0, 0], [55, 69]] |
| A | multiple | complex | todas | 58 (32 / 26) | 58 (32 / 26) | 17.2% | 0.054 [0.000; 0.126] | 0.031 | 0.077 [0.000; 0.200] | [[1, 31], [24, 2]] |
| A | multiple | complex | solo concluyentes | 58 (32 / 26) | 10 (8 / 2) | 17.2% | 0.562 [0.500; 0.688] | 0.125 | 1.000 [1.000; 1.000] | [[1, 7], [0, 2]] |
| A | multiple | simple | todas | 6 (0 / 6) | 6 (0 / 6) | 0.0% | no medible | no medible | 0.000 [0.000; 0.000] | [[0, 0], [6, 0]] |
| A | multiple | simple | solo concluyentes | 6 (0 / 6) | 0 (0 / 0) | 0.0% | no medible | no medible | no medible | [[0, 0], [0, 0]] |
| B | compound | complex | todas | 94 (37 / 57) | 94 (37 / 57) | 17.0% | 0.053 [0.017; 0.096] | 0.000 | 0.105 [0.034; 0.191] | [[0, 37], [51, 6]] |
| B | compound | complex | solo concluyentes | 94 (37 / 57) | 16 (7 / 9) | 17.0% | 0.333 [0.167; 0.500] | 0.000 | 0.667 [0.333; 1.000] | [[0, 7], [3, 6]] |
| B | compound | simple | todas | 22 (0 / 22) | 22 (0 / 22) | 68.2% | no medible | no medible | 0.591 [0.364; 0.818] | [[0, 0], [9, 13]] |
| B | compound | simple | solo concluyentes | 22 (0 / 22) | 15 (0 / 15) | 68.2% | no medible | no medible | 0.867 [0.667; 1.000] | [[0, 0], [2, 13]] |
| B | leaflet | complex | todas | 244 (37 / 207) | 244 (37 / 207) | 24.2% | 0.116 [0.075; 0.166] | 0.054 | 0.179 [0.126; 0.230] | [[2, 35], [170, 37]] |
| B | leaflet | complex | solo concluyentes | 244 (37 / 207) | 59 (11 / 48) | 24.2% | 0.476 [0.353; 0.616] | 0.182 | 0.771 [0.646; 0.881] | [[2, 9], [11, 37]] |
| B | leaflet | simple | todas | 198 (0 / 198) | 198 (0 / 198) | 62.6% | no medible | no medible | 0.419 [0.348; 0.490] | [[0, 0], [115, 83]] |
| B | leaflet | simple | solo concluyentes | 198 (0 / 198) | 124 (0 / 124) | 62.6% | no medible | no medible | 0.669 [0.589; 0.750] | [[0, 0], [41, 83]] |
| B | multiple | complex | todas | 58 (32 / 26) | 58 (32 / 26) | 17.2% | 0.054 [0.000; 0.123] | 0.031 | 0.077 [0.000; 0.200] | [[1, 31], [24, 2]] |
| B | multiple | complex | solo concluyentes | 58 (32 / 26) | 10 (8 / 2) | 17.2% | 0.562 [0.500; 0.700] | 0.125 | 1.000 [1.000; 1.000] | [[1, 7], [0, 2]] |
| B | multiple | simple | todas | 6 (0 / 6) | 6 (0 / 6) | 0.0% | no medible | no medible | 0.000 [0.000; 0.000] | [[0, 0], [6, 0]] |
| B | multiple | simple | solo concluyentes | 6 (0 / 6) | 0 (0 / 0) | 0.0% | no medible | no medible | no medible | [[0, 0], [0, 0]] |
| B_prime | compound | complex | todas | 94 (37 / 57) | 94 (37 / 57) | 17.0% | 0.080 [0.031; 0.138] | 0.054 | 0.105 [0.034; 0.191] | [[2, 35], [51, 6]] |
| B_prime | compound | complex | solo concluyentes | 94 (37 / 57) | 16 (7 / 9) | 17.0% | 0.476 [0.250; 0.709] | 0.286 | 0.667 [0.333; 1.000] | [[2, 5], [3, 6]] |
| B_prime | compound | simple | todas | 22 (0 / 22) | 22 (0 / 22) | 68.2% | no medible | no medible | 0.318 [0.136; 0.545] | [[0, 0], [15, 7]] |
| B_prime | compound | simple | solo concluyentes | 22 (0 / 22) | 15 (0 / 15) | 68.2% | no medible | no medible | 0.467 [0.200; 0.733] | [[0, 0], [8, 7]] |
| B_prime | leaflet | complex | todas | 244 (37 / 207) | 244 (37 / 207) | 24.2% | 0.117 [0.065; 0.179] | 0.108 | 0.126 [0.081; 0.172] | [[4, 33], [181, 26]] |
| B_prime | leaflet | complex | solo concluyentes | 244 (37 / 207) | 59 (11 / 48) | 24.2% | 0.453 [0.302; 0.622] | 0.364 | 0.542 [0.400; 0.680] | [[4, 7], [22, 26]] |
| B_prime | leaflet | simple | todas | 198 (0 / 198) | 198 (0 / 198) | 62.6% | no medible | no medible | 0.232 [0.177; 0.293] | [[0, 0], [152, 46]] |
| B_prime | leaflet | simple | solo concluyentes | 198 (0 / 198) | 124 (0 / 124) | 62.6% | no medible | no medible | 0.371 [0.290; 0.452] | [[0, 0], [78, 46]] |
| B_prime | multiple | complex | todas | 58 (32 / 26) | 58 (32 / 26) | 17.2% | 0.070 [0.014; 0.150] | 0.062 | 0.077 [0.000; 0.200] | [[2, 30], [24, 2]] |
| B_prime | multiple | complex | solo concluyentes | 58 (32 / 26) | 10 (8 / 2) | 17.2% | 0.625 [0.500; 0.786] | 0.250 | 1.000 [1.000; 1.000] | [[2, 6], [0, 2]] |
| B_prime | multiple | simple | todas | 6 (0 / 6) | 6 (0 / 6) | 0.0% | no medible | no medible | 0.000 [0.000; 0.000] | [[0, 0], [6, 0]] |
| B_prime | multiple | simple | solo concluyentes | 6 (0 / 6) | 0 (0 / 0) | 0.0% | no medible | no medible | no medible | [[0, 0], [0, 0]] |

No medible: falta la clase necesaria en esa lectura. Cobertura = concluyentes / total de la celda; matrices y soportes corresponden a cada lectura.


## Hipótesis (fijadas antes de ejecutar)

| H | Contraste | Diferencia [IC95] | Veredicto |
|---|---|---|---|
| H7 | Cobertura de B en enfermas: `leaflet + simple` (0,626, n = 198) − resto (0,233, n = 318) | 0,394 [0,310; 0,477] | **apoyada** |
| H8 | Recall de enfermas de B, «todas»: `leaflet + simple` (0,419) − resto (0,182) | 0,237 [0,156; 0,319] | **apoyada** |
| H9 | BA «todas»: B′ − B (emparejado, 106 sanas / 516 enfermas) | −0,029 [−0,057; 0,002] | **no concluyente** |

Matices, sin cambiar los veredictos:

- **H7 y H8 no son independientes.** En la lectura «todas» una no concluyente cuenta como error, así
  que parte de la ventaja de H8 viene de la cobertura (H7).
- **La mezcla de enfermedades cambia entre condiciones** (D054). Aun así, la cobertura de
  `leaflet + simple` es mayor que la del resto en las cinco enfermedades por separado (descriptivo,
  `tables.md`).
- **H9:** B′ no mejora la BA. Sube algo el recall de sanas (+0,047 [0,000; 0,094], descriptivo) y baja
  el de enfermas (0,273 → 0,169 en «todas»). Mide «gray world con los umbrales de T04» (D052), no un
  sistema reajustado para color normalizado.

## Lectura de los resultados

- **H7** indica que la condición `leaflet + simple` aumenta la **analizabilidad** (cobertura), no que
  la clasificación sea correcta.
- **H8 depende de H7:** con un sistema sesgado hacia «diseased», el recall de enfermas sube con la
  cobertura. No es evidencia independiente de buena clasificación.
- **Limitación principal de T08:** ninguna sana tiene fondo `simple`, así que la especificidad en la
  condición válida (`leaflet + simple`) no se ha podido medir.
- **B en «solo concluyentes» (BA 0,414) está por debajo del azar** por el recall de sanas (3/26);
  el sesgo de T07 se repite, pero depende de la condición: en `leaflet + simple`, B clasifica como
  sanas 41 de 124 enfermas concluyentes; con fondo complejo, casi todas las sanas salen enfermas (23
  de 26 sanas concluyentes). Lectura descriptiva (la mezcla de enfermedades cambia entre
  condiciones): las decisiones siguen las condiciones de captura más que los síntomas.
- **H9:** no concluyente, con punto estimado negativo. Gray world con los umbrales de T04 no corrige
  el sesgo.
- **Autoría de la anotación:** las etiquetas de `data/annotations/taiwan_conditions.csv` son de
  Rubba; el commit `c8afde9` (hecho por el implementador) fue solo mecánico.

## Enfermedades no vistas (B)

| Enfermedad | n | Recall «todas» | Recall «solo concluyentes» (n) |
|---|---:|---:|---|
| black_mold | 67 | 0,358 | 0,706 (34) |
| gray_spot | 84 | 0,262 | 0,786 (28) |
| powdery_mildew | 157 | 0,191 | 0,698 (43) |
| *vistas:* bacterial_spot | 110 | 0,345 | 0,717 (53) |
| *vistas:* late_blight | 98 | 0,276 | 0,675 (40) |

Entre las concluyentes, las no vistas se detectan como «enfermas» en proporción parecida a las vistas;
con un sesgo tan fuerte hacia «enferma» (sanas mal clasificadas), eso no indica que se reconozcan.

## Comparación con PlantDoc (descriptiva)

| Variante | Fuente | Cobertura | BA «todas» | BA «solo concluyentes» |
|---|---|---:|---:|---:|
| A | PlantDoc | 22,6 % | 0,114 | 0,505 |
| A | Taiwán | 36,0 % | 0,129 | 0,349 |
| B | PlantDoc | 22,6 % | 0,115 | 0,508 |
| B | Taiwán | 36,0 % | 0,151 | 0,414 |

## Observaciones de las rejillas (cualitativas)

`grid_B_<hojas>_<fondo>.jpg`: hasta 8 errores y 8 aciertos por condición, elegidos con semilla 42.
- En la rejilla de `leaflet + simple` el fondo de las 16 imágenes es **negro** (el de PlantVillage
  es gris claro), y varias no concluyentes son hojas que llenan el encuadre o tocan el borde.
- En `complex`, la máscara de K-means se fragmenta y mezcla hoja, otras hojas y fondo.

Ninguna de estas observaciones se ha usado para ajustar nada; Taiwán queda usado.
