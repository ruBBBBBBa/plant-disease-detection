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
  sanas concluyentes, B solo acierta 3 (recall de sanas 0,115). Es el mismo patrón que en T07: casi
  todo lo sano se llama «enfermo».
- BA de `val` en PlantVillage (referencia): A 0,959, B 0,946, B′ 0,963.

## Por condición (B, lectura «todas»)

| Hojas | Fondo | n (sanas / enfermas) | Cobertura | Recall enfermas [IC95] |
|---|---|---|---:|---|
| leaflet | simple | 198 (0 / 198) | 62,6 % | 0,419 [0,348; 0,490] |
| compound | simple | 22 (0 / 22) | 68,2 % | 0,591 [0,364; 0,818] |
| multiple | simple | 6 (0 / 6) | 0,0 % | 0,000 |
| leaflet | complex | 244 (37 / 207) | 24,2 % | 0,179 [0,126; 0,230] |
| compound | complex | 94 (37 / 57) | 17,0 % | 0,105 [0,034; 0,191] |
| multiple | complex | 58 (32 / 26) | 17,2 % | 0,077 [0,000; 0,200] |

La BA solo existe en las celdas `complex` (las únicas con sanas) y en todas ellas es ≤ 0,116.

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
