## Variantes en todo Taiwán

| Variante | Subconjunto | Lectura | n | Cobertura | BA [IC95] | Recall sanas [IC95] | Recall enfermas [IC95] | Matriz [[TN, FP], [FN, TP]] |
|---|---|---|---:|---:|---|---|---|---|
| A | todas | all | 622 | 36.0% | 0.129 [0.108; 0.153] | 0.019 [0.000; 0.047] | 0.238 [0.203; 0.278] | [[2, 104], [393, 123]] |
| A | todas | conclusive | 224 | 36.0% | 0.349 [0.293; 0.418] | 0.077 [0.000; 0.192] | 0.621 [0.555; 0.688] | [[2, 24], [75, 123]] |
| A | vistas (sanas + vistas) | all | 314 | 37.9% | 0.151 [0.121; 0.183] | 0.019 [0.000; 0.048] | 0.284 [0.226; 0.343] | [[2, 104], [149, 59]] |
| A | vistas (sanas + vistas) | conclusive | 119 | 37.9% | 0.356 [0.288; 0.431] | 0.077 [0.000; 0.192] | 0.634 [0.532; 0.733] | [[2, 24], [34, 59]] |
| B | todas | all | 622 | 36.0% | 0.151 [0.127; 0.177] | 0.028 [0.000; 0.063] | 0.273 [0.236; 0.313] | [[3, 103], [375, 141]] |
| B | todas | conclusive | 224 | 36.0% | 0.414 [0.350; 0.491] | 0.115 [0.000; 0.250] | 0.712 [0.650; 0.773] | [[3, 23], [57, 141]] |
| B | vistas (sanas + vistas) | all | 314 | 37.9% | 0.170 [0.138; 0.206] | 0.028 [0.000; 0.064] | 0.312 [0.251; 0.376] | [[3, 103], [143, 65]] |
| B | vistas (sanas + vistas) | conclusive | 119 | 37.9% | 0.407 [0.334; 0.486] | 0.115 [0.000; 0.250] | 0.699 [0.600; 0.789] | [[3, 23], [28, 65]] |
| B_prime | todas | all | 622 | 36.0% | 0.122 [0.093; 0.154] | 0.075 [0.029; 0.128] | 0.169 [0.136; 0.200] | [[8, 98], [429, 87]] |
| B_prime | todas | conclusive | 224 | 36.0% | 0.374 [0.286; 0.476] | 0.308 [0.148; 0.500] | 0.439 [0.372; 0.510] | [[8, 18], [111, 87]] |
| B_prime | vistas (sanas + vistas) | all | 314 | 37.9% | 0.148 [0.114; 0.187] | 0.075 [0.029; 0.129] | 0.221 [0.166; 0.278] | [[8, 98], [162, 46]] |
| B_prime | vistas (sanas + vistas) | conclusive | 119 | 37.9% | 0.401 [0.302; 0.510] | 0.308 [0.130; 0.486] | 0.495 [0.393; 0.596] | [[8, 18], [47, 46]] |

## Recall por enfermedad (no vistas: black_mold, gray_spot, powdery_mildew)

| Variante | Enfermedad | Vista | n | Recall «todas» | Recall «solo concluyentes» (n) |
|---|---|---|---:|---:|---|
| A | bacterial_spot | sí | 110 | 0.291 | 0.604 (53) |
| A | black_mold | no | 67 | 0.194 | 0.382 (34) |
| A | gray_spot | no | 84 | 0.262 | 0.786 (28) |
| A | healthy | sí | 106 | 0.019 | 0.077 (26) |
| A | late_blight | sí | 98 | 0.276 | 0.675 (40) |
| A | powdery_mildew | no | 157 | 0.185 | 0.674 (43) |
| B | bacterial_spot | sí | 110 | 0.345 | 0.717 (53) |
| B | black_mold | no | 67 | 0.358 | 0.706 (34) |
| B | gray_spot | no | 84 | 0.262 | 0.786 (28) |
| B | healthy | sí | 106 | 0.028 | 0.115 (26) |
| B | late_blight | sí | 98 | 0.276 | 0.675 (40) |
| B | powdery_mildew | no | 157 | 0.191 | 0.698 (43) |
| B_prime | bacterial_spot | sí | 110 | 0.191 | 0.396 (53) |
| B_prime | black_mold | no | 67 | 0.149 | 0.294 (34) |
| B_prime | gray_spot | no | 84 | 0.202 | 0.607 (28) |
| B_prime | healthy | sí | 106 | 0.075 | 0.308 (26) |
| B_prime | late_blight | sí | 98 | 0.255 | 0.625 (40) |
| B_prime | powdery_mildew | no | 157 | 0.089 | 0.326 (43) |

## Por condición `leaves × background` (lectura «todas»)

| Variante | Hojas | Fondo | n (sanas / enfermas) | Cobertura | BA [IC95] | Recall sanas | Recall enfermas [IC95] |
|---|---|---|---|---:|---|---:|---|
| A | compound | complex | 94 (37 / 57) | 17.0% | 0.061 [0.023; 0.107] | 0.000 | 0.123 [0.047; 0.214] |
| A | compound | simple | 22 (0 / 22) | 68.2% | — | — | 0.500 [0.317; 0.727] |
| A | leaflet | complex | 244 (37 / 207) | 24.2% | 0.096 [0.062; 0.135] | 0.027 | 0.164 [0.114; 0.215] |
| A | leaflet | simple | 198 (0 / 198) | 62.6% | — | — | 0.348 [0.288; 0.414] |
| A | multiple | complex | 58 (32 / 26) | 17.2% | 0.054 [0.000; 0.126] | 0.031 | 0.077 [0.000; 0.200] |
| A | multiple | simple | 6 (0 / 6) | 0.0% | — | — | 0.000 [0.000; 0.000] |
| B | compound | complex | 94 (37 / 57) | 17.0% | 0.053 [0.017; 0.096] | 0.000 | 0.105 [0.034; 0.191] |
| B | compound | simple | 22 (0 / 22) | 68.2% | — | — | 0.591 [0.364; 0.818] |
| B | leaflet | complex | 244 (37 / 207) | 24.2% | 0.116 [0.075; 0.166] | 0.054 | 0.179 [0.126; 0.230] |
| B | leaflet | simple | 198 (0 / 198) | 62.6% | — | — | 0.419 [0.348; 0.490] |
| B | multiple | complex | 58 (32 / 26) | 17.2% | 0.054 [0.000; 0.123] | 0.031 | 0.077 [0.000; 0.200] |
| B | multiple | simple | 6 (0 / 6) | 0.0% | — | — | 0.000 [0.000; 0.000] |
| B_prime | compound | complex | 94 (37 / 57) | 17.0% | 0.080 [0.031; 0.138] | 0.054 | 0.105 [0.034; 0.191] |
| B_prime | compound | simple | 22 (0 / 22) | 68.2% | — | — | 0.318 [0.136; 0.545] |
| B_prime | leaflet | complex | 244 (37 / 207) | 24.2% | 0.117 [0.065; 0.179] | 0.108 | 0.126 [0.081; 0.172] |
| B_prime | leaflet | simple | 198 (0 / 198) | 62.6% | — | — | 0.232 [0.177; 0.293] |
| B_prime | multiple | complex | 58 (32 / 26) | 17.2% | 0.070 [0.014; 0.150] | 0.062 | 0.077 [0.000; 0.200] |
| B_prime | multiple | simple | 6 (0 / 6) | 0.0% | — | — | 0.000 [0.000; 0.000] |

## Hipótesis (fijadas antes de ejecutar, `configs/taiwan.json`)

| H | Contraste | Grupos (n) | Diferencia [IC95] | Veredicto |
|---|---|---|---|---|
| H7 | coverage de B (enfermas): leaflet + simple − resto | 198 / 318 (0.626 / 0.233) | 0.394 [0.310; 0.477] | apoyada |
| H8 | recall_diseased_all de B (enfermas): leaflet + simple − resto | 198 / 318 (0.419 / 0.182) | 0.237 [0.156; 0.319] | apoyada |
| H9 | BA «todas»: B′ − B (emparejado) | 106 sanas / 516 enfermas | -0.029 [-0.057; 0.002] | no concluyente |

## Descriptivo, sin veredicto

- Recall de sanas, B′ − B (lectura «todas», emparejado): 0.047 [0.000; 0.094].
- `compound + simple` (B, enfermas, n = 22): cobertura 0.682, recall de enfermas «todas» 0.591.

H7 y H8 por enfermedad (B, enfermas; la mezcla cambia entre condiciones, D054):

| Enfermedad | n foco / resto | Cobertura foco / resto | Recall «todas» foco / resto |
|---|---|---|---|
| bacterial_spot | 70 / 40 | 0.629 / 0.225 | 0.414 / 0.225 |
| black_mold | 54 / 13 | 0.574 / 0.231 | 0.407 / 0.154 |
| gray_spot | 20 / 64 | 0.550 / 0.266 | 0.300 / 0.250 |
| late_blight | 30 / 68 | 0.633 / 0.309 | 0.333 / 0.250 |
| powdery_mildew | 24 / 133 | 0.792 / 0.180 | 0.667 / 0.105 |

H9 por condición (BA «todas», B′ − B; sin IC):

| Hojas | Fondo | n | BA B | BA B′ | Recall sanas B / B′ | Recall enfermas B / B′ |
|---|---|---:|---:|---:|---|---|
| compound | complex | 94 | 0.053 | 0.080 | 0.000 / 0.054 | 0.105 / 0.105 |
| compound | simple | 22 | — | — | — / — | 0.591 / 0.318 |
| leaflet | complex | 244 | 0.116 | 0.117 | 0.054 / 0.108 | 0.179 / 0.126 |
| leaflet | simple | 198 | — | — | — / — | 0.419 / 0.232 |
| multiple | complex | 58 | 0.054 | 0.070 | 0.031 / 0.062 | 0.077 / 0.077 |
| multiple | simple | 6 | — | — | — / — | 0.000 / 0.000 |

Comparación con PlantDoc (T06; otro dataset, otras clases):

| Variante | Fuente | Cobertura | BA «todas» | BA «solo concluyentes» |
|---|---|---:|---:|---:|
| A | PlantDoc | 22.6% | 0.114 | 0.505 |
| A | Taiwán | 36.0% | 0.129 | 0.349 |
| B | PlantDoc | 22.6% | 0.115 | 0.508 |
| B | Taiwán | 36.0% | 0.151 | 0.414 |

BA de val de PlantVillage (referencia): A 0.959, B 0.946, B_prime 0.963.

Motivos de «no concluyente» (imágenes de 622; pueden solaparse): mask_empty 15, leaf_area_frac 282, border_touch_frac 142, mask_solidity 119, cualquiera 398.

