## Variantes preregistradas

| Variante | Lectura | n | Cobertura | BA [IC95] | Recall sanas | Recall enfermas | F1 sanas | F1 enfermas | ROC-AUC [IC95] |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| A | all | 743 | 22.6% | 0.114 [0.095; 0.138] | 0.016 [0.000; 0.055] | 0.212 [0.182; 0.243] | 0.003 | 0.325 | 0.036 [0.019; 0.054] |
| A | conclusive | 168 | 22.6% | 0.505 [0.459; 0.578] | 0.062 [0.000; 0.200] | 0.947 [0.908; 0.980] | 0.080 | 0.926 | 0.629 [0.495; 0.763] |
| B | all | 743 | 22.6% | 0.115 [0.095; 0.139] | 0.016 [0.000; 0.055] | 0.213 [0.182; 0.244] | 0.003 | 0.327 | 0.035 [0.019; 0.054] |
| B | conclusive | 168 | 22.6% | 0.508 [0.462; 0.580] | 0.062 [0.000; 0.200] | 0.954 [0.918; 0.982] | 0.083 | 0.929 | 0.622 [0.500; 0.742] |
| C | all | 743 | 22.6% | 0.111 [0.096; 0.127] | 0.000 [0.000; 0.000] | 0.222 [0.192; 0.254] | 0.000 | 0.338 | 0.041 [0.023; 0.061] |
| C | conclusive | 168 | 22.6% | 0.497 [0.490; 0.500] | 0.000 [0.000; 0.000] | 0.993 [0.980; 1.000] | 0.000 | 0.947 | 0.719 [0.614; 0.810] |
| C_lineal | all | 743 | 22.6% | 0.114 [0.095; 0.138] | 0.016 [0.000; 0.055] | 0.212 [0.180; 0.243] | 0.003 | 0.325 | 0.034 [0.018; 0.052] |
| C_lineal | conclusive | 168 | 22.6% | 0.505 [0.458; 0.577] | 0.062 [0.000; 0.200] | 0.947 [0.908; 0.980] | 0.080 | 0.926 | 0.600 [0.469; 0.730] |
| control | all | 743 | 22.6% | 0.112 [0.096; 0.128] | 0.000 [0.000; 0.000] | 0.224 [0.193; 0.256] | 0.000 | 0.340 | 0.030 [0.016; 0.047] |
| control | conclusive | 168 | 22.6% | 0.500 [0.500; 0.500] | 0.000 [0.000; 0.000] | 1.000 [1.000; 1.000] | 0.000 | 0.950 | 0.532 [0.377; 0.675] |

## Caídas Δ = BA(val PlantVillage) − BA(PlantDoc, lectura «todas»)

| Variante | BA val | BA PlantDoc | Δ [IC95] |
|---|---:|---:|---|
| A | 0.959 | 0.114 | 0.846 [0.822; 0.865] |
| B | 0.946 | 0.115 | 0.831 [0.807; 0.851] |
| C | 0.953 | 0.111 | 0.842 [0.826; 0.857] |
| C_lineal | 0.943 | 0.114 | 0.829 [0.805; 0.848] |
| control | 0.943 | 0.112 | 0.831 [0.815; 0.846] |

## Hipótesis (Δ_peor − Δ_mejor > 0; BA de val fija)

| H | Contraste | Δ_peor − Δ_mejor [IC95] | Veredicto |
|---|---|---|---|
| H1 | A − B | 0.014 [0.010; 0.019] | apoyada |
| H2 | B − C_lineal | 0.003 [0.001; 0.003] | apoyada |
| H3 | control − A | -0.015 [-0.026; 0.004] | no concluyente |

Control (descriptivo, sin veredicto): BA «todas» = 0.112 IC [0.096; 0.128]; incluye 0,5: False.

## Apéndice descriptivo: todas las combinaciones (BA «todas» / BA «solo concluyentes»)

| Grupo | Modelo | BA todas | BA concluyentes | Cobertura |
|---|---|---:|---:|---:|
| color | decision_tree | 0.153 | 0.658 | 22.6% |
| color | knn | 0.108 | 0.484 | 22.6% |
| color | random_forest | 0.112 | 0.500 | 22.6% |
| color | svm_lineal | 0.114 | 0.505 | 22.6% |
| color | svm_rbf | 0.112 | 0.500 | 22.6% |
| color_manchas | decision_tree | 0.156 | 0.666 | 22.6% |
| color_manchas | knn | 0.110 | 0.493 | 22.6% |
| color_manchas | random_forest | 0.112 | 0.500 | 22.6% |
| color_manchas | svm_lineal | 0.115 | 0.508 | 22.6% |
| color_manchas | svm_rbf | 0.111 | 0.497 | 22.6% |
| color_manchas_sin_l_v | decision_tree | 0.104 | 0.464 | 22.6% |
| color_manchas_sin_l_v | knn | 0.110 | 0.493 | 22.6% |
| color_manchas_sin_l_v | random_forest | 0.111 | 0.497 | 22.6% |
| color_manchas_sin_l_v | svm_lineal | 0.114 | 0.505 | 22.6% |
| color_manchas_sin_l_v | svm_rbf | 0.111 | 0.497 | 22.6% |
| combinadas | decision_tree | 0.104 | 0.464 | 22.6% |
| combinadas | knn | 0.111 | 0.497 | 22.6% |
| combinadas | random_forest | 0.112 | 0.500 | 22.6% |
| combinadas | svm_lineal | 0.114 | 0.505 | 22.6% |
| combinadas | svm_rbf | 0.112 | 0.500 | 22.6% |
| control_nitidez | decision_tree | 0.112 | 0.500 | 22.6% |
| control_nitidez | knn | 0.112 | 0.500 | 22.6% |
| control_nitidez | random_forest | 0.112 | 0.500 | 22.6% |
| control_nitidez | svm_lineal | 0.112 | 0.500 | 22.6% |
| control_nitidez | svm_rbf | 0.112 | 0.500 | 22.6% |
| manchas | decision_tree | 0.169 | 0.709 | 22.6% |
| manchas | knn | 0.107 | 0.475 | 22.6% |
| manchas | random_forest | 0.176 | 0.738 | 22.6% |
| manchas | svm_lineal | 0.170 | 0.715 | 22.6% |
| manchas | svm_rbf | 0.175 | 0.734 | 22.6% |
| textura | decision_tree | 0.112 | 0.500 | 22.6% |
| textura | knn | 0.112 | 0.500 | 22.6% |
| textura | random_forest | 0.112 | 0.500 | 22.6% |
| textura | svm_lineal | 0.108 | 0.484 | 22.6% |
| textura | svm_rbf | 0.112 | 0.500 | 22.6% |
| textura_suavizada | decision_tree | 0.133 | 0.562 | 22.6% |
| textura_suavizada | knn | 0.127 | 0.548 | 22.6% |
| textura_suavizada | random_forest | 0.110 | 0.488 | 22.6% |
| textura_suavizada | svm_lineal | 0.105 | 0.449 | 22.6% |
| textura_suavizada | svm_rbf | 0.120 | 0.530 | 22.6% |

## Casos especiales (variante A)

| Caso | n | No concluyentes | Recall «todas» | Recall «solo concluyentes» |
|---|---:|---:|---:|---:|
| mosaic_virus | 54 | 40 | 0.241 | 0.929 (n=14) |
| yellow_virus | 75 | 64 | 0.133 | 0.909 (n=11) |
| spider_mites | 2 | 2 | 0.000 | nan (n=0) |
| tomato_V8.jpg | 1 | 1 | 0.000 | nan (n=0) |

## Motivos de «no concluyente» en PlantDoc (% de las 743; pueden solaparse)

| Motivo | PlantDoc % | PlantVillage train+val % (referencia) |
|---|---:|---:|
| mask_empty | 0.8 | 0.0 |
| leaf_area_frac | 49.7 | 1.1 |
| border_touch_frac | 20.5 | 0.5 |
| mask_solidity | 39.9 | 1.1 |
| cualquiera | 77.4 | 2.3 |

## Diagnóstico de dominio (mediana; PlantVillage = train)

| Característica | PlantVillage | PlantDoc | Cociente |
|---|---:|---:|---:|
| sharpness_laplacian_var | 1.282e+04 | 921.2 | 0.07 |
| l_std | 18.21 | 13.64 | 0.75 |
| v_std | 0.1768 | 0.1421 | 0.80 |
| glcm_contrast | 23.77 | 4.573 | 0.19 |
| leaf_area_frac | 0.4278 | 0.255 | 0.60 |
| border_touch_frac | 0 | 0.06941 | — |
