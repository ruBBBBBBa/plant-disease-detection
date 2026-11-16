# Características — análisis exploratorio (solo train)

`python scripts/analyze_features.py --seed 42` sobre `data/processed/features_v1_subset.csv`. Solo filas `train` del subconjunto (1392 imágenes; 14 «no concluyentes» excluidas). `val` y `test` no se leen. AUC: 1 = enferma; `sep` = |AUC − 0,5| (0 = no separa, 0,5 = separa del todo).

Características constantes en train (sin información, excluidas del análisis): ninguna.

## Resumen por familia

| family | features | median_sep | max_sep |
|---|---|---|---|
| textura | 22 | 0.397 | 0.487 |
| manchas | 7 | 0.214 | 0.246 |
| color | 43 | 0.193 | 0.448 |
| calidad | 3 | 0.07 | 0.124 |

## AUC univariante de cada característica (ordenado por separación)

| feature | family | auc | sep | median_healthy | median_diseased |
|---|---|---|---|---|---|
| glcm_contrast | textura | 0.013 | 0.487 | 35.4679 | 10.2767 |
| glcm_homogeneity | textura | 0.96 | 0.46 | 0.2281 | 0.3853 |
| lbp_02 | textura | 0.951 | 0.451 | 0.0239 | 0.0301 |
| l_std | color | 0.052 | 0.448 | 20.5883 | 14.0143 |
| v_std | color | 0.052 | 0.448 | 0.2001 | 0.1353 |
| lbp_03 | textura | 0.942 | 0.442 | 0.0114 | 0.0188 |
| glcm_energy | textura | 0.932 | 0.432 | 0.0553 | 0.0933 |
| lbp_04 | textura | 0.923 | 0.423 | 0.008 | 0.0146 |
| lbp_15 | textura | 0.077 | 0.423 | 0.0689 | 0.056 |
| lbp_13 | textura | 0.913 | 0.413 | 0.0139 | 0.0204 |
| lbp_05 | textura | 0.909 | 0.409 | 0.0071 | 0.014 |
| lbp_14 | textura | 0.908 | 0.408 | 0.0279 | 0.033 |
| lbp_16 | textura | 0.098 | 0.402 | 0.1073 | 0.0925 |
| lbp_06 | textura | 0.892 | 0.392 | 0.0075 | 0.016 |
| lbp_12 | textura | 0.89 | 0.39 | 0.0101 | 0.0161 |
| lbp_01 | textura | 0.111 | 0.389 | 0.0638 | 0.0538 |
| a_hist_04 | color | 0.117 | 0.383 | 0.3698 | 0.1056 |
| lbp_17 | textura | 0.117 | 0.383 | 0.4988 | 0.4453 |
| lbp_00 | textura | 0.125 | 0.375 | 0.0999 | 0.088 |
| lbp_07 | textura | 0.875 | 0.375 | 0.0092 | 0.0195 |
| lbp_10 | textura | 0.873 | 0.373 | 0.0096 | 0.0173 |
| lbp_11 | textura | 0.862 | 0.362 | 0.0096 | 0.0157 |
| lbp_08 | textura | 0.858 | 0.358 | 0.0109 | 0.0232 |
| a_hist_01 | color | 0.855 | 0.355 | 0.0045 | 0.1096 |
| lbp_09 | textura | 0.846 | 0.346 | 0.0111 | 0.0213 |
| glcm_correlation | textura | 0.833 | 0.333 | 0.5483 | 0.7095 |
| a_hist_02 | color | 0.812 | 0.312 | 0.0502 | 0.2402 |
| s_mean | color | 0.802 | 0.302 | 0.3242 | 0.4302 |
| a_mean | color | 0.2 | 0.3 | -10.2862 | -14.1764 |
| a_hist_05 | color | 0.204 | 0.296 | 0.118 | 0.0508 |
| frac_dark | color | 0.206 | 0.294 | 0.0056 | 0.0011 |
| a_hist_00 | color | 0.79 | 0.29 | 0.0001 | 0.0096 |
| h_hist_02 | color | 0.223 | 0.277 | 0.5708 | 0.2602 |
| a_std | color | 0.772 | 0.272 | 3.6957 | 5.1011 |
| spot_area_frac | manchas | 0.746 | 0.246 | 0.0007 | 0.0073 |
| spot_max_area_frac | manchas | 0.743 | 0.243 | 0.0006 | 0.0031 |
| spot_mean_area_frac | manchas | 0.741 | 0.241 | 0.0005 | 0.0016 |
| s_std | color | 0.261 | 0.239 | 0.2 | 0.1681 |
| b_hist_02 | color | 0.262 | 0.238 | 0.3627 | 0.2308 |
| frac_yellow | color | 0.73 | 0.23 | 0.004 | 0.0133 |
| l_mean | color | 0.275 | 0.225 | 45.3445 | 39.6552 |
| spot_count | manchas | 0.714 | 0.214 | 1.0 | 4.0 |
| h_circ_var | color | 0.704 | 0.204 | 0.0201 | 0.0408 |
| h_hist_04 | color | 0.704 | 0.204 | 0.0028 | 0.0349 |
| v_mean | color | 0.299 | 0.201 | 0.4465 | 0.3956 |
| b_hist_05 | color | 0.697 | 0.197 | 0.0064 | 0.0344 |
| b_std | color | 0.696 | 0.196 | 5.2648 | 6.4938 |
| a_hist_06 | color | 0.307 | 0.193 | 0.047 | 0.0225 |
| b_mean | color | 0.688 | 0.188 | 14.266 | 18.5832 |
| b_hist_04 | color | 0.681 | 0.181 | 0.0264 | 0.093 |
| h_circ_mean_deg | color | 0.668 | 0.168 | 85.3415 | 92.788 |
| frac_brown | color | 0.667 | 0.167 | 0.0002 | 0.0011 |
| b_hist_06 | color | 0.66 | 0.16 | 0.0012 | 0.0057 |
| b_hist_01 | color | 0.345 | 0.155 | 0.3032 | 0.1634 |
| b_hist_07 | color | 0.642 | 0.142 | 0.0 | 0.0007 |
| leaf_area_frac | calidad | 0.376 | 0.124 | 0.4397 | 0.4073 |
| b_hist_00 | color | 0.409 | 0.091 | 0.0459 | 0.0256 |
| h_hist_08 | color | 0.409 | 0.091 | 0.0 | 0.0 |
| h_hist_09 | color | 0.414 | 0.086 | 0.0 | 0.0 |
| spot_mean_solidity | manchas | 0.575 | 0.075 | 0.8065 | 0.8394 |
| h_hist_05 | color | 0.574 | 0.074 | 0.0 | 0.0001 |
| spot_mean_eccentricity | manchas | 0.573 | 0.073 | 0.7409 | 0.7928 |
| h_hist_07 | color | 0.428 | 0.072 | 0.0 | 0.0 |
| mask_solidity | calidad | 0.43 | 0.07 | 0.88 | 0.8683 |
| h_hist_11 | color | 0.434 | 0.066 | 0.0 | 0.0 |
| h_hist_10 | color | 0.437 | 0.063 | 0.0 | 0.0 |
| a_hist_03 | color | 0.441 | 0.059 | 0.2874 | 0.2225 |
| h_hist_03 | color | 0.552 | 0.052 | 0.3832 | 0.4364 |
| h_hist_01 | color | 0.544 | 0.044 | 0.0234 | 0.0272 |
| h_hist_00 | color | 0.461 | 0.039 | 0.0005 | 0.0003 |
| h_hist_06 | color | 0.468 | 0.032 | 0.0 | 0.0 |
| b_hist_03 | color | 0.529 | 0.029 | 0.1415 | 0.175 |
| a_hist_07 | color | 0.519 | 0.019 | 0.0027 | 0.004 |
| border_touch_frac | calidad | 0.485 | 0.015 | 0.0 | 0.0 |
| spot_mean_circularity | manchas | 0.492 | 0.008 | 0.8006 | 0.7536 |

## Pares muy correlacionados (|r| ≥ 0.9)

104 pares. Apariciones por familia: {'textura': 190, 'color': 11, 'manchas': 7}.

| a | b | family_a | family_b | r |
|---|---|---|---|---|
| v_std | l_std | color | color | 0.998 |
| v_mean | l_mean | color | color | 0.995 |
| lbp_07 | lbp_08 | textura | textura | 0.989 |
| lbp_07 | lbp_09 | textura | textura | 0.985 |
| lbp_08 | lbp_09 | textura | textura | 0.985 |
| lbp_05 | lbp_06 | textura | textura | 0.983 |
| lbp_06 | lbp_07 | textura | textura | 0.982 |
| lbp_04 | lbp_05 | textura | textura | 0.979 |
| lbp_06 | lbp_17 | textura | textura | -0.977 |
| lbp_06 | lbp_10 | textura | textura | 0.976 |
| lbp_05 | lbp_17 | textura | textura | -0.973 |
| lbp_09 | lbp_10 | textura | textura | 0.973 |
| lbp_07 | lbp_17 | textura | textura | -0.972 |
| lbp_06 | lbp_09 | textura | textura | 0.97 |
| lbp_09 | lbp_17 | textura | textura | -0.969 |
| lbp_07 | lbp_10 | textura | textura | 0.968 |
| spot_mean_solidity | spot_mean_circularity | manchas | manchas | 0.968 |
| lbp_10 | lbp_17 | textura | textura | -0.966 |
| h_hist_01 | spot_area_frac | color | manchas | 0.965 |
| lbp_00 | lbp_06 | textura | textura | -0.965 |
| lbp_03 | lbp_04 | textura | textura | 0.965 |
| lbp_04 | lbp_06 | textura | textura | 0.965 |
| lbp_05 | lbp_10 | textura | textura | 0.965 |
| lbp_10 | lbp_11 | textura | textura | 0.964 |
| lbp_06 | lbp_08 | textura | textura | 0.964 |
| lbp_04 | lbp_17 | textura | textura | -0.964 |
| lbp_00 | lbp_09 | textura | textura | -0.963 |
| lbp_11 | lbp_12 | textura | textura | 0.962 |
| lbp_00 | lbp_07 | textura | textura | -0.962 |
| lbp_05 | lbp_07 | textura | textura | 0.961 |
| lbp_08 | lbp_17 | textura | textura | -0.959 |
| lbp_12 | lbp_13 | textura | textura | 0.958 |
| spot_area_frac | spot_max_area_frac | manchas | manchas | 0.957 |
| lbp_00 | lbp_10 | textura | textura | -0.957 |
| lbp_08 | lbp_10 | textura | textura | 0.956 |
| lbp_05 | lbp_15 | textura | textura | -0.954 |
| lbp_01 | lbp_06 | textura | textura | -0.953 |
| lbp_00 | lbp_05 | textura | textura | -0.953 |
| lbp_01 | lbp_05 | textura | textura | -0.953 |
| lbp_04 | lbp_15 | textura | textura | -0.953 |
| lbp_11 | lbp_17 | textura | textura | -0.951 |
| lbp_10 | lbp_12 | textura | textura | 0.951 |
| lbp_12 | lbp_15 | textura | textura | -0.95 |
| lbp_15 | lbp_17 | textura | textura | 0.95 |
| lbp_00 | lbp_17 | textura | textura | 0.949 |
| lbp_01 | lbp_10 | textura | textura | -0.949 |
| lbp_00 | lbp_08 | textura | textura | -0.949 |
| lbp_10 | lbp_15 | textura | textura | -0.948 |
| lbp_06 | lbp_15 | textura | textura | -0.948 |
| lbp_03 | lbp_05 | textura | textura | 0.947 |
| lbp_00 | lbp_01 | textura | textura | 0.947 |
| lbp_01 | lbp_17 | textura | textura | 0.947 |
| lbp_05 | lbp_09 | textura | textura | 0.946 |
| lbp_01 | lbp_04 | textura | textura | -0.945 |
| lbp_09 | lbp_11 | textura | textura | 0.944 |
| lbp_04 | lbp_10 | textura | textura | 0.944 |
| lbp_11 | lbp_15 | textura | textura | -0.944 |
| lbp_12 | lbp_17 | textura | textura | -0.944 |
| lbp_13 | lbp_15 | textura | textura | -0.943 |
| lbp_05 | lbp_12 | textura | textura | 0.941 |
| lbp_06 | lbp_11 | textura | textura | 0.941 |
| lbp_03 | lbp_13 | textura | textura | 0.941 |
| lbp_04 | lbp_12 | textura | textura | 0.94 |
| lbp_03 | lbp_17 | textura | textura | -0.939 |
| lbp_05 | lbp_11 | textura | textura | 0.939 |
| lbp_04 | lbp_13 | textura | textura | 0.938 |
| lbp_13 | lbp_17 | textura | textura | -0.938 |
| lbp_01 | lbp_11 | textura | textura | -0.937 |
| lbp_05 | lbp_08 | textura | textura | 0.937 |
| lbp_01 | lbp_07 | textura | textura | -0.936 |
| glcm_homogeneity | glcm_energy | textura | textura | 0.936 |
| lbp_03 | lbp_15 | textura | textura | -0.936 |
| lbp_04 | lbp_07 | textura | textura | 0.936 |
| spot_mean_solidity | spot_mean_eccentricity | manchas | manchas | 0.936 |
| lbp_00 | lbp_11 | textura | textura | -0.936 |
| lbp_01 | lbp_09 | textura | textura | -0.934 |
| lbp_00 | lbp_04 | textura | textura | -0.933 |
| lbp_01 | lbp_15 | textura | textura | 0.932 |
| lbp_06 | lbp_12 | textura | textura | 0.93 |
| lbp_11 | lbp_13 | textura | textura | 0.927 |
| lbp_01 | lbp_12 | textura | textura | -0.927 |
| lbp_05 | lbp_13 | textura | textura | 0.925 |
| a_hist_07 | frac_brown | color | color | 0.924 |
| lbp_07 | lbp_15 | textura | textura | -0.924 |
| lbp_09 | lbp_15 | textura | textura | -0.923 |
| lbp_04 | lbp_11 | textura | textura | 0.923 |
| lbp_07 | lbp_11 | textura | textura | 0.923 |
| lbp_03 | lbp_06 | textura | textura | 0.922 |
| lbp_00 | lbp_15 | textura | textura | 0.92 |
| h_hist_01 | frac_yellow | color | color | 0.919 |
| lbp_01 | lbp_03 | textura | textura | -0.918 |
| lbp_04 | lbp_09 | textura | textura | 0.916 |
| h_hist_09 | h_hist_10 | color | color | 0.915 |
| lbp_01 | lbp_08 | textura | textura | -0.915 |
| lbp_00 | lbp_12 | textura | textura | -0.913 |
| lbp_03 | lbp_12 | textura | textura | 0.913 |
| lbp_10 | lbp_13 | textura | textura | 0.913 |
| lbp_01 | lbp_13 | textura | textura | -0.911 |
| lbp_04 | lbp_08 | textura | textura | 0.909 |
| lbp_09 | lbp_12 | textura | textura | 0.908 |
| lbp_08 | lbp_11 | textura | textura | 0.906 |
| lbp_06 | lbp_13 | textura | textura | 0.905 |
| glcm_homogeneity | lbp_03 | textura | textura | 0.904 |
| lbp_08 | lbp_15 | textura | textura | -0.903 |

## Conclusión factual (train)

- Familia con mayor separación mediana: **textura** (0.397); la menor: **calidad** (0.07).
- Característica más separadora (sin contar calidad): `glcm_contrast` (AUC 0.013).
- Las características de **calidad** de la máscara no son síntomas; si separan, delatan diferencias de captura entre clases (encuadre, tamaño de la hoja): `leaf_area_frac` AUC 0.376, `mask_solidity` AUC 0.43, `border_touch_frac` AUC 0.485.
- **Diagnóstico de captura** (nitidez = varianza del laplaciano dentro de la hoja, no es una característica): AUC sano/enfermo 0.014. Mediana por enfermedad: bacterial_spot 951.8, early_blight 3073.7, healthy 20111.9, late_blight 5814.1, leaf_mold 3847.1, septoria_leaf_spot 5895.9, spider_mites 8493.8, target_spot 9310.8, yellow_leaf_curl_virus 3925.8. Si las sanas son mucho más nítidas, la separación de la textura puede deberse en parte a la captura de cada lote y no a los síntomas.
- Separación univariante no es capacidad de clasificación: no se anticipa ningún resultado de la fase 4.

Figuras: `boxplots_train.png` (10 más separadoras), `spots_train.jpg` (3 hojas por enfermedad: contorno de la hoja en verde, regiones candidatas en magenta; las regiones candidatas son evidencia visual de tonos anómalos, no la zona que usa el clasificador).
