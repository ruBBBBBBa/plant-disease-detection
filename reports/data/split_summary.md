# Manifiesto y splits — resumen

Generado con `python scripts/prepare_dataset.py --seed 42 --fractions 0.7 0.15 0.15 --near-dup-threshold 2 --subset-healthy 500 --subset-diseased 500`. Decisiones: D006–D011 (`docs/DECISIONS.md`).

- Filas del manifiesto: **18906** (data/metadata.csv).
- Usadas: **13558**; excluidas: **5348**.

## Distribución original (antes de excluir)

| binary_label | disease_name | plantdoc | plantvillage | total |
|---|---|---|---|---|
| diseased | bacterial_spot | 109 | 2127 | 2236 |
| diseased | conflict:bacterial_spot|septoria_leaf_spot | 1 | 0 | 1 |
| diseased | early_blight | 88 | 1000 | 1088 |
| diseased | late_blight | 111 | 1909 | 2020 |
| diseased | leaf_mold | 91 | 952 | 1043 |
| diseased | mosaic_virus | 54 | 373 | 427 |
| diseased | septoria_leaf_spot | 151 | 1771 | 1922 |
| diseased | spider_mites | 2 | 1676 | 1678 |
| diseased | target_spot | 0 | 1404 | 1404 |
| diseased | yellow_leaf_curl_virus | 0 | 5357 | 5357 |
| diseased | yellow_virus | 76 | 0 | 76 |
| healthy | healthy | 63 | 1591 | 1654 |
| total |  | 746 | 18160 | 18906 |

## Excluidas por motivo

| source | disease_name | exact_duplicate_of | no_leaf_id | total |
|---|---|---|---|---|
| plantdoc | septoria_leaf_spot | 2 | 0 | 2 |
| plantdoc | yellow_virus | 1 | 0 | 1 |
| plantvillage | early_blight | 0 | 1 | 1 |
| plantvillage | healthy | 0 | 592 | 592 |
| plantvillage | late_blight | 0 | 989 | 989 |
| plantvillage | mosaic_virus | 0 | 373 | 373 |
| plantvillage | septoria_leaf_spot | 0 | 771 | 771 |
| plantvillage | yellow_leaf_curl_virus | 0 | 2619 | 2619 |
| total |  | 3 | 5345 | 5348 |

## Distribución usada por split

| split | images | groups |
|---|---|---|
| external_test | 743 | 743 |
| test | 1900 | 475 |
| train | 8987 | 2248 |
| val | 1928 | 482 |

| source | binary_label | disease_name | external_test | test | train | val | total |
|---|---|---|---|---|---|---|---|
| plantdoc | diseased | bacterial_spot | 109 | 0 | 0 | 0 | 109 |
| plantdoc | diseased | conflict:bacterial_spot|septoria_leaf_spot | 1 | 0 | 0 | 0 | 1 |
| plantdoc | diseased | early_blight | 88 | 0 | 0 | 0 | 88 |
| plantdoc | diseased | late_blight | 111 | 0 | 0 | 0 | 111 |
| plantdoc | diseased | leaf_mold | 91 | 0 | 0 | 0 | 91 |
| plantdoc | diseased | mosaic_virus | 54 | 0 | 0 | 0 | 54 |
| plantdoc | diseased | septoria_leaf_spot | 149 | 0 | 0 | 0 | 149 |
| plantdoc | diseased | spider_mites | 2 | 0 | 0 | 0 | 2 |
| plantdoc | diseased | yellow_virus | 75 | 0 | 0 | 0 | 75 |
| plantdoc | healthy | healthy | 63 | 0 | 0 | 0 | 63 |
| plantvillage | diseased | bacterial_spot | 0 | 316 | 1491 | 320 | 2127 |
| plantvillage | diseased | early_blight | 0 | 148 | 703 | 148 | 999 |
| plantvillage | diseased | late_blight | 0 | 136 | 644 | 140 | 920 |
| plantvillage | diseased | leaf_mold | 0 | 140 | 668 | 144 | 952 |
| plantvillage | diseased | septoria_leaf_spot | 0 | 148 | 700 | 152 | 1000 |
| plantvillage | diseased | spider_mites | 0 | 248 | 1176 | 252 | 1676 |
| plantvillage | diseased | target_spot | 0 | 208 | 984 | 212 | 1404 |
| plantvillage | diseased | yellow_leaf_curl_virus | 0 | 408 | 1918 | 412 | 2738 |
| plantvillage | healthy | healthy | 0 | 148 | 703 | 148 | 999 |
| total |  |  | 743 | 1900 | 8987 | 1928 | 13558 |

## Subconjunto inicial (`in_subset`)

| binary_label | disease_name | test | train | val | total |
|---|---|---|---|---|---|
| diseased | bacterial_spot | 8 | 44 | 8 | 60 |
| diseased | early_blight | 8 | 44 | 8 | 60 |
| diseased | late_blight | 8 | 44 | 8 | 60 |
| diseased | leaf_mold | 8 | 40 | 8 | 56 |
| diseased | septoria_leaf_spot | 8 | 40 | 8 | 56 |
| diseased | spider_mites | 8 | 44 | 8 | 60 |
| diseased | target_spot | 8 | 40 | 8 | 56 |
| diseased | yellow_leaf_curl_virus | 8 | 43 | 8 | 59 |
| healthy | healthy | 72 | 348 | 72 | 492 |
| total |  | 136 | 687 | 136 | 959 |

## Grupos, duplicados y comprobaciones

- Imágenes cuyo `leaf_id` es de otra clase que su carpeta: **0**.
- Target_Spot (`Com.G_TgS_FL nnnn` ↔ CSV `Com.G_FL nnnn`): 1404 ficheros, 1404 números distintos; 1404 números en el CSV; coinciden 1:1: **sí**.
- Duplicados exactos entre las imágenes usadas de PlantVillage: 0 filas.
- Casi-duplicados (dHash 64 bits, Hamming ≤ 2) entre imágenes usadas de PlantVillage: **0** pares.
- Grupos de hoja: 3205 antes de unir duplicados y 3205 después.
- Comprobación de fugas (`check_leakage`): **sin problemas**.

Rejilla de comprobación visual de Target_Spot: `target_spot_leaves.jpg` (una fila por hoja: `pv:Tomato___Target_Spot:::78.0`, `pv:Tomato___Target_Spot:::150.0`, `pv:Tomato___Target_Spot:::11.0`).
