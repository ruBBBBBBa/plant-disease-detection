# Taiwán — manifiesto del test externo (T08, fase A)

Generado con `python scripts/prepare_taiwan.py --near-dup-threshold 2`. Dataset: Huang, M.-L. y Chang, Y.-H. (2020), *Dataset of Tomato Leaves*, Mendeley Data, V1, DOI 10.17632/ngdgg79rzb.1, licencia CC BY 4.0. Solo `Preprocessed data`; `data augmentation` no se lee.

- Imágenes: **622**; `external_test`: **622**; excluidas: **0**.
- Casi-duplicados frente a PlantVillage (dHash, Hamming ≤ 2, contra 18160 imágenes de PlantVillage): **0** imágenes de Taiwán, en 0 pares.
- Distancia mínima de cada imagen de Taiwán a PlantVillage (descriptivo): ≤ 2: 0, 3–5: 0, 6–10: 22, > 10: 600 (mínimo 7).
- Pares casi duplicados **dentro** de Taiwán (mismo umbral; solo informativo, no se excluyen): **0**.

## Recuento por carpeta (y si la enfermedad se vio al entrenar)

| folder | disease_name | seen_in_training | external_test | All |
|---|---|---|---|---|
| Bacterial spot | bacterial_spot | True | 110 | 110 |
| Black mold | black_mold | False | 67 | 67 |
| Gray spot | gray_spot | False | 84 | 84 |
| Late blight | late_blight | True | 98 | 98 |
| health | healthy | True | 106 | 106 |
| powdery mildew | powdery_mildew | False | 157 | 157 |
| All |  |  | 622 | 622 |

## Exclusiones

Ninguna.

## Pares casi duplicados con PlantVillage

Ninguno.
