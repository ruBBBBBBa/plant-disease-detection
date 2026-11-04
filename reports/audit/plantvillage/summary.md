# Auditoría — plantvillage (tomato)

Generado con `python scripts/audit_dataset.py --source plantvillage --species tomato --seed 42`. Raíz: `data/raw/plantvillage`. Suposición: carpeta = clase.

- Ficheros: **18160** en 10 clases.
- Ilegibles para OpenCV: **0**.

## Recuentos por clase

| class_name | original_split | files | unreadable |
|---|---|---|---|
| Tomato___Bacterial_spot |  | 2127 | 0 |
| Tomato___Early_blight |  | 1000 | 0 |
| Tomato___Late_blight |  | 1909 | 0 |
| Tomato___Leaf_Mold |  | 952 | 0 |
| Tomato___Septoria_leaf_spot |  | 1771 | 0 |
| Tomato___Spider_mites Two-spotted_spider_mite |  | 1676 | 0 |
| Tomato___Target_Spot |  | 1404 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus |  | 5357 | 0 |
| Tomato___Tomato_mosaic_virus |  | 373 | 0 |
| Tomato___healthy |  | 1591 | 0 |

## Extensiones y canales

| extension | count |
|---|---|
| .jpg | 18159 |
| .jpeg | 1 |

| channels | count |
|---|---|
| 3 | 18160 |

## Resoluciones (píxeles)

| stat | width | height |
|---|---|---|
| min | 256.0 | 256.0 |
| median | 256.0 | 256.0 |
| max | 256.0 | 256.0 |

Más frecuentes (ancho×alto):

| size | count |
|---|---|
| 256x256 | 18160 |

## Ficheros ilegibles

Ninguno.

## Duplicados exactos (mismo SHA-1)

- Grupos dentro de la misma clase: 14
- Grupos entre clases distintas: 0
- Ficheros implicados: 28

| files | scope | classes | paths |
|---|---|---|---|
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/9662364c-aaba-45e3-b907-10792d60578c___GH_HL Leaf 220.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/d2ce7896-6fa4-45e6-96c5-d162da0e3e1c___GH_HL Leaf 220.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/d6e6897a-5083-4914-9903-804c5684a956___GHLB2 Leaf 102.JPG | data/raw/plantvillage/raw/color/Tomato___Late_blight/d81682aa-746b-4e07-af2b-52ebb6f4c017___GHLB2 Leaf 102.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/5f21282c-e2ef-4c4a-ace1-b5701fe7effc___GHLB2 Leaf 8999.JPG | data/raw/plantvillage/raw/color/Tomato___Late_blight/e5d707cd-077c-43af-bda9-6138e516ff51___GHLB2 Leaf 8999.JPG |
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/37aad83b-7ff8-4b35-b3ed-fb8e0f54910b___GH_HL Leaf 342.1.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/e786ac89-29fe-47e3-b49e-b9a9ee7edd9d___GH_HL Leaf 342.1.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/3fae9c64-18f0-4a67-9f97-554248bb1bed___GHLB_PS Leaf 24 Day 16.jpg | data/raw/plantvillage/raw/color/Tomato___Late_blight/5de6da85-f8c4-48c4-b463-3e6bd78884cc___GHLB_PS Leaf 24 Day 16.jpg |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/48c55974-9fe9-4f4b-94f7-c8cd127d1e05___GHLB_PS Leaf 23.7 Day 13.jpg | data/raw/plantvillage/raw/color/Tomato___Late_blight/bd4f09bd-ee85-4ab1-bce0-8cde3fdd7f1b___GHLB_PS Leaf 23.7 Day 13.jpg |
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/068e324c-faf6-40d6-8f83-578907f1cac5___GH_HL Leaf 466.1.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/34c81c57-e1fa-49dd-a49d-34fe8b2385fe___GH_HL Leaf 466.1.JPG |
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/37203047-d8ba-43f7-b31e-d496c41c569c___GH_HL Leaf 389.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/505465db-407b-4e0a-8110-7479dad5261c___GH_HL Leaf 389.JPG |
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/488feb1c-4b9f-44e7-8aa6-4103a9601f5f___GH_HL Leaf 434.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/a5de43e7-fc2f-4a14-a8e6-c0f2f94c84f1___GH_HL Leaf 434.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/5688ea99-c949-41d0-bbab-9cbf0ffb8bcd___GHLB2 Leaf 8677.JPG | data/raw/plantvillage/raw/color/Tomato___Late_blight/e20107e7-137f-400e-817f-5dc4c58e5d70___GHLB2 Leaf 8677.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/1a69b38b-c4eb-42c4-9584-bcb14fb8db0c___GHLB2 Leaf 9011.JPG | data/raw/plantvillage/raw/color/Tomato___Late_blight/2c47b891-3c97-48f1-a2cc-5aa53d3a1148___GHLB2 Leaf 9011.JPG |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/98586693-fe1f-4ea0-8e27-4501b61cf09b___GHLB_PS Leaf 23.5 Day 13.jpg | data/raw/plantvillage/raw/color/Tomato___Late_blight/c1775bad-7c02-41fb-bb7d-f8df91d60ac3___GHLB_PS Leaf 23.5 Day 13.jpg |
| 2 | same_class | Tomato___Late_blight | data/raw/plantvillage/raw/color/Tomato___Late_blight/861d8b00-57a4-46b0-b51a-09d102dbe119___GHLB_PS Leaf 24 Day 13.jpg | data/raw/plantvillage/raw/color/Tomato___Late_blight/d82b634e-9a19-497b-b698-aaf100d622a1___GHLB_PS Leaf 24 Day 13.jpg |
| 2 | same_class | Tomato___healthy | data/raw/plantvillage/raw/color/Tomato___healthy/1af0bfe1-4bcf-4b8b-be66-5d0953eb647e___GH_HL Leaf 482.2.JPG | data/raw/plantvillage/raw/color/Tomato___healthy/cfd491d6-4af5-4728-8f0e-0d330a07174a___GH_HL Leaf 482.2.JPG |
## Emparejamiento color ↔ segmented (por nombre base)

| class_name | paired | only_color | only_segmented |
|---|---|---|---|
| Tomato___Bacterial_spot | 2127 | 0 | 0 |
| Tomato___Early_blight | 1000 | 0 | 0 |
| Tomato___Late_blight | 1909 | 0 | 0 |
| Tomato___Leaf_Mold | 952 | 0 | 0 |
| Tomato___Septoria_leaf_spot | 1771 | 0 | 0 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1675 | 1 | 1 |
| Tomato___Target_Spot | 1404 | 0 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 | 0 | 0 |
| Tomato___Tomato_mosaic_virus | 373 | 0 | 0 |
| Tomato___healthy | 1591 | 0 | 0 |

## Agrupación por hoja (`leaf-map.json`, lógica de `plant_village.py`)

`distinct_keys` = valores distintos de la parte del nombre tras `___`; `with_leaf_id` = imágenes encontradas en el mapa.

| class_name | images | distinct_keys | with_leaf_id | leaves | imgs_per_leaf_min | imgs_per_leaf_median | imgs_per_leaf_max |
|---|---|---|---|---|---|---|---|
| Tomato___Bacterial_spot | 2127 | 2127 | 2127 | 532 | 3 | 4.0 | 4 |
| Tomato___Early_blight | 1000 | 1000 | 999 | 250 | 3 | 4.0 | 4 |
| Tomato___Late_blight | 1909 | 1901 | 920 | 230 | 4 | 4.0 | 4 |
| Tomato___Leaf_Mold | 952 | 952 | 952 | 238 | 4 | 4.0 | 4 |
| Tomato___Septoria_leaf_spot | 1771 | 1771 | 1000 | 250 | 4 | 4.0 | 4 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1676 | 1676 | 1676 | 419 | 4 | 4.0 | 4 |
| Tomato___Target_Spot | 1404 | 1404 | 0 | 0 | 0 | 0.0 | 0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 | 5357 | 2738 | 685 | 3 | 4.0 | 4 |
| Tomato___Tomato_mosaic_virus | 373 | 373 | 0 | 0 | 0 | 0.0 | 0 |
| Tomato___healthy | 1591 | 1585 | 999 | 250 | 3 | 4.0 | 4 |

## Partición `data_distribution_for_SVM` frente a la agrupación por hoja

| train_images | train_with_leaf_id | train_leaves | test_images | test_with_leaf_id | test_leaves | leaves_in_both |
|---|---|---|---|---|---|---|
| 2853 | 1818 | 1329 | 3582 | 2263 | 566 | 0 |

