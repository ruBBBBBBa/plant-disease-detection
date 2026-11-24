# Auditoría — taiwan (tomato)

Generado con `python scripts/audit_dataset.py --source taiwan --species tomato --seed 42`. Raíz: `data/raw/taiwan`. Suposición: carpeta = clase.

- Ficheros: **622** en 6 clases.
- Ilegibles para OpenCV: **0**.

## Recuentos por clase

| class_name | original_split | files | unreadable |
|---|---|---|---|
| Bacterial spot | test | 22 | 0 |
| Bacterial spot | train | 88 | 0 |
| Black mold | test | 14 | 0 |
| Black mold | train | 53 | 0 |
| Gray spot | test | 17 | 0 |
| Gray spot | train | 67 | 0 |
| Late blight | test | 20 | 0 |
| Late blight | train | 78 | 0 |
| health | test | 22 | 0 |
| health | train | 84 | 0 |
| powdery mildew | test | 32 | 0 |
| powdery mildew | train | 125 | 0 |

## Extensiones y canales

| extension | count |
|---|---|
| .jpg | 622 |

| channels | count |
|---|---|
| 3 | 622 |

## Resoluciones (píxeles)

| stat | width | height |
|---|---|---|
| min | 227.0 | 227.0 |
| median | 227.0 | 227.0 |
| max | 227.0 | 227.0 |

Más frecuentes (ancho×alto):

| size | count |
|---|---|
| 227x227 | 622 |

## Ficheros ilegibles

Ninguno.

## Duplicados exactos (mismo SHA-1)

- Grupos dentro de la misma clase: 0
- Grupos entre clases distintas: 0
- Ficheros implicados: 0


## Duplicados exactos con otra fuente

Comparado con `reports/audit/plantvillage/inventory.csv`: **0** pares idénticos.


