# Auditoría — plantdoc (tomato)

Generado con `python scripts/audit_dataset.py --source plantdoc --species tomato --seed 42`. Raíz: `data/raw/plantdoc`. Suposición: carpeta = clase.

- Ficheros: **746** en 9 clases.
- Ilegibles para OpenCV: **0**.

## Recuentos por clase

| class_name | original_split | files | unreadable |
|---|---|---|---|
| Tomato Early blight leaf | test | 9 | 0 |
| Tomato Early blight leaf | train | 79 | 0 |
| Tomato Septoria leaf spot | test | 11 | 0 |
| Tomato Septoria leaf spot | train | 140 | 0 |
| Tomato leaf | test | 8 | 0 |
| Tomato leaf | train | 55 | 0 |
| Tomato leaf bacterial spot | test | 9 | 0 |
| Tomato leaf bacterial spot | train | 101 | 0 |
| Tomato leaf late blight | test | 10 | 0 |
| Tomato leaf late blight | train | 101 | 0 |
| Tomato leaf mosaic virus | test | 10 | 0 |
| Tomato leaf mosaic virus | train | 44 | 0 |
| Tomato leaf yellow virus | test | 6 | 0 |
| Tomato leaf yellow virus | train | 70 | 0 |
| Tomato mold leaf | test | 6 | 0 |
| Tomato mold leaf | train | 85 | 0 |
| Tomato two spotted spider mites leaf | train | 2 | 0 |

## Extensiones y canales

| extension | count |
|---|---|
| .jpg | 745 |
| .png | 1 |

| channels | count |
|---|---|
| 3 | 745 |
| 4 | 1 |

## Resoluciones (píxeles)

| stat | width | height |
|---|---|---|
| min | 115.0 | 69.0 |
| median | 640.0 | 562.0 |
| max | 6000.0 | 6000.0 |

Más frecuentes (ancho×alto):

| size | count |
|---|---|
| 800x600 | 21 |
| 1600x1200 | 17 |
| 1024x768 | 15 |
| 900x675 | 13 |
| 1200x900 | 12 |

## Ficheros ilegibles

Ninguno.

## Duplicados exactos (mismo SHA-1)

- Grupos dentro de la misma clase: 2
- Grupos entre clases distintas: 1
- Ficheros implicados: 6

| files | scope | classes | paths |
|---|---|---|---|
| 2 | same_class | Tomato Septoria leaf spot | data/raw/plantdoc/train/Tomato Septoria leaf spot/early-blight-septoria-ls-fig-3.jpg | data/raw/plantdoc/test/Tomato Septoria leaf spot/early-blight-septoria-ls-fig-3.jpg |
| 2 | same_class | Tomato leaf yellow virus | data/raw/plantdoc/train/Tomato leaf yellow virus/tylcv-seminar-1-638.jpg | data/raw/plantdoc/test/Tomato leaf yellow virus/tylcv-seminar-1-638.jpg |
| 2 | cross_class | Tomato Septoria leaf spot | Tomato leaf bacterial spot | data/raw/plantdoc/train/Tomato Septoria leaf spot/tomato_V8.jpg | data/raw/plantdoc/test/Tomato leaf bacterial spot/tomato_V8.jpg |
## Duplicados exactos con otra fuente

Comparado con `reports/audit/plantvillage/inventory.csv`: **0** pares idénticos.


