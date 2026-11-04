# Datos

Los datasets **no se versionan** (`data/raw/` y `data/processed/` están en `.gitignore`).
`data/raw/` es de solo lectura: ningún script escribe, mueve ni borra nada ahí.

## Qué se ha comprobado y qué no

Comprobado el 4-nov-2026 listando el árbol git de los repositorios oficiales (sin descargar
las imágenes), y descargando una clase pequeña para probar el comando de descarga parcial.

### PlantVillage — https://github.com/spMohanty/PlantVillage-Dataset

- Carpetas de primer nivel: `raw/` (con `color/`, `grayscale/`, `segmented/`), `leaf_grouping/`,
  `leaf-map.json`, `data_distribution_for_SVM/` (`train/`, `test/`, `train_mapping.txt`,
  `test_mapping.txt`), `scripts/`, `plant_village.py`, `README.md`, `README_HF.md`, `CITATION.cff`.
- Ficheros en `raw/`: 54.305 en `color`, 54.305 en `grayscale`, 54.306 en `segmented`.
- Su README dice que los scripts del repo son para **Python 2.7** y recomienda la versión de
  Hugging Face (`mohanty/PlantVillage`), que según ese README ofrece particiones 80/20 que respetan
  la agrupación por hoja. **No verificado por nosotros** (se investiga en T01).
- Tomate: 10 carpetas, con el mismo número de ficheros en `color`, `grayscale` y `segmented`:

| Carpeta (`raw/color/…`) | Ficheros | `binary_label` previsto |
|---|---:|---|
| `Tomato___healthy` | 1.591 | healthy |
| `Tomato___Bacterial_spot` | 2.127 | diseased |
| `Tomato___Early_blight` | 1.000 | diseased |
| `Tomato___Late_blight` | 1.909 | diseased |
| `Tomato___Leaf_Mold` | 952 | diseased |
| `Tomato___Septoria_leaf_spot` | 1.771 | diseased |
| `Tomato___Spider_mites Two-spotted_spider_mite` (con espacio) | 1.676 | diseased |
| `Tomato___Target_Spot` | 1.404 | diseased |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5.357 | diseased |
| `Tomato___Tomato_mosaic_virus` | 373 | diseased |
| **Total** | **18.160** (1.591 sanas / 16.569 enfermas) | |

- Nombres de fichero de ejemplo: `raw/color/Tomato___healthy/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG`
  y su par segmentado `raw/segmented/Tomato___healthy/000146ff-…___GH_HL Leaf 259.1_final_masked.jpg`.
  Hay extensiones `.JPG` y `.jpg`. La parte tras `___` (`GH_HL Leaf 259.1`, `RS_HL 0105`)
  **podría** identificar la hoja o la sesión de captura: es una hipótesis sin verificar.
- Ojo: el desequilibrio es grande (≈ 1 sana por cada 10 enfermas) y la clase
  *Yellow Leaf Curl Virus* es casi un tercio de las enfermas.

### PlantDoc — https://github.com/pratikkayal/PlantDoc-Dataset

- Carpetas de primer nivel: `train/`, `test/`, `README.md`, `LICENSE.txt`. 28 clases en total.
- Tomate (nombres de carpeta tal cual, distintos de PlantVillage):

| Carpeta | `train` | `test` | Correspondencia probable |
|---|---:|---:|---|
| `Tomato leaf` | 55 | 8 | healthy |
| `Tomato Early blight leaf` | 79 | 9 | Early_blight |
| `Tomato Septoria leaf spot` | 140 | 11 | Septoria_leaf_spot |
| `Tomato leaf bacterial spot` | 101 | 9 | Bacterial_spot |
| `Tomato leaf late blight` | 101 | 10 | Late_blight |
| `Tomato leaf mosaic virus` | 44 | 10 | Tomato_mosaic_virus |
| `Tomato leaf yellow virus` | 70 | 6 | Tomato_Yellow_Leaf_Curl_Virus (¿?) |
| `Tomato mold leaf` | 85 | 6 | Leaf_Mold |
| `Tomato two spotted spider mites leaf` | 2 | 0 | Spider_mites |

- Total tomate: 746 ficheros, de los que solo **63 son sanos**. No hay clase *Target Spot*.
  Como test externo estará muy desequilibrado: hay que tenerlo en cuenta en las métricas.
- La correspondencia de la última columna es una propuesta; el mapeo definitivo se fija en T02
  en `configs/` de forma explícita.

**Sin comprobar todavía (tarea T01):** contenido y formato de `leaf_grouping/`, `leaf-map.json`
y `*_mapping.txt`; resoluciones; imágenes ilegibles; duplicados dentro y entre fuentes;
correspondencia 1:1 entre `color` y `segmented`.

## Descarga (solo tomate, descarga parcial con git)

Usa *sparse checkout* para bajar únicamente las carpetas de tomate en lugar del repositorio entero.
Probado con una clase: 373 imágenes ocupan 7,5 MB en `color` y 3,2 MB en `segmented`; por
extrapolación, todo el tomate (`color` + `segmented`) ronda **0,5 GB** (estimación, no medido).

Desde la raíz del repo:

```bash
# PlantVillage: tomate en color y segmentado + metadatos de agrupación por hoja
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/plantvillage
git -C data/raw/plantvillage sparse-checkout set --no-cone \
  '/raw/color/Tomato___*/' '/raw/segmented/Tomato___*/' \
  '/leaf_grouping/' '/leaf-map.json' '/data_distribution_for_SVM/*_mapping.txt' \
  '/README.md' '/README_HF.md' '/CITATION.cff' '/plant_village.py'
git -C data/raw/plantvillage checkout

# PlantDoc: solo tomate (test externo)
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/plantdoc
git -C data/raw/plantdoc sparse-checkout set --no-cone \
  '/train/Tomato*/' '/test/Tomato*/' '/README.md' '/LICENSE.txt'
git -C data/raw/plantdoc checkout
```

Comprobación rápida:

```bash
ls data/raw/plantvillage/raw/color | grep -c Tomato     # 10
find data/raw/plantvillage/raw/color -type f | wc -l     # 18160
find data/raw/plantdoc/train data/raw/plantdoc/test -type f | wc -l   # 746
```

Si más adelante hacen falta patata o pimiento, se añade el patrón a `sparse-checkout`
(`git -C data/raw/plantvillage sparse-checkout add '/raw/color/Potato___*/'`) sin volver a clonar.

## Rutas esperadas

```text
data/raw/plantvillage/raw/color/Tomato___<clase>/<fichero>.JPG|jpg
data/raw/plantvillage/raw/segmented/Tomato___<clase>/<fichero>_final_masked.jpg
data/raw/plantvillage/leaf_grouping/ , leaf-map.json , data_distribution_for_SVM/*_mapping.txt
data/raw/plantdoc/{train,test}/Tomato*/<fichero>
data/metadata.csv            manifiesto generado por scripts/prepare_dataset.py (T02)
```

## Uso previsto de cada fuente

- **PlantVillage `color`**: entrada del sistema (train / val / test interno).
- **PlantVillage `segmented`**: solo como **referencia para evaluar** nuestra segmentación de la
  hoja; nunca sustituye a la segmentación propia del pipeline.
- **PlantDoc**: `external_test` (fondos e iluminación naturales). Nunca para entrenar ni ajustar.
- Fotos propias: pruebas cualitativas; cuantitativas solo con etiquetas fiables.

## Referencias de los datasets

- Mohanty, S. P., Hughes, D. P., Salathé, M. (2016). *Using Deep Learning for Image-Based Plant
  Disease Detection*. Frontiers in Plant Science, 7:1419. (Se cita por el dataset, no por el método.)
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N. (2020). *PlantDoc: A Dataset
  for Visual Plant Disease Detection*. CoDS-COMAD 2020.
- Licencias: ver `CITATION.cff` (PlantVillage) y `LICENSE.txt` (PlantDoc) tras la descarga.
