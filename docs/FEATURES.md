# Diccionario de características (versión `v1`, T04)

Cada imagen de PlantVillage se convierte en **75 características** explícitas, calculadas **solo
dentro de la máscara de la hoja** (D017). Código: `src/features.py`. Parámetros (ajustados solo con
`train`): `configs/features.json`. Extracción: `scripts/extract_features.py`, que escribe
`data/processed/features_v1_*.csv` (no versionado).

- **Máscara:** segmentación congelada de T03 (K-means k = 3 en a\*b\* + postprocesado,
  `configs/segmentation.json`). La variante de sombra elegida con `train` es `raw`, es decir, la
  máscara tal cual (ver `reports/features/tuning_train.md`).
- **Rangos de OpenCV:** H 0–179 (grados / 2); S, V, L 0–255; a\*, b\* desplazados +128. Abajo los
  rangos ya están convertidos a unidades interpretables.
- **Cautela (regla 8 de `AGENTS.md`):** ninguna característica es por sí sola un diagnóstico. «Por
  qué podría relacionarse» es una hipótesis que el clasificador pondera. Verde no implica sano.
- **Sesgo de captura detectado en `train`:** las hojas sanas de PlantVillage están fotografiadas
  mucho más nítidas (varianza del laplaciano: mediana 20.112 frente a 952–9.311 en las enfermas;
  AUC 0,014). Las características de **textura** y la dispersión del brillo (`l_std`, `v_std`)
  separan casi perfectamente, pero en parte por eso. Ver `reports/features/features_train.md`.

## Calidad de la máscara (3) — no son síntomas

| Nombre | Cálculo | Rango | Uso |
|---|---|---|---|
| `leaf_area_frac` | píxeles de hoja / píxeles de la imagen (máscara de T03) | 0–1 | Calidad; regla candidata de «no concluyente» |
| `border_touch_frac` | fracción de los píxeles del borde de la imagen que son hoja | 0–1 | Calidad; regla candidata |
| `mask_solidity` | área / área de la envolvente convexa de la mayor región | 0–1 | **Regla activa de «no concluyente»** (< 0,6504, p1 de `train`) |

## Color (43) — Tema 2 (espacios de color, histogramas) y Tema 4

| Nombre | Cálculo | Rango | Por qué podría relacionarse con síntomas |
|---|---|---|---|
| `h_circ_mean_deg` | media **circular** del tono, H × 2 en grados | 0–360 | La clorosis y la necrosis alejan el tono del verde (p5–p95 de las sanas de `train`: ≈ 64°–106°) hacia amarillos y naranjas |
| `h_circ_var` | varianza circular del tono, 1 − \|vector medio\| | 0–1 | Una hoja con manchas mezcla tonos y tiene más dispersión |
| `s_mean`, `s_std` | media y desviación de S / 255 | 0–1 | Las lesiones amarillas o marrones cambian la saturación |
| `v_mean`, `v_std` | media y desviación de V / 255 | 0–1 | Brillo; `v_std` también refleja textura e iluminación (sesgo de captura) |
| `l_mean`, `l_std` | media y desviación de L\* (0–100) | 0–100 | Luminosidad perceptual; ídem |
| `a_mean`, `a_std` | media y desviación de a\* con signo (− verde, + rojo) | ≈ −128–127 | La necrosis desplaza a\* hacia valores positivos |
| `b_mean`, `b_std` | media y desviación de b\* con signo (− azul, + amarillo) | ≈ −128–127 | El amarilleo sube b\* |
| `h_hist_00` … `h_hist_11` | histograma de H en 12 intervalos de 15 unidades (30°), normalizado | 0–1 (suma 1) | Distribución gruesa del tono |
| `a_hist_00` … `a_hist_07` | histograma de a\* en 8 intervalos dentro de [−27, 5] (p1–p99 de `train`; los extremos acumulan lo de fuera) | 0–1 (suma 1) | Distribución verde–rojo |
| `b_hist_00` … `b_hist_07` | histograma de b\* en 8 intervalos dentro de [1, 48] | 0–1 (suma 1) | Distribución azul–amarillo |
| `frac_yellow` | fracción de hoja con V ≥ 50, S ≥ 40 y 20 ≤ H < 29 (40°–58°) | 0–1 | Clorosis o amarilleo |
| `frac_brown` | fracción con V ≥ 50, S ≥ 40 y H < 20 o H ≥ 160 | 0–1 | Necrosis marrón o rojiza |
| `frac_dark` | fracción con V < 15 | 0–1 | Tejido muy oscuro. En `train` las sanas tienen **más** (AUC 0,21) |

Umbrales de tono (`train`, hojas sanas): `h_green_min` = 29 (p1 de H en píxeles con S ≥ 40 y
V ≥ 50); `v_dark` = 15 (p1 de V); `h_brown_yellow` = 20 (teoría del color). El tono solo cuenta si
V ≥ 50 y S ≥ 40, porque en píxeles oscuros o grises H es ruido.

## Textura (22) — Tema 4

| Nombre | Cálculo | Rango | Por qué podría relacionarse con síntomas |
|---|---|---|---|
| `glcm_contrast` | contraste de la GLCM (gris en 32 niveles, distancia 1, media de 0°/45°/90°/135°) | ≥ 0 | Cambios bruscos de gris (bordes de lesión, nervios, granulado) |
| `glcm_homogeneity` | homogeneidad (inverse difference moment) | 0–1 | Tejido uniforme frente a moteado |
| `glcm_energy` | energía (√ASM) | 0–1 | Regularidad de la textura |
| `glcm_correlation` | correlación lineal de grises vecinos | −1–1 | Estructura de gran escala frente a ruido fino |
| `lbp_00` … `lbp_17` | histograma LBP uniforme (P = 16, R = 2): 17 patrones uniformes + 1 «no uniforme», normalizado | 0–1 (suma 1) | Micropatrones: zonas planas, bordes, esquinas, puntos |

**Exclusión del fondo:**
- **GLCM:** el fondo va al nivel 0 y se eliminan la fila y la columna 0 de la matriz antes de
  normalizar. No cuenta ningún par hoja–fondo.
- **LBP:** el histograma solo usa píxeles de la máscara erosionada R + 1 píxeles. Los vecinos
  interpolados nunca tocan el fondo; lo comprueba `tests/test_features.py`.

Parámetros elegidos con `train` por la media de |AUC − 0,5| de la familia: GLCM con distancias
1, 2 y 4; LBP con (P, R) = (8, 1), (16, 2) y (24, 3).

## Manchas / regiones candidatas (7) — Tema 6

Regiones candidatas = píxeles de la hoja `yellow` ∪ `brown` ∪ `dark`, con una apertura 3×3 y sin
las componentes de menos de 10 píxeles. Es **evidencia visual de tonos anómalos**, no la zona en la
que se fija el clasificador. En `train` capta bien las lesiones marrones (*blight*, *septoria*,
*bacterial spot*) y apenas la clorosis suave (YLCV, *leaf mold*) o el daño de *spider mites*
(`reports/features/spots_train.jpg`).

| Nombre | Cálculo | Rango | Por qué podría relacionarse con síntomas |
|---|---|---|---|
| `spot_count` | número de componentes conexas (8-vecindad) | ≥ 0 | Muchas manchas pequeñas: p. ej. mancha bacteriana o *septoria* |
| `spot_area_frac` | área total de las regiones / área de la hoja | 0–1 | Extensión de la zona afectada |
| `spot_mean_area_frac` | área media de una región / área de la hoja | 0–1 | Manchas pequeñas frente a grandes tizones |
| `spot_max_area_frac` | área de la mayor región / área de la hoja | 0–1 | Lesión dominante |
| `spot_mean_solidity` | media de área / envolvente convexa | 0–1 | Lesiones compactas frente a irregulares |
| `spot_mean_eccentricity` | media de la excentricidad de la elipse equivalente | 0–1 | Manchas redondas frente a alargadas |
| `spot_mean_circularity` | media de 4πA/P² (máximo 1) | 0–1 | Redondez del contorno |

Sin regiones, todas valen 0.
