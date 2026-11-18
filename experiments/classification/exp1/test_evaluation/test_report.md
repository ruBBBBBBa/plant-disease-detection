# Evaluación en Test (congelado: 34d88cde)

Configuración elegida en val: **combinadas + random_forest**
Commit: 34d88cde1e6d9616128cef926fb567e1b5723ee6
Fecha: 2026-11-16 16:59:22

## Resultados completos (todas las combinaciones grupo × modelo)

### Grupo: color
Features: 43

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.9392 | 0.9122 | 0.9297 | 0.8933 |  |
| svm_lineal | 0.9459 | 0.9595 | 0.9440 | 0.9555 |  |
| svm_rbf | 0.9595 | 0.9527 | 0.9611 | 0.9561 |  |
| decision_tree | 0.9324 | 0.8986 | 0.9366 | 0.9070 |  |
| random_forest | 0.9459 | 0.9527 | 0.9416 | 0.9441 |  |

### Grupo: textura
Features: 22

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.9527 | 0.9595 | 0.9547 | 0.9635 |  |
| svm_lineal | 0.9561 | 0.9595 | 0.9547 | 0.9566 |  |
| svm_rbf | 0.9662 | 0.9865 | 0.9590 | 0.9720 |  |
| decision_tree | 0.9459 | 0.9865 | 0.9367 | 0.9680 |  |
| random_forest | 0.9493 | 0.9730 | 0.9437 | 0.9618 |  |

### Grupo: manchas
Features: 7

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.6047 | 0.7432 | 0.5984 | 0.7306 |  |
| svm_lineal | 0.7365 | 0.6014 | 0.6687 | 0.4658 |  |
| svm_rbf | 0.7500 | 0.6014 | 0.6785 | 0.4583 |  |
| decision_tree | 0.7365 | 0.5743 | 0.6674 | 0.4361 |  |
| random_forest | 0.7399 | 0.6419 | 0.6832 | 0.5285 |  |

### Grupo: combinadas
Features: 72

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.9730 | 0.9595 | 0.9636 | 0.9406 |  |
| svm_lineal | 0.9730 | 0.9662 | 0.9773 | 0.9749 |  |
| svm_rbf | 0.9899 | 0.9932 | 0.9853 | 0.9840 |  |
| decision_tree | 0.9561 | 0.9797 | 0.9303 | 0.9281 |  |
| random_forest | 0.9696 | 0.9865 | 0.9652 | 0.9777 | ✓ |

### Grupo: color_manchas
Features: 50

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.9324 | 0.8986 | 0.9283 | 0.8904 |  |
| svm_lineal | 0.9392 | 0.9392 | 0.9490 | 0.9589 |  |
| svm_rbf | 0.9662 | 0.9662 | 0.9606 | 0.9549 |  |
| decision_tree | 0.9291 | 0.8986 | 0.9304 | 0.9013 |  |
| random_forest | 0.9561 | 0.9595 | 0.9487 | 0.9446 |  |

### Grupo: control_nitidez
Features: 1

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.9257 | 0.9595 | 0.9217 | 0.9515 |  |
| svm_lineal | 0.9459 | 0.9662 | 0.9386 | 0.9515 |  |
| svm_rbf | 0.9392 | 0.9662 | 0.9375 | 0.9629 |  |
| decision_tree | 0.9392 | 0.9797 | 0.9388 | 0.9789 |  |
| random_forest | 0.9189 | 0.9527 | 0.9254 | 0.9658 |  |

### Grupo: textura_suavizada
Features: 22

| Modelo | Test Subset Balanced Acc | Test Subset Recall Diseased | Test Full Balanced Acc | Test Full Recall Diseased | Elegido |
|---|---:|---:|---:|---:|:---:|
| knn | 0.7838 | 0.6554 | 0.8208 | 0.7295 |  |
| svm_lineal | 0.8615 | 0.7973 | 0.8729 | 0.8202 |  |
| svm_rbf | 0.8615 | 0.7973 | 0.8812 | 0.8368 |  |
| decision_tree | 0.7872 | 0.6824 | 0.8104 | 0.7289 |  |
| random_forest | 0.8277 | 0.7703 | 0.8321 | 0.7791 |  |

## Recall por enfermedad (test_full) - Modelo elegido

| Enfermedad | Recall | Soporte |
|---|---:|---:|
| bacterial_spot | 1.0000 | 316 |
| early_blight | 0.9797 | 148 |
| late_blight | 1.0000 | 136 |
| leaf_mold | 1.0000 | 140 |
| septoria_leaf_spot | 0.9865 | 148 |
| spider_mites | 0.9516 | 248 |
| target_spot | 0.8942 | 208 |
| yellow_leaf_curl_virus | 1.0000 | 408 |

## Análisis de errores

### Errores en val (modelo elegido: combinadas + RF)

- **Falsos negativos (enfermas → sanas): 0** (recall_diseased = 1.0 en val)
- **Falsos positivos (sanas → enfermas): 11** (todas healthy)

Matriz de confusión val: TN=137, FP=11, FN=0, TP=148

### Errores en test_full (modelo elegido: combinadas + RF)

- **Falsos negativos (enfermas → sanas): 39**
  - target_spot: 22 (recall 0.894)
  - spider_mites: 12 (recall 0.952)
  - early_blight: 3 (recall 0.980)
  - septoria_leaf_spot: 2 (recall 0.986)
- **Falsos positivos (sanas → enfermas): 7** (todas healthy)

Matriz de confusión test_full: TN=141, FP=7, FN=39, TP=1713

### Clasificación manual de tipos de error (val + test)

| Tipo | Descripción | Ejemplos |
|---|---|---|
| FN: target_spot | Lesiones circulares bien definidas que la textura/color confunde con hoja sana | 22 casos en test |
| FN: spider_mites | Punteado fino disperso que la segmentación a veces no captura bien como manchas | 12 casos en test |
| FN: early_blight | Lesiones concéntricas tempranas con poco contraste cromático | 3 casos en test |
| FN: septoria | Manchas pequeñas con halo amarillento, frontera difusa | 2 casos en test |
| FP: healthy | Textura de fondo, iluminación inhomogénea o reflejos que generan "manchas" espurias | 11 val + 7 test |

## Importancias por permutación (val, modelo elegido: combinadas + RF)

| Feature | Importancia media | Std |
|---|---:|---:|
| glcm_contrast | 0.0247 | 0.0080 |
| l_std | 0.0068 | 0.0043 |
| lbp_04 | 0.0054 | 0.0053 |
| v_std | 0.0044 | 0.0030 |
| lbp_03 | 0.0037 | 0.0032 |

**Features clave de sesgo:**
- `glcm_contrast` (textura): **0.0247** — feature más importante; correlacionada con nitidez
- `l_std` (color L*): 0.0068 — variabilidad de luminosidad en hoja
- `v_std` (color V): 0.0044 — variabilidad de valor HSV
- `sharpness_laplacian_var`: **NOT IN GROUP** (excluida de combinadas)

> Nota: La permutación mide la caída en balanced_accuracy al barajar cada feature. Valores cercanos a 0 indican que el modelo no depende de esa feature en val.

### Importancias por permutación (val, color_manchas + svm_lineal — modelo B para fase 5)

| Feature | Importancia media | Std |
|---|---:|---:|
| s_std | 0.0615 | 0.0089 |
| h_circ_var | 0.0574 | 0.0114 |
| v_std | 0.0466 | 0.0122 |
| h_hist_11 | 0.0432 | 0.0090 |
| l_mean | 0.0409 | 0.0096 |
| b_std | 0.0409 | 0.0123 |
| frac_yellow | 0.0307 | 0.0107 |
| spot_area_frac | 0.0274 | 0.0097 |

**Features clave:** `v_std` (0.0466), `spot_area_frac` (0.0274), `frac_yellow` (0.0307). Sin textura ni nitidez explícita.

## Análisis del sesgo de nitidez (ampliado: mismo modelo en todos los grupos)

### Tabla: Random Forest (test_subset)

| Grupo | Balanced Acc | Recall Diseased |
|---|---:|---:|
| control_nitidez | 0.9189 | 0.9527 |
| textura | 0.9493 | 0.9730 |
| textura_suavizada | 0.8277 | 0.7703 |
| combinadas | 0.9696 | 0.9865 |
| color_manchas | 0.9561 | 0.9595 |

### Tabla: SVM RBF (test_subset)

| Grupo | Balanced Acc | Recall Diseased |
|---|---:|---:|
| control_nitidez | 0.9392 | 0.9662 |
| textura | 0.9662 | 0.9865 |
| textura_suavizada | 0.8615 | 0.7973 |
| combinadas | 0.9899 | 0.9932 |
| color_manchas | 0.9662 | 0.9662 |

### Conclusiones del sesgo

1. **El atajo de nitidez es real y sustancial**: `control_nitidez` sola (RF) alcanza 0.9189 balanced_acc en test. SVM RBF llega a 0.9392.

2. **El suavizado reduce el sesgo pero no lo elimina**: AUC post-suavizado (sigma=3.0, máxima probada) en `control_nitidez` = **0.161** (bajó de 0.988 sin blur). Sin embargo, la mediana de nitidez en sanas (4.65) sigue por encima del Q3 de enfermas (4.22) — el criterio estricto no se cumple con ninguna sigma probada.

3. **El suavizado destruye textura real**: 
   - RF: `textura` 0.9493 → `textura_suavizada` 0.8277 (−12 pts)
   - SVM RBF: `textura` 0.9662 → `textura_suavizada` 0.8615 (−10 pts)
   La caída drástica confirma que gran parte del poder de la textura original proviene de la nitidez de captura.

4. **Añadir textura a color+manchas mejora, pero parte es nitidez**:
   - RF: `combinadas` 0.9696 vs `color_manchas` 0.9561 (+1.3 pts)
   - SVM RBF: `combinadas` 0.9899 vs `color_manchas` 0.9662 (+2.4 pts)
   La mejora es real (las manchas y textura aportan señal de síntoma), pero el análisis de `textura_suavizada` muestra que una fracción significativa del aporte de textura es el atajo de nitidez.

5. **Modelo B para fase 5 (PlantDoc)**: `color_manchas` + **svm_lineal** (mejor en val: 0.9459 balanced_acc).
   - Hipótesis preregistrada (D030): *«B pierde menos rendimiento que A en PlantDoc porque no depende de la textura, ligada a la nitidez de captura (D023)»*.
   - Justificación: D023 (riesgo documentado) y resultados de val, **no test**.
   - Test_subset: A=0.9696, B=0.9392 (A gana +3.0 pts en PlantVillage).
   - Test_full: A=0.9652, B=0.9490 (A gana +1.6 pts).
   - En PlantDoc (fotos naturales, distinta nitidez), se espera que B sea más robusto.

## Comando para regenerar modelos desde cero

```bash
# 1. Calcular sigma (si no existe)
python scripts/train_classifiers.py compute-sigma --seed 42

# 2. Generar v1_blur
python scripts/train_classifiers.py generate-blur --sigma 3.0 --seed 42

# 3. CV + val
python scripts/train_classifiers.py cv-val --cv-folds 5 --seed 42 --out experiments/classification/exp1

# 4. Elegir mejor config (escribe configs/classification.json)
python scripts/train_classifiers.py choose-best --out experiments/classification/exp1

# 5. (Opcional) Evaluar test congelado
python scripts/eval_test.py --config configs/classification.json --frozen-commit <hash>
```

Los modelos entrenados se guardan en `models/` (no versionado). El comando 4 regenera `configs/classification.json` con el commit congelado actual.

---
*Nota: Las puntuaciones no son probabilidades calibradas.*