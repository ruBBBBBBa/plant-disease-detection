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

## Análisis del sesgo de nitidez

### control_nitidez (kNN) vs modelo elegido

- control_nitidez test_subset: balanced_acc=0.9257, recall_diseased=0.9595
- modelo elegido test_subset: balanced_acc=0.9696, recall_diseased=0.9865

### textura vs textura_suavizada (kNN en test_subset)
- textura: balanced_acc=0.9527
- textura_suavizada: balanced_acc=0.7838

### combinadas vs color_manchas (kNN en test_subset)
- combinadas: balanced_acc=0.9730
- color_manchas: balanced_acc=0.9324

---
*Nota: Las puntuaciones no son probabilidades calibradas.*