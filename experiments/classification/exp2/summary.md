# Resultados CV (train) + Val por grupo × modelo

## Grupo: color
Features: 43

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9303 | 0.9189 | 0.9054 | 0.9486 |
| svm_lineal | 0.9622 | 0.9324 | 0.9324 | 0.9841 |
| svm_rbf | 0.9560 | 0.9324 | 0.9392 | 0.9819 |
| decision_tree | 0.9430 | 0.8919 | 0.8378 | 0.9156 |
| random_forest | 0.9608 | 0.9189 | 0.9122 | 0.9791 |

## Grupo: textura
Features: 22

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9488 | 0.9561 | 0.9932 | 0.9767 |
| svm_lineal | 0.9573 | 0.9595 | 0.9932 | 0.9864 |
| svm_rbf | 0.9545 | 0.9493 | 1.0000 | 0.9804 |
| decision_tree | 0.9374 | 0.9392 | 0.9797 | 0.9546 |
| random_forest | 0.9481 | 0.9493 | 0.9865 | 0.9729 |

## Grupo: manchas
Features: 7

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.5987 | 0.5878 | 0.7365 | 0.5924 |
| svm_lineal | 0.7042 | 0.6723 | 0.5000 | 0.7348 |
| svm_rbf | 0.7127 | 0.6892 | 0.4797 | 0.7455 |
| decision_tree | 0.7084 | 0.6723 | 0.4392 | 0.7135 |
| random_forest | 0.7247 | 0.7230 | 0.5811 | 0.7719 |

## Grupo: combinadas
Features: 72

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9531 | 0.9257 | 0.9257 | 0.9698 |
| svm_lineal | 0.9765 | 0.9595 | 0.9662 | 0.9875 |
| svm_rbf | 0.9772 | 0.9561 | 0.9730 | 0.9851 |
| decision_tree | 0.9438 | 0.9122 | 0.9392 | 0.8979 |
| random_forest | 0.9680 | 0.9628 | 1.0000 | 0.9895 |

## Grupo: color_manchas
Features: 50

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9360 | 0.9223 | 0.8986 | 0.9495 |
| svm_lineal | 0.9644 | 0.9459 | 0.9459 | 0.9905 |
| svm_rbf | 0.9581 | 0.9392 | 0.9527 | 0.9843 |
| decision_tree | 0.9365 | 0.8885 | 0.8311 | 0.9107 |
| random_forest | 0.9672 | 0.9324 | 0.9054 | 0.9822 |

## Grupo: color_manchas_sin_l_v
Features: 48

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9275 | 0.9122 | 0.8784 | 0.9466 |
| svm_lineal | 0.9643 | 0.9426 | 0.9459 | 0.9912 |
| svm_rbf | 0.9588 | 0.9527 | 0.9595 | 0.9872 |
| decision_tree | 0.9046 | 0.9054 | 0.9459 | 0.9328 |
| random_forest | 0.9537 | 0.9392 | 0.9257 | 0.9835 |

## Grupo: control_nitidez
Features: 1

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.9417 | 0.9358 | 0.9797 | 0.9457 |
| svm_lineal | 0.9368 | 0.9426 | 0.9662 | 0.9761 |
| svm_rbf | 0.9382 | 0.9493 | 0.9797 | 0.9588 |
| decision_tree | 0.9382 | 0.9459 | 0.9865 | 0.9677 |
| random_forest | 0.9382 | 0.9426 | 0.9865 | 0.9741 |

## Grupo: textura_suavizada
Features: 22

| Modelo | CV balanced_acc | Val balanced_acc | Val recall_diseased | Val ROC-AUC |
|---|---:|---:|---:|---:|
| knn | 0.7938 | 0.8243 | 0.7500 | 0.8940 |
| svm_lineal | 0.8449 | 0.8480 | 0.8446 | 0.9226 |
| svm_rbf | 0.8514 | 0.8682 | 0.8649 | 0.9431 |
| decision_tree | 0.7701 | 0.8378 | 0.7770 | 0.8818 |
| random_forest | 0.8201 | 0.8649 | 0.8311 | 0.9274 |
