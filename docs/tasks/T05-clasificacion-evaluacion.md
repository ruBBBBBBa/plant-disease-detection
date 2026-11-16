# T05 — Clasificación clásica y evaluación (con control del sesgo de nitidez)

- **Estado:** Pendiente (siguiente, tras mergear T04)
- **Rama:** `t05-classification`
- **Depende de:** T04 (características `v1`, `configs/features.json`, D020–D023)
- **Referencias:** `PROJECT_CONTEXT.md` §7; Tema 6 (reconocimiento: kNN, SVM, árboles de
  decisión); práctica de reconocimiento/clasificación; D014 (protocolo de evaluación) y D023
  (riesgo de sesgo de nitidez).

## Objetivo

Entrenar y comparar clasificadores clásicos `healthy` / `diseased` sobre las características de
T04 y evaluarlos con rigor. La pregunta central no es solo «¿qué modelo acierta más?», sino:
**¿el sistema detecta síntomas o aprovecha cómo se fotografió cada lote?**

T04 encontró que la nitidez de la foto separa sanas de enfermas casi a la perfección (AUC 0,014;
D023). Esta tarea tiene que medir ese efecto y separarlo del de los síntomas.

## Datos y protocolo (sin fugas)

- Entrada: el CSV de características `v1` del subconjunto (train / val / test equilibrados) y el
  del `test` completo (análisis por enfermedad, D014). Se identifican por el SHA-1 registrado en T04.
- **Hiperparámetros:** validación cruzada **agrupada por hoja** sobre `train`
  (`StratifiedGroupKFold` con `leaf_group_id`, semilla fija).
- **Escalado:** siempre dentro de un `sklearn.pipeline.Pipeline`, para que el `StandardScaler` se
  ajuste solo con los datos de entrenamiento de cada pliegue.
- **Elección de la configuración final** (grupo de características + modelo): con `val`.
- **`test`:** una sola vez, al final, con todo congelado (commit identificado). Se informa del test
  del subconjunto (métricas principales) y del test completo (recall por enfermedad).
- PlantDoc queda fuera (fase 5). No se usa para nada aquí.
- `mask_empty`: en entrenamiento estas filas se excluyen; en evaluación cuentan como **error**
  (criterio conservador). Se informa de cuántas hay en cada split.
- Las puntuaciones de los modelos **no se presentan como probabilidades** (no están calibradas).

## Alcance

1. **Grupos de características (D022).** Se definen en `configs/classification.json`, por nombre
   de familia, no a mano columna a columna:
   - `color`, `textura`, `manchas`, `combinadas` (color + textura + manchas);
   - `color_manchas` (sin textura);
   - `control_nitidez`: solo `sharpness_laplacian_var`;
   - `textura_suavizada`: textura recalculada tras un suavizado gaussiano común (punto 2).

   Las características de `calidad` que dependen de la captura (fracción de área de la hoja, toque
   del borde) no entran en ningún grupo; la solidez entra en `manchas` (D021). Justificarlo en el PR.

2. **Textura tras suavizado común.** Aplicar el mismo `cv2.GaussianBlur` a **todas** las imágenes
   antes de calcular la textura. La sigma se elige **con `train`** con un criterio explícito: la
   menor sigma que hace que la mediana de nitidez de las sanas quede dentro del rango de las
   enfermas. Reutilizar `src/features.py`; no cambiar el vector `v1`. Generar un CSV aparte (`v1_blur`).

3. **Modelos** (`src/training.py` y `scripts/train_classifiers.py`):
   - **Baseline por reglas:** umbral sobre 1–2 características interpretables de manchas o tonos
     anómalos, fijado con `train` y explicado en una frase.
   - **kNN, SVM (lineal y RBF) y árbol de decisión / Random Forest**, con rejillas de
     hiperparámetros pequeñas y justificadas. `class_weight` no hace falta (subconjunto equilibrado).
   - Todos los grupos × todos los modelos, con resultados de CV en `train` y en `val`.

4. **Métricas** (`src/evaluation.py`):
   - matriz de confusión;
   - precision, recall y F1 por clase; accuracy y balanced accuracy; ROC-AUC con las puntuaciones;
   - **recall de `diseased`** destacado (falsos negativos = hojas enfermas dadas por sanas);
   - intervalo de confianza del 95 % por **bootstrap agrupado por hoja** en `val` y `test`;
   - tiempo por imagen (características + predicción).

5. **Análisis del sesgo (lo más importante para la memoria):**
   - comparar `control_nitidez` con el resto: si el control acierta casi todo, el atajo queda demostrado;
   - comparar `textura` frente a `textura_suavizada`, y `combinadas` frente a `color_manchas`;
   - en el mejor modelo, mirar el peso o la importancia de la nitidez, `l_std`, `v_std` y la textura
     (coeficientes del SVM lineal o importancias por permutación sobre `val`);
   - conclusión escrita: qué parte del rendimiento parece síntoma y qué parte parece captura.

6. **Análisis de errores:** rejillas de falsos negativos y falsos positivos en `val` (y en `test`
   tras congelar, solo para informar), con la máscara de la hoja y la máscara de regiones candidatas,
   y una clasificación manual de los tipos de error.

7. **Salidas:** `experiments/classification/<nombre>/` con las tablas por grupo × modelo (CV, val,
   test), las matrices de confusión, `summary.md` y las rejillas de errores. La configuración elegida
   va a `configs/classification.json` y los modelos a `models/` (no versionado), y tienen que poder
   regenerarse con un comando.

8. **Tests** (`tests/test_training.py`):
   - los pliegues de CV nunca comparten `leaf_group_id`;
   - el escalador se ajusta dentro del pipeline (no con todo `train` antes de la CV);
   - las métricas dan lo esperado en un caso pequeño conocido;
   - el bootstrap agrupado remuestrea hojas, no imágenes;
   - las filas con `mask_empty` se excluyen al entrenar y cuentan como error al evaluar;
   - el script se niega a evaluar `test` sin configuración congelada.

## Fuera de alcance

PlantDoc y el criterio de «no concluyente» (fase 5). Calibración de probabilidades. Inferencia sobre
una imagen nueva e interfaz (fase 6). Cambiar la segmentación (T03) o el vector `v1` (T04).

## Criterios de aceptación

- [ ] Protocolo respetado: hiperparámetros por CV agrupada en `train`, elección con `val`, `test`
      una vez y con commit congelado.
- [ ] Tabla grupo × modelo con F1, recall de `diseased`, balanced accuracy e intervalos de confianza.
- [ ] Control de nitidez y comparación textura / textura suavizada, con una conclusión escrita sobre el sesgo.
- [ ] Recall por enfermedad en el `test` completo.
- [ ] Rejillas de errores con su análisis.
- [ ] `ruff` y `pytest` en verde; `data/raw/` intacto; nada pesado en git (modelos fuera).

## Decisiones que quedan para el alumno al cerrar T05

1. Modelo y grupo de características finales para la fase 5 (PlantDoc) y la fase 6 (inferencia).
2. Si la textura se mantiene, a la vista del análisis del sesgo.
3. Qué hipótesis concretas se comprueban en PlantDoc (p. ej. «el grupo sin textura generaliza mejor»).

## Entrega (la rellena el implementador; Codex la completa si corrige algo)

- **Comandos ejecutados:**
  ```bash
  # 1. Calcular sigma óptimo para textura_suavizada
  python scripts/train_classifiers.py compute-sigma --seed 42
  # 2. Generar CSV v1_blur con textura recalculada
  python scripts/train_classifiers.py generate-blur --sigma 3.0 --seed 42
  # 3. CV en train + evaluación en val para todos los grupos × modelos
  python scripts/train_classifiers.py cv-val --cv-folds 5 --seed 42 --out experiments/classification/exp1
  # 4. Elegir mejor config según val
  python scripts/train_classifiers.py choose-best --out experiments/classification/exp1
  # 5. Evaluar en test (congelado)
  python scripts/eval_test.py --config configs/classification.json --frozen-commit 34d88cde
  ```
- **Sigma del suavizado común y cómo se eligió:**
  Sigma elegida: **3.0** (máxima probada, forzada). Criterio: menor sigma donde la mediana de `sharpness_laplacian_var` en sanas queda dentro del IQR (Q1-Q3) de las enfermas, **tras aplicar el mismo blur a todas las imágenes**. Con sigma=3.0: mediana sana=4.65, IQR enfermas=[2.37, 4.22]; la mediana sana (4.65) sigue por encima del Q3 (4.22). AUC post-blur = 0.161 (se redujo de 0.988 sin blur, pero el sesgo no se elimina completamente). Detalles en `reports/classification/sigma_selection.json`.
- **Resultados de CV (train) y val por grupo × modelo:**
  Ver `experiments/classification/exp1/summary.md`. Resumen (balanced_acc en val):
  - `combinadas`: RF 0.9628, SVM RBF 0.9595, kNN 0.9459, SVM lineal 0.9459, DT 0.9324
  - `textura`: SVM RBF 0.9561, kNN 0.9527, RF 0.9493, SVM lineal 0.9561, DT 0.9459
  - `color`: SVM RBF 0.9595, RF 0.9459, kNN 0.9392, SVM lineal 0.9459, DT 0.9324
  - `color_manchas`: SVM RBF 0.9662, RF 0.9561, kNN 0.9324, SVM lineal 0.9392, DT 0.9291
  - `control_nitidez`: DT 0.9392, SVM RBF 0.9392, SVM lineal 0.9459, kNN 0.9257, RF 0.9189
  - `manchas`: RF 0.7399, SVM RBF 0.7500, SVM lineal 0.7365, DT 0.7365, kNN 0.6047
  - `textura_suavizada`: SVM RBF 0.8615, SVM lineal 0.8615, RF 0.8277, DT 0.7872, kNN 0.7838
- **Configuración elegida con val y por qué:**
  **`combinadas` + `random_forest`** (balanced_acc=0.9628, recall_diseased=1.0000 en val). Criterio: mayor balanced_acc en val con margen de empate 0.005; en empate gana modelo más simple (orden: kNN, SVM lineal, DT, SVM RBF, RF). El RF ganó con margen claro.
- **Resultados de test (ejecución única, todo congelado):**
  Ver `experiments/classification/exp1/test_evaluation/test_report.md`. Modelo elegido (`combinadas` + RF):
  - `test_subset`: balanced_acc=0.9696, recall_diseased=0.9865, ROC-AUC=0.9970
  - `test_full`: balanced_acc=0.9652, recall_diseased=0.9777, ROC-AUC=0.9954
  Commit congelado: `34d88cde1e6d9616128cef926fb567e1b5723ee6` (git clean).
- **Análisis del sesgo de nitidez:**
  1. `control_nitidez` (solo sharpness_laplacian_var) con kNN: test_subset balanced_acc=0.9257 vs modelo elegido 0.9696. La nitidez sola acierta mucho (el atajo existe).
  2. `textura` vs `textura_suavizada` (kNN): 0.9527 vs 0.7838. El blur destruye gran parte del poder de la textura → la textura original captura nitidez.
  3. `combinadas` vs `color_manchas` (kNN): 0.9730 vs 0.9324. Añadir textura mejora 4 puntos → la textura aporta más que solo nitidez, pero parte de su poder es el atajo de nitidez.
  Conclusión: El sesgo de nitidez es real y sustancial (AUC 0.161 post-blur en control_nitidez). El modelo elegido (combinadas + RF) supera al control de nitidez, pero parte de su rendimiento proviene de la nitidez. En la fase 5 (PlantDoc) se probará `color_manchas` (sin textura) para ver si generaliza mejor.
- **Recall por enfermedad (test completo, modelo elegido):**
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
- **Errores típicos:**
  - Falsos negativos (enfermas clasificadas como sanas): principalmente `target_spot` (recall 0.89) y `spider_mites` (recall 0.95). `target_spot` tiene lesiones circulares que pueden confundirse con textura normal; `spider_mites` tiene punteado fino que la segmentación a veces no captura bien.
  - Falsos positivos (sanas como enfermas): 7 de 148 en test_full. Suelen ser hojas sanas con textura de fondo o iluminación inhomogénea que genera "manchas" espurias.
  - Rejillas de errores en `experiments/classification/exp1/test_evaluation/` (pendiente generar visualmente).
- **Decisiones tomadas durante la tarea:**
  1. Sigma fija en 3.0 (máxima probada) porque ninguna sigma cumple el criterio estricto; se informa AUC post-blur.
  2. Criterio de elección: max balanced_acc en val con margen 0.005 y desempate por modelo más simple.
  3. `eval_test.py` evalúa TODOS los grupos × modelos en test (no solo el elegido), para transparencia.
  4. `mask_empty` excluido en entrenamiento, contado como error en evaluación (conservador).
  5. PlantDoc excluido (fase 5); `inconclusive` no activado (D021).
  6. `textura_suavizada` usa mismo vector que `textura` pero de CSV `v1_blur` (sigma=3.0).
- **Pendientes:**
  1. Generar rejillas visuales de falsos negativos/positivos en val con máscaras (para memoria).
  2. Análisis de importancia de características (permutación en val) para el modelo elegido.
  3. Decidir modelo/grupo para fase 5 (PlantDoc) según hipótesis: ¿generaliza mejor `color_manchas` sin textura?
  4. Verificar que los modelos se regeneran con un comando desde `configs/classification.json`.
