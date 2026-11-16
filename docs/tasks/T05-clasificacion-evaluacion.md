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
- **Sigma del suavizado común y cómo se eligió:**
- **Resultados de CV (train) y val por grupo × modelo:**
- **Configuración elegida con val y por qué:**
- **Resultados de test (ejecución única, todo congelado):**
- **Análisis del sesgo de nitidez:**
- **Recall por enfermedad (test completo):**
- **Errores típicos:**
- **Decisiones tomadas durante la tarea:**
- **Pendientes:**
