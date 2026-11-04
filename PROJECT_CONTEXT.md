# Plant Disease Detection — Contexto del proyecto para Claude Code / Codex

> **Estado:** definición y preparación de datos. Todavía no hay un modelo entrenado ni resultados experimentales.
> **Idioma de trabajo:** español para explicaciones, documentación y memoria; nombres de variables, módulos y funciones en inglés cuando resulte natural.
> **Entorno previsto:** Python + OpenCV en VS Code (preferentemente en un entorno virtual). Repositorio Git.

## 1. Contexto académico

Proyecto **individual** de la asignatura *Fundamentos / Introducción a la Visión por Computador* de la Universidad de Deusto.

Según el enunciado (`proyecto.pdf`), hay que:
- Resolver un problema del mundo real mediante **visión por computador clásica**, demostrando técnicas de bajo y medio nivel (procesamiento, segmentación, extracción de características, clasificación y, si procede, seguimiento de objetos).
- Entregar **código funcional y comentado** y una **memoria** que explique el proyecto.
- Incluir definición y motivación del problema, **3–5 referencias relevantes**, explicación del valor añadido y limitaciones de soluciones existentes.
- Describir metodología, algoritmos, librerías, datos y condiciones de adquisición.
- Presentar métricas, analizar fallos y comentar posibles mejoras, con bibliografía.
- Registrar el tema en el Excel compartido para su validación; no pueden repetirse temas.

Las clases cubren: imágenes digitales, espacios de color e histogramas; transformaciones, filtrado y umbralización; detección de bordes; segmentación por contornos, regiones, watershed y clustering; descriptores globales/locales (color, textura, forma, HOG, SIFT/ORB); clasificadores clásicos (kNN, SVM, árboles de decisión); y técnicas de vídeo/tracking. El manual de prácticas usa OpenCV.

**Restricción no negociable del proyecto:** no usar redes neuronales, CNN, transformers, modelos de deep learning ni modelos visuales preentrenados para ninguna etapa del sistema. Sí se permiten algoritmos tradicionales de aprendizaje automático que trabajen sobre **características diseñadas y extraídas explícitamente**, como SVM, kNN o árboles de decisión/Random Forest. El núcleo debe ser comprensible y justificable desde la visión por computador clásica.

## 2. Idea y objetivo

**Título provisional:** *Detección de posibles enfermedades en hojas de plantas mediante visión por computador tradicional*.

Desarrollar una aplicación a la que se entregue una **fotografía** de una hoja y que determine si visualmente parece **sana o con posibles síntomas de enfermedad**, mostrando las evidencias visuales que han influido en el resultado. Queremos estudiar después el análisis de **vídeo** o cámara, si el prototipo con fotografías funciona adecuadamente.

Objetivo inicial realista:
- **Clasificación binaria:** `healthy` / `diseased` en hojas de **tomate**.
- Evidencias visuales: cambios de color, manchas, textura, bordes y propiedades de regiones.
- Poder mostrar máscaras, regiones sospechosas, estadísticas y resultados intermedios.
- Comparar varios métodos tradicionales y evaluar su generalización.

**No prometer** un diagnóstico clínico/fitopatológico universal, identificar cualquier especie o cualquier enfermedad, ni inferir que toda hoja amarilla está enferma. Se buscan **síntomas visibles compatibles**; factores como iluminación, edad de la hoja, plagas, carencias nutricionales y daños mecánicos pueden producir efectos parecidos. Si no se detecta una hoja con fiabilidad o la imagen es inadecuada, el sistema debería poder devolver *análisis no concluyente*.

## 3. Alcance por prioridades

### MVP (obligatorio)
1. Cargar imágenes y preparar un conjunto de datos **reproducible**.
2. Preprocesar y, cuando sea viable, **segmentar la hoja respecto al fondo**.
3. Extraer características interpretables de **color, textura y regiones/manchas**.
4. Entrenar/evaluar al menos un **clasificador clásico**, preferentemente SVM, y compararlo con un baseline basado en reglas y/o kNN.
5. Mostrar predicción, indicadores visuales, alguna representación de las zonas sospechosas y métricas de evaluación.
6. Documentar el diseño, los experimentos y los fallos.

### Ampliaciones condicionadas a resultados y tiempo
- Añadir **patata** y **pimiento** (primero clasificación sana/enferma por especie).
- Distinguir **enfermedades concretas**, solo si hay evidencia de que el sistema binario funciona.
- Añadir **vídeo**, procesamiento periódico de frames, seguimiento tradicional (MeanShift/CamShift/KLT o flujo óptico) y suavizado temporal de resultados.
- Desarrollar una interfaz cómoda, por ejemplo con **Streamlit**.

**Prioridad:** un clasificador de imágenes sólido y evaluado antes que una interfaz vistosa o múltiples funcionalidades incompletas.

## 4. Datasets discutidos

### Fuente principal preferida
- **PlantVillage original:** https://github.com/spMohanty/PlantVillage-Dataset
  - Preferido frente a versiones aumentadas, para controlar duplicados, imágenes relacionadas y particiones.
  - Revisar las modalidades disponibles: imágenes originales, posibles versiones segmentadas y metadatos/agrupaciones por hoja. **Verificar** la estructura y particiones reales antes de asumir que existen determinados campos.
  - Las máscaras o imágenes ya segmentadas, si las hubiera, pueden ayudar a **evaluar** nuestra segmentación, pero no deben reemplazar el desarrollo de una segmentación propia en el pipeline final.

### Fuentes adicionales
- **New Plant Diseases (Kaggle; PlantVillage ampliado):** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data
  - Alternativa, pero vigilar que contiene imágenes aumentadas y puede solaparse con PlantVillage original.
- **Crop Disease Dataset (Kaggle):** https://www.kaggle.com/datasets/muhammadumairjatt123/crop-disease-dataset-with-46-classes/data
  - **Pendiente de auditar**: verificar clases, etiquetas, origen, duplicados y utilidad real antes de integrarlo.
- **PlantDoc:** https://github.com/pratikkayal/PlantDoc-Dataset
  - Usarlo preferentemente como **evaluación externa** con fondos, iluminación y condiciones más naturales; comprobar clases y compatibilidad de las etiquetas.
- **Fotos propias con móvil:** pequeño conjunto de pruebas cualitativas, y cuantitativas solo si tienen etiquetas fiables.
- **Notebook PyTorch consultado:** https://www.kaggle.com/code/imtkaggleteam/plant-diseases-detection-pytorch/input
  - Referencia opcional sobre organización de datos/experimentos, **no reutilizar sus redes neuronales**.

### Especies propuestas
1. **Tomate**: primera especie y objetivo del MVP. Unificar distintas enfermedades en `diseased` y mantener la etiqueta de enfermedad original en metadatos.
2. **Patata**: ampliación posterior.
3. **Pimiento**: ampliación posterior.

No hace falta buscar más datasets por defecto. Buscar datos extra cuando se observe un problema concreto: baja variedad, clases poco representadas, poca presencia de fondos naturales, mala generalización, etc. **Calidad y diversidad > cantidad bruta**.

## 5. Preparación de los datos (fase actual)

### Reglas de calidad y prevención de fugas
- **Auditar antes de entrenar:** carpetas, clases, recuentos, extensiones, imágenes ilegibles, tamaños, etiquetas y ejemplos visuales.
- No suponer que todas las carpetas significan lo mismo en datasets distintos. Normalizar etiquetas mediante un **mapeo explícito y revisable**.
- Conservar: `species`, `disease_name` original, `binary_label`, `source`, `image_path`, identificador/grupo de hoja si se proporciona y `split`.
- Controlar **duplicados exactos y casi duplicados**. Si distintas transformaciones proceden de una misma foto/hoja, deben pertenecer a la **misma partición**.
- **Nunca** dejar que versiones aumentadas de una foto original aparezcan a la vez en entrenamiento y validación/test.
- Priorizar una partición oficial por grupo/hoja si existe. Si no, crear particiones reproducibles con semilla fija, atendiendo a grupos y clases, y documentar el riesgo residual de duplicados.
- Separar `train`, `val` y `test` antes de ajustar parámetros y aplicar aumentos **solo a train**. No ajustar decisiones usando el test.
- Mantener **PlantDoc como test externo** inicialmente. No mezclarlo sin criterio con el entrenamiento.
- Equilibrar razonablemente las clases, especialmente porque varias enfermedades distintas se agregan a `diseased`. Registrar la distribución original y la usada; evitar que el modelo aprenda únicamente la enfermedad mayoritaria.
- El primer experimento puede comenzar con un subconjunto manejable, por ejemplo **~500 imágenes sanas y ~500 enfermas de tomate**, elegidas cuidadosamente y con diversidad entre enfermedades. **Es un objetivo orientativo**, no una restricción fija ni una cifra ya descargada.
- No modificar/destruir los datos originales: `data/raw/` debe permanecer intacto.
- No subir datasets grandes ni archivos derivados pesados al repositorio Git.

### Metadatos propuestos

`data/metadata.csv` (o varios manifiestos por split), con campos como:

| campo | significado |
|---|---|
| `image_id` | identificador estable |
| `source` | `plantvillage`, `plantdoc`, `own`, etc. |
| `image_path` | ruta relativa al proyecto |
| `species` | especie normalizada |
| `disease_name` | etiqueta original normalizada, si procede |
| `binary_label` | `healthy` / `diseased` |
| `leaf_group_id` | grupo de hoja/foto original si se conoce; vacío si no |
| `split` | `train`, `val`, `test`, `external_test` |

Guardar también scripts y configuraciones para **regenerar** estos ficheros, en lugar de editarlos manualmente.

## 6. Pipeline técnico previsto (a validar experimentalmente)

```text
Imagen / frame
  -> control de calidad y normalización de resolución
  -> preprocesamiento (suavizado, contraste/iluminación si conviene)
  -> segmentación clásica de hoja / ROI
  -> extracción de características explícitas
       - color: HSV, histogramas, proporción de tonos anómalos
       - textura: GLCM, LBP y/o estadísticas locales
       - geometría/regiones: área, contornos, manchas, irregularidad
  -> escalado de características (ajustado solo con train)
  -> clasificación tradicional (SVM / kNN / árbol o Random Forest)
  -> resultado interpretable + representación de regiones sospechosas
```

Consideraciones:
- **No asumir que el verde = sano ni que un color marrón = enfermedad.** Contrastar grupos de características y combinaciones.
- Una segmentación por umbral de verde puede **perder áreas enfermas marrones/amarillas**: medirlo y probar alternativas clásicas (por ejemplo, GrabCut con ROI inicial, contornos o segmentación por regiones). No ocultar fallos.
- Una máscara de lesión no tiene por qué coincidir con la predicción del clasificador: son salidas distintas. No inventar una localización exacta de enfermedad a partir de una simple decisión global del SVM.
- Porcentajes como “80% enferma” **no son probabilidades fiables por defecto**. Si se desean mostrar probabilidades, calibrarlas y evaluarlas con datos de validación independientes; si no, comunicar una puntuación/confianza no calibrada sin presentarla como probabilidad clínica.
- Si se desarrolla vídeo, reutilizar el pipeline por imagen y añadir seguimiento/estabilidad temporal **sin redes neuronales**.

## 7. Diseño de experimentos y evaluación

Comparar, como mínimo:
1. Baseline simple por reglas de color/manchas.
2. Clasificador clásico con características de color.
3. Clasificador clásico con características de textura.
4. Clasificador clásico con características combinadas.

Métricas: matriz de confusión, **precision, recall, F1**, accuracy y tiempo de procesamiento por imagen; opcionalmente ROC-AUC cuando corresponda. Analizar especialmente falsos negativos (hojas enfermas clasificadas como sanas) y falsos positivos. Evitar conclusiones basadas solo en accuracy en datasets desequilibrados.

Evaluar por separado:
- **Test interno:** imágenes inéditas, particionadas correctamente dentro de la fuente principal.
- **Test externo:** imágenes naturales compatibles (PlantDoc y, si se consigue, fotos propias etiquetadas), sin haberlas usado para entrenar o ajustar parámetros.

Guardar parámetros, semillas, versiones de datos/modelos, tablas de resultados, gráficos y ejemplos de errores para la memoria. No afirmar precisión o robustez antes de medirlas.

## 8. Stack y organización inicial

**Librerías previstas:** Python 3, `opencv-contrib-python`, `numpy`, `pandas`, `scikit-image`, `scikit-learn`, `matplotlib`; `streamlit` solo cuando se construya la interfaz. No instalar PyTorch, TensorFlow ni dependencias neuronales.

Estructura propuesta (**ajustable si hay una razón clara**):

```text
plant-disease-detection/
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               # no versionar
│   │   ├── plantvillage/
│   │   └── plantdoc/
│   ├── processed/         # regenerable, no versionar archivos pesados
│   └── metadata.csv       # generado por scripts
├── scripts/
│   ├── audit_dataset.py
│   └── prepare_dataset.py
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── training.py
│   └── inference.py
├── experiments/
├── tests/
└── app.py                # interfaz, fase posterior
```

En la fase actual no es necesario implementar todos estos ficheros. Preferir módulos pequeños, funciones documentadas, opciones por CLI o archivo de configuración, pruebas básicas y mensajes de error útiles.

## 9. Forma de colaborar con Claude Code / Codex

- **Explicar primero** qué se va a hacer y por qué encaja con las técnicas del curso.
- Trabajar **por fases**, con resultados verificables. No construir toda la aplicación de golpe.
- Antes de escribir código, inspeccionar el repositorio y preguntar solo por información que realmente falte (p. ej., ubicación de datasets no descargados).
- No fingir que se han descargado datasets, ejecutado scripts o obtenido métricas si no ha ocurrido.
- Si se requieren credenciales de Kaggle, indicar el procedimiento; no guardar secretos en el repositorio.
- Priorizar código legible y razonado frente a soluciones opacas. Facilitar que el alumno entienda, modifique y justifique cada algoritmo.
- Evitar introducir redes neuronales, modelos preentrenados, servicios externos de diagnóstico o bibliotecas que oculten el razonamiento central.
- Mantener el progreso reproducible y documentado en README.

## 10. Primera tarea solicitada a Claude

**Empezar únicamente por la preparación y auditoría de datos de tomate.**

1. Inspeccionar el repositorio y confirmar si están los datasets descargados y cómo se organizan realmente.
2. Si no están, dejar claros los enlaces, pasos de descarga y rutas esperadas, **sin inventar contenido ni estructuras**.
3. Implementar `scripts/audit_dataset.py` para recorrer las carpetas y generar un informe de clases, recuentos, dimensiones, imágenes corruptas y ejemplos visuales.
4. Diseñar y, después, implementar `scripts/prepare_dataset.py` para normalizar etiquetas, crear un manifiesto y establecer splits reproducibles **sin leakage**.
5. Preparar un subconjunto inicial de tomate **sana/enferma**; conservar información sobre las enfermedades originales.
6. Probar lo que sea posible con los datos presentes y mostrar comandos de ejecución y resultados reales.
7. **Detenerse tras esta fase**, explicar qué se ha comprobado y qué decisiones de datos faltan antes de empezar segmentación o entrenamiento.

**Criterio de éxito de la primera fase:** dataset auditado, etiquetas y particiones trazables, scripts reproducibles y ninguna fuga evidente entre entrenamiento, validación y test. Todavía no hace falta desarrollar clasificador, vídeo ni interfaz.

## 11. Pendientes / decisiones abiertas

- Confirmación del profesor y disponibilidad del tema en el Excel compartido.
- Elección definitiva entre PlantVillage original y las variantes de Kaggle tras auditar sus archivos reales.
- Compatibilidad de etiquetas y duplicados de PlantDoc y otros datasets.
- Evaluación de si la segmentación automatizada es suficiente en fotos naturales o requiere seleccionar manualmente una ROI.
- Qué clasificadores y características concretas dan mejores resultados: decidir por experimentación, no por intuición.
- Inclusión de patata, pimiento, diagnóstico multicategoría y vídeo, condicionada al tiempo y rendimiento.

---

**Resumen operativo:** crear un detector **sana/enferma** de hojas de **tomate** usando exclusivamente **visión clásica + características manuales + clasificación tradicional**, con evaluación rigurosa y explicaciones visuales. Empezar auditando y preparando PlantVillage, reservar imágenes naturales para evaluación externa y ampliar el alcance solo cuando el MVP funcione.
