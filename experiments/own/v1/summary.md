# T07 — Test con fotos propias (ejecución única)

- **Ejecución:** `python scripts/eval_own.py --frozen-commit 2b6a416` (2026-11-21), una sola vez.
  Registro en `execution_2b6a416729fd92166e5e6266d7f7c5bd804545c4.json` (SHA-1 de los CSV, de
  `configs/own.json` y de las predicciones). Código congelado en `fba1cac`; entre `fba1cac` y
  `2b6a416` solo entraron `.gitignore` y los dos CSV del alumno.
- **Regenerable:** `python scripts/report_own.py` (tablas e hipótesis, `tables.md`,
  `metrics.json`) y `python scripts/plot_own.py` (rejillas y `contributions_B_roi.csv`; antes de
  dibujar comprueba que reproduce cada predicción guardada).
- **Datos:** 10 hojas compuestas sanas (`c01`–`c06`, `h01`–`h04`), cada una en los tres fondos
  (30 fotos, 97 folíolos con ROI) y una hoja alterada (`a01`, 3 fotos, 3 ROI). Notas: `residuo` en
  las 33 fotos, `compuesta` en 18, `gotas` en 6, `fondo_incompleto` en ninguna.
- **Protocolo:** fijado antes de procesar las fotos (D040–D043). Fallo = no concluyente o FP, sobre
  todas las unidades; tasas = media de las tasas por hoja; IC por bootstrap de hojas.

## Resultado principal

**Con fotos reales de móvil, el sistema no sirve para hojas sanas.** La variante B marca como
`diseased` 95 de los 97 folíolos sanos en modo ROI, y en modo automático el 80 % de las fotos son
no concluyentes y el resto, FP. Es coherente con el colapso en PlantDoc (D034) y con el límite de
validez de D035.

## Tasas por fondo × modo (sanas; media por hoja)

| Variante | Modo | Fondo | Hojas | Unidades | Fallo | No concl. | FP / concl. | Cobertura |
|---|---|---|---|---|---|---|---|---|
| B | auto | blanco | 10 | 10 | 1,000 | 0,800 | 1,000 | 0,200 |
| B | auto | gris oscuro | 10 | 10 | 1,000 | 0,900 | 1,000 | 0,100 |
| B | auto | natural | 10 | 10 | 1,000 | 0,700 | 1,000 | 0,300 |
| B | auto | todos | 10 | 30 | 1,000 | 0,800 | 1,000 | 0,200 |
| B | ROI | blanco | 10 | 33 | 1,000 | 0,000 | 1,000 | 1,000 |
| B | ROI | gris oscuro | 10 | 32 | 0,986 | 0,000 | 0,986 | 1,000 |
| B | ROI | natural | 10 | 32 | 0,986 | 0,034 | 0,983 | 0,966 |
| B | ROI | todos | 10 | 97 | 0,990 | 0,011 | 0,990 | 0,989 |
| A (ref.) | auto | todos | 10 | 30 | 0,933 | 0,800 | 0,667 | 0,200 |
| A (ref.) | ROI | todos | 10 | 97 | 0,967 | 0,011 | 0,967 | 0,989 |

Tabla completa (A por fondo y estratificación por notas) en `tables.md`. «FP / concl.» en el modo
automático sale de muy pocas hojas con alguna foto concluyente (cobertura 0,1–0,3).

## Veredicto H4–H6 (variante B, 10 hojas, 10 000 réplicas, semilla 42)

| | Comparación | Diferencia (peor − mejor) | IC 95 % | Unidades | Veredicto |
|---|---|---|---|---|---|
| H4 | ROI: natural − gris oscuro | 0,000 | [0,000, 0,000] | 64 | no concluyente |
| H5 | ROI: blanco − gris oscuro | 0,014 | [0,000, 0,043] | 65 | no concluyente |
| H6 | auto − ROI | 0,010 | [0,000, 0,029] | 127 | no concluyente |

Las tres son **no concluyentes por efecto techo**: la tasa de fallo vale ~1 en todos los fondos y
modos, así que no queda margen para que un fondo o un modo sea peor que otro. No es evidencia de
que el fondo o la ROI no importen; es que con este clasificador todo falla. D037 sigue **sin
verificar**.

## Por qué falla (descriptivo; no se ajusta nada)

- **La segmentación no es el problema en modo ROI.** En las rejillas, los contornos se ajustan a los
  folíolos; la mediana de `leaf_area_frac` en el recorte es 0,42 (dentro de rango) y solo 2 de 97
  folíolos quedan no concluyentes (`c02_natural f3` y `c06_natural f5`, con máscara casi vacía).
- **En modo automático sí lo es:** la hoja compuesta ocupa poco encuadre (mediana de
  `leaf_area_frac` 0,15) y hay varias hojas, tierra o la mano en la foto. Motivos de no concluyente
  en B, sobre las 27 fotos no concluyentes (incluidas las 3 de la alterada): `leaf_area_frac` 17,
  `mask_solidity` 9, `border_touch_frac` 2 (una foto puede tener varios).
- **El clasificador recibe colores fuera del rango de PlantVillage.** Contribución media de B sobre
  los 95 folíolos sanos concluyentes (`contributions_B_roi.csv`):

  | Característica | Contribución media | z medio |
  |---|---|---|
  | `b_hist_07` (último bin de b*) | +3,10 | +6,3 |
  | `l_mean` | −2,62 | +2,7 |
  | `v_mean` | −2,40 | +3,2 |
  | `a_hist_00` (primer bin de a*, el más verde) | +1,86 | +6,5 |
  | `v_std` | +1,80 | −2,2 |
  | `b_mean` | +1,73 | +4,2 |

  Ejemplo `c01_gris f1`: b* medio 63,6, por encima del rango del histograma ajustado con train
  ([1, 48]); el 97,6 % de los píxeles cae en el último bin (`b_hist_07`, z = +11,0). Las hojas
  fotografiadas con el móvil son mucho más amarillo-verdosas y saturadas que las de PlantVillage.
  El mismo coeficiente que en PlantVillage separa «más amarillo = más enfermo» empuja aquí a sanas
  hacia `diseased`. Las manchas casi no intervienen (`spot_area_frac` mediano 0,000).
  Esto ilustra la regla 8 de `AGENTS.md`: no se puede asumir que amarillo = enfermo.
- **Exposición según el fondo (descriptivo, sin veredicto):** la puntuación media por hoja es más
  alta sobre blanco (mediana 11,6) que sobre natural (7,6) y gris oscuro (4,1); 9 de 10 hojas
  puntúan más sobre blanco que sobre gris. Va en el sentido de D037, pero la puntuación **no es la
  medida preregistrada** y no se le da veredicto. En las rejillas, sobre la cartulina blanca las hojas
  salen más oscuras (la cámara compensa la exposición).
- **Notas:** `residuo` está en todas las fotos, así que no se puede comparar con/sin. Por unidad
  (no por hoja), en B-ROI: con `gotas` 34/36 fallos y sin ellas 61/61; con `compuesta` 83/85 y sin
  ella 12/12. Diferencias de una o dos unidades: sin lectura.

## Hoja alterada (`a01`, descriptivo)

| Foto | Modo automático | Modo ROI (B) | Modo ROI (A) |
|---|---|---|---|
| `a01_blanco` | no concluyente | `diseased`, +15,9 | `diseased`, +5,6 |
| `a01_gris` | no concluyente | no concluyente (`mask_solidity` 0,965 > 0,961) | no concluyente |
| `a01_natural` | no concluyente | `diseased`, +2,4 | `healthy`, −0,46 |

Es una hoja amarillenta, no enferma. Como las sanas también se predicen enfermas, el sistema no
distingue «alterada» de «sana»: no se puede leer nada de su predicción.

## Figuras

Todas contienen fotos propias, así que están en `data/processed/own_v1/` (fuera de git) hasta que
el alumno decida si se versionan:

- `grid_photos_{gris_oscuro,blanco,natural}.jpg`: cada foto con sus ROI coloreadas por resultado
  (verde acierto, rojo FP, gris no concluyente) y la máscara del modo automático.
- `grid_rois.jpg`: los 100 recortes con su máscara, resultado y puntuación.
- `examples/` (`predict.py`), candidatas para la memoria: `c01_gris_auto` (no concluyente,
  `mask_solidity`), `c01_gris_roi` (FP con la contribución de `b_hist_07`), `c06_natural_roi`
  (`f2`, uno de los dos aciertos, −2,4) y `a01_blanco_roi` (hoja alterada, +15,9).

## Limitaciones

- 10 hojas de una sola planta, un móvil y una sesión: no generaliza a otras condiciones.
- Todas las hojas tienen residuo blanco de tratamiento: es un factor de confusión que no se puede
  separar del cambio de cámara con estos datos.
- Con el efecto techo, H4–H6 no tienen potencia; la ausencia de veredicto no es ausencia de efecto.
