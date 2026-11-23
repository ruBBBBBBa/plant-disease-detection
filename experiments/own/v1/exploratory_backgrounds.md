# T07 — Análisis exploratorio a posteriori: salida de B según el fondo

> **Exploratorio y a posteriori** (D050): no cambia ningún veredicto (H4–H6 siguen «no
> concluyentes»), umbral ni modelo. Solo usa `predictions.csv.gz` de la ejecución única;
> no se han vuelto a procesar las fotos. Regenerable con
> `python scripts/explore_own_backgrounds.py`.

## 1. ¿Se pueden emparejar los folíolos entre fondos?

`leaflet_id` debía nombrar el mismo folíolo en las tres fotos de una hoja, pero eso
no se puede verificar sin mirar las fotos. Coherencia geométrica (ROI por fondo en
orden blanco/gris oscuro/natural; ρ de Spearman del área relativa de cada ROI frente
a gris oscuro):

```
leaf_id rois_por_fondo  rho_blanco  rho_natural  emparejable
    c01          6/5/5        0.30         0.90        False
    c02          5/5/5        1.00         0.90        False
    c03          3/3/3        0.50         0.50        False
    c04          3/3/3        1.00         1.00        False
    c05          5/5/5        0.70         0.70        False
    c06          7/7/7        0.79         0.36        False
    h01          1/1/1         NaN          NaN         True
    h02          1/1/1         NaN          NaN         True
    h03          1/1/1         NaN          NaN         True
    h04          1/1/1         NaN          NaN         True
```

**Conclusión:** el emparejamiento por `leaflet_id` **no es fiable en todas las hojas**
(p. ej. `c01` tiene una ROI más en blanco y áreas que no cuadran). No se fuerza: por
folíolo solo se analizan las hojas con una ROI por foto; el resto, por hoja.

## 2. Acuerdo y variación de la puntuación

| Nivel | Unidades | Mismo resultado en los 3 fondos | DT dentro (entre fondos) | DT entre unidades | ICC(1) |
|---|---|---|---|---|---|
| Hoja (mayoría de folíolos; puntuación media) | 10 | 10/10 | 3.27 | 1.83 | -0.02 |
| Folíolo (solo hojas con una ROI por foto) | 4 | 4/4 | 3.62 | 0.51 | -0.46 |

**Solo un indicio.** Con n = 10 hojas (4 folíolos) y la compresión distinta de `h01`,
el ICC no permite concluir nada; además trata el fondo como ruido aleatorio, cuando
puede ser un efecto sistemático.

Puntuación media de B por hoja y fondo (folíolos concluyentes) y fondo con la puntuación
más alta en cada hoja (descriptivo, sin veredicto):

```
background  blanco  gris_oscuro  natural fondo_max
leaf_id                                           
c01          11.48         4.14     7.21    blanco
c02           6.53         6.69     9.39   natural
c03          12.82         4.93     7.46    blanco
c04          12.32         3.96     7.80    blanco
c05           4.60         4.15     4.67   natural
c06           5.68         2.42     1.65    blanco
h01          11.69         3.20     7.38    blanco
h02          12.38         4.03     8.44    blanco
h03          11.88         5.69     7.94    blanco
h04          11.07         5.70     8.73    blanco
```

Recuento del fondo con la puntuación más alta: blanco 8, natural 2 (de 10 hojas).

Folíolos emparejables (resultado y puntuación):

```
           resultado                       puntuación                    
background    blanco gris_oscuro   natural     blanco gris_oscuro natural
leaf_id                                                                  
h01         diseased    diseased  diseased      11.69        3.20    7.38
h02         diseased    diseased  diseased      12.38        4.03    8.44
h03         diseased    diseased  diseased      11.88        5.69    7.94
h04         diseased    diseased  diseased      11.07        5.70    8.73
```

## 3. b* y `b_hist_07` por fondo

**No se puede calcular con lo guardado.** `contributions_B_roi.csv` solo tiene la
media de cada contribución sobre los 95 folíolos concluyentes, no los valores por
folíolo, así que no hay b* ni `b_hist_07` por fondo ni diferencias emparejadas.
Obtenerlos exigiría volver a procesar las fotos, que esta tarea no permite.
