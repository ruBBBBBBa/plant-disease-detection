# Fotos propias — tablas (generado por scripts/report_own.py)

Fallo = no concluyente o falso positivo, sobre todas las unidades. Tasas = media de las tasas por hoja compuesta.

## Por variante × modo × fondo (sanas)

```text
variant mode  background  n_leaves  n_units  fail_rate  inconclusive_rate  fp_rate_conclusive  coverage
      A auto      blanco        10       10      1.000              0.800               1.000     0.200
      A auto gris_oscuro        10       10      0.900              0.900               0.000     0.100
      A auto     natural        10       10      0.900              0.700               0.667     0.300
      A auto       todos        10       30      0.933              0.800               0.667     0.200
      A  roi      blanco        10       33      1.000              0.000               1.000     1.000
      A  roi gris_oscuro        10       32      0.900              0.000               0.900     1.000
      A  roi     natural        10       32      1.000              0.034               1.000     0.966
      A  roi       todos        10       97      0.967              0.011               0.967     0.989
      B auto      blanco        10       10      1.000              0.800               1.000     0.200
      B auto gris_oscuro        10       10      1.000              0.900               1.000     0.100
      B auto     natural        10       10      1.000              0.700               1.000     0.300
      B auto       todos        10       30      1.000              0.800               1.000     0.200
      B  roi      blanco        10       33      1.000              0.000               1.000     1.000
      B  roi gris_oscuro        10       32      0.986              0.000               0.986     1.000
      B  roi     natural        10       32      0.986              0.034               0.983     0.966
      B  roi       todos        10       97      0.990              0.011               0.990     0.989
```

## Hipótesis (variante B, bootstrap por hoja)

```text
   mode     compare    worse       better  diff             ci95 n_leaves n_units         verdict
H4  roi  background  natural  gris_oscuro 0.000       [0.0, 0.0]       10      64  no concluyente
H5  roi  background   blanco  gris_oscuro 0.014  [0.0, 0.042857]       10      65  no concluyente
H6  NaN        mode     auto          roi 0.010  [0.0, 0.028571]       10     127  no concluyente
```

## Estratificación por notas (descriptiva, sin veredicto)

```text
         keyword  present variant mode  n_leaves  n_units  fail_rate  inconclusive_rate
         residuo     True       A auto        10       30      0.933              0.800
         residuo    False       A auto         0        0        NaN                NaN
         residuo     True       A  roi        10       97      0.990              0.021
         residuo    False       A  roi         0        0        NaN                NaN
         residuo     True       B auto        10       30      1.000              0.800
         residuo    False       B auto         0        0        NaN                NaN
         residuo     True       B  roi        10       97      0.979              0.021
         residuo    False       B  roi         0        0        NaN                NaN
           gotas     True       A auto         2        6      1.000              1.000
           gotas    False       A auto         8       24      0.917              0.750
           gotas     True       A  roi         2       36      1.000              0.028
           gotas    False       A  roi         8       61      0.984              0.016
           gotas     True       B auto         2        6      1.000              1.000
           gotas    False       B auto         8       24      1.000              0.750
           gotas     True       B  roi         2       36      0.944              0.028
           gotas    False       B  roi         8       61      1.000              0.016
       compuesta     True       A auto         6       18      0.944              0.778
       compuesta    False       A auto         4       12      0.917              0.833
       compuesta     True       A  roi         6       85      1.000              0.024
       compuesta    False       A  roi         4       12      0.917              0.000
       compuesta     True       B auto         6       18      1.000              0.778
       compuesta    False       B auto         4       12      1.000              0.833
       compuesta     True       B  roi         6       85      0.976              0.024
       compuesta    False       B  roi         4       12      1.000              0.000
fondo_incompleto     True       A auto         0        0        NaN                NaN
fondo_incompleto    False       A auto        10       30      0.933              0.800
fondo_incompleto     True       A  roi         0        0        NaN                NaN
fondo_incompleto    False       A  roi        10       97      0.990              0.021
fondo_incompleto     True       B auto         0        0        NaN                NaN
fondo_incompleto    False       B auto        10       30      1.000              0.800
fondo_incompleto     True       B  roi         0        0        NaN                NaN
fondo_incompleto    False       B  roi        10       97      0.979              0.021
```

## Hojas alteradas (descriptivo, sin métricas de acierto)

```text
           file mode leaflet_id variant  inconclusive                                        reasons prediction                score
 a01_blanco.jpg auto                  B          True leaf_area_frac = 0.119 fuera de [0.180, 0.616]                                
 a01_blanco.jpg auto                  A          True leaf_area_frac = 0.119 fuera de [0.180, 0.616]                                
 a01_blanco.jpg  roi         f1       B         False                                                  diseased   15.941616579704453
 a01_blanco.jpg  roi         f1       A         False                                                  diseased   5.6487498787308015
   a01_gris.jpg auto                  B          True leaf_area_frac = 0.066 fuera de [0.180, 0.616]                                
   a01_gris.jpg auto                  A          True leaf_area_frac = 0.066 fuera de [0.180, 0.616]                                
   a01_gris.jpg  roi         f1       B          True  mask_solidity = 0.965 fuera de [0.593, 0.961]                                
   a01_gris.jpg  roi         f1       A          True  mask_solidity = 0.965 fuera de [0.593, 0.961]                                
a01_natural.jpg auto                  B          True leaf_area_frac = 0.017 fuera de [0.180, 0.616]                                
a01_natural.jpg auto                  A          True leaf_area_frac = 0.017 fuera de [0.180, 0.616]                                
a01_natural.jpg  roi         f1       B         False                                                  diseased      2.4150140764981
a01_natural.jpg  roi         f1       A         False                                                   healthy -0.45685929600852526
```
