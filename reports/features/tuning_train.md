# Ajuste de características con train

`python scripts/tune_features.py --seed 42` — 1406 imágenes `train` del subconjunto. Nada de `val` ni `test`.

## Variantes de máscara frente a sombras (D019)

Criterio fijado de antemano (`docs/tasks/T04-caracteristicas.md`): AUC de la fracción de píxeles oscuros (V < 60) entre sanas y late/early blight; desempate por simplicidad salvo mejora ≥ 0.01. `dark_px_removed_blight`: fracción media de píxeles oscuros de la máscara (a) que la variante quita en hojas de blight (pérdida de lesión). Las columnas `posthoc_auc_blight_v30/v15` repiten el criterio con V < 30 y V < 15: son un **análisis de sensibilidad posterior**, no intervienen en la elección.

| variant | auc_healthy_vs_blight | auc_healthy_vs_diseased | median_dark_healthy | median_dark_blight | dark_px_removed_blight | posthoc_auc_blight_v30 | posthoc_auc_blight_v15 |
|---|---|---|---|---|---|---|---|
| {"kind": "raw"} | 0.4802 | 0.366 | 0.1785 | 0.1728 | 0.0 | 0.3475 | 0.2988 |
| {"kind": "erode", "n": 2} | 0.4754 | 0.3717 | 0.1657 | 0.1604 | 0.16 | 0.3716 | 0.3211 |
| {"kind": "erode", "n": 4} | 0.4732 | 0.3764 | 0.16 | 0.1588 | 0.248 | 0.3786 | 0.3339 |
| {"kind": "erode", "n": 6} | 0.4764 | 0.3845 | 0.1561 | 0.1586 | 0.3118 | 0.3839 | 0.3398 |
| {"kind": "erode", "n": 8} | 0.484 | 0.3952 | 0.1517 | 0.1549 | 0.3732 | 0.3897 | 0.3456 |
| {"kind": "shadow", "v_max": 50, "s_max": 40} | 0.4803 | 0.3662 | 0.1762 | 0.1727 | 0.0035 | 0.3471 | 0.2985 |
| {"kind": "shadow", "v_max": 50, "s_max": 60} | 0.482 | 0.3682 | 0.1753 | 0.1724 | 0.0102 | 0.3477 | 0.2985 |
| {"kind": "shadow", "v_max": 50, "s_max": 80} | 0.4893 | 0.375 | 0.1704 | 0.1719 | 0.0214 | 0.3516 | 0.2983 |
| {"kind": "shadow", "v_max": 70, "s_max": 40} | 0.4802 | 0.3662 | 0.1761 | 0.1727 | 0.0045 | 0.3471 | 0.2985 |
| {"kind": "shadow", "v_max": 70, "s_max": 60} | 0.4811 | 0.3678 | 0.1752 | 0.1722 | 0.0156 | 0.3484 | 0.2989 |
| {"kind": "shadow", "v_max": 70, "s_max": 80} | 0.4846 | 0.3725 | 0.1693 | 0.1648 | 0.0402 | 0.3527 | 0.2996 |
| {"kind": "shadow", "v_max": 90, "s_max": 40} | 0.4804 | 0.3662 | 0.1763 | 0.1728 | 0.0048 | 0.3474 | 0.2987 |
| {"kind": "shadow", "v_max": 90, "s_max": 60} | 0.4822 | 0.3683 | 0.1755 | 0.1725 | 0.0189 | 0.3497 | 0.2998 |
| {"kind": "shadow", "v_max": 90, "s_max": 80} | 0.4886 | 0.3763 | 0.1696 | 0.1697 | 0.048 | 0.3563 | 0.303 |

**Variante elegida:** `{"kind": "raw"}`. Rejilla: `shadow_variants_train.jpg` (filas 1–4: sanas con más oscuro en (a); filas 5–8: blight con más oscuro en el borde; columnas: mejor (a), (b) y (c)).

## Umbrales de tonos (píxeles de hojas sanas)

`{"v_dark": 15, "s_min": 40, "v_hue_min": 50, "h_brown_yellow": 20, "h_green_min": 29}` (H en escala OpenCV 0–179). Rangos de histograma a*/b*: `{"a": [-27.0, 5.0], "b": [1.0, 48.0]}`.

## Textura (media de |AUC − 0,5| de la familia)

| family | params | mean_abs_auc |
|---|---|---|
| glcm | {"glcm_levels": 32, "glcm_distance": 1} | 0.4286 |
| glcm | {"glcm_levels": 32, "glcm_distance": 2} | 0.4205 |
| glcm | {"glcm_levels": 32, "glcm_distance": 4} | 0.4052 |
| lbp | {"lbp_points": 8, "lbp_radius": 1} | 0.3869 |
| lbp | {"lbp_points": 16, "lbp_radius": 2} | 0.397 |
| lbp | {"lbp_points": 24, "lbp_radius": 3} | 0.3805 |

Elegido: `{"glcm_levels": 32, "glcm_distance": 1, "lbp_points": 16, "lbp_radius": 2}`.

## Criterio «no concluyente» (D017)

Reglas candidatas (p1 / p99 de train): `{"leaf_area_frac_min": 0.2154, "mask_solidity_min": 0.6504, "border_touch_frac_max": 0.1627}`. Con las tres juntas se marcarían 41 de 1406 imágenes de train ({"leaf_area_frac < min": 15, "mask_solidity < min": 14, "border_touch_frac > max": 16}).

Revisión visual (`inconclusive_train.jpg`, un bloque por regla): las reglas de área y de borde marcan sobre todo hojas bien segmentadas (pequeñas o en primer plano); la de solidez recoge la mayoría de las máscaras fragmentadas. **Regla activa por defecto:** `{"mask_solidity_min": 0.6504}` → **14** imágenes de train no concluyentes (por etiqueta: {'diseased': 14}).
