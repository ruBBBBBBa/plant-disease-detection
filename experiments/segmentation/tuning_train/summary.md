# Ajuste de segmentación con train

`python scripts/tune_segmentation.py --sample-per-class 30 --seed 42` — 270 imágenes de `train` (30 por enfermedad).

- Umbral de la referencia (valle del histograma de max(B,G,R)): **22**.
- Canal de Otsu elegido: **ExG** (histogramas en `channel_histograms.png`).
- Núcleo del postprocesado: **3**.
- Criterio: media por imagen de (IoU + cobertura no verde) / 2 (solo IoU si no hay píxeles no verdes). Métodos sin postprocesado, salvo las filas `postprocess` (media de los 5 métodos con postprocesado).

## Mejor combinación por método

| method | params | mean_iou | mean_non_green_cov | score |
|---|---|---|---|---|
| grabcut | {"channel_name": "ExG", "iterations": 3} | 0.8824 | 0.6305 | 0.7564 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 30, "v_min": 20} | 0.8978 | 0.6271 | 0.7624 |
| kmeans | {"k": 3, "border_share": 0.3, "features": "ab", "seed": 42} | 0.9252 | 0.7485 | 0.8368 |
| otsu | {"channel_name": "ExG"} | 0.8985 | 0.4624 | 0.6805 |
| postprocess | {"kernel": 3} | 0.904 | 0.6369 | 0.7705 |
| watershed | {"channel_name": "ExG", "erode": 5, "dilate": 15} | 0.8956 | 0.5588 | 0.7272 |

## Todas las combinaciones

| method | params | mean_iou | mean_non_green_cov | score |
|---|---|---|---|---|
| otsu | {"channel_name": "S"} | 0.7 | 0.4073 | 0.5537 |
| otsu | {"channel_name": "a"} | 0.8882 | 0.366 | 0.6271 |
| otsu | {"channel_name": "L"} | 0.195 | 0.6199 | 0.4074 |
| otsu | {"channel_name": "ExG"} | 0.8985 | 0.4624 | 0.6805 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 30, "v_min": 20} | 0.8969 | 0.6229 | 0.7599 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 30, "v_min": 40} | 0.8776 | 0.6038 | 0.7407 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 50, "v_min": 20} | 0.8328 | 0.3312 | 0.582 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 50, "v_min": 40} | 0.8105 | 0.3129 | 0.5617 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 70, "v_min": 20} | 0.7019 | 0.2726 | 0.4873 |
| hsv_green | {"h_min": 20, "h_max": 85, "s_min": 70, "v_min": 40} | 0.6757 | 0.2547 | 0.4652 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 30, "v_min": 20} | 0.8978 | 0.6271 | 0.7624 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 30, "v_min": 40} | 0.8785 | 0.6078 | 0.7432 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 50, "v_min": 20} | 0.8334 | 0.3337 | 0.5835 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 50, "v_min": 40} | 0.8112 | 0.3154 | 0.5633 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 70, "v_min": 20} | 0.7022 | 0.2738 | 0.488 |
| hsv_green | {"h_min": 20, "h_max": 95, "s_min": 70, "v_min": 40} | 0.6761 | 0.2558 | 0.4659 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 30, "v_min": 20} | 0.8891 | 0.4922 | 0.6907 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 30, "v_min": 40} | 0.8652 | 0.4795 | 0.6724 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 50, "v_min": 20} | 0.8231 | 0.2198 | 0.5214 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 50, "v_min": 40} | 0.7971 | 0.2078 | 0.5024 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 70, "v_min": 20} | 0.6919 | 0.1839 | 0.4379 |
| hsv_green | {"h_min": 25, "h_max": 85, "s_min": 70, "v_min": 40} | 0.6629 | 0.1721 | 0.4175 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 30, "v_min": 20} | 0.89 | 0.4964 | 0.6932 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 30, "v_min": 40} | 0.8661 | 0.4836 | 0.6748 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 50, "v_min": 20} | 0.8237 | 0.2223 | 0.523 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 50, "v_min": 40} | 0.7978 | 0.2102 | 0.504 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 70, "v_min": 20} | 0.6923 | 0.185 | 0.4386 |
| hsv_green | {"h_min": 25, "h_max": 95, "s_min": 70, "v_min": 40} | 0.6633 | 0.1732 | 0.4182 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 30, "v_min": 20} | 0.8638 | 0.2409 | 0.5524 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 30, "v_min": 40} | 0.8377 | 0.2403 | 0.539 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 50, "v_min": 20} | 0.7983 | 0.0 | 0.3992 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 50, "v_min": 40} | 0.7707 | 0.0 | 0.3853 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 70, "v_min": 20} | 0.6689 | 0.0 | 0.3344 |
| hsv_green | {"h_min": 30, "h_max": 85, "s_min": 70, "v_min": 40} | 0.6389 | 0.0 | 0.3195 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 30, "v_min": 20} | 0.8646 | 0.2451 | 0.5549 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 30, "v_min": 40} | 0.8386 | 0.2444 | 0.5415 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 50, "v_min": 20} | 0.799 | 0.0025 | 0.4007 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 50, "v_min": 40} | 0.7714 | 0.0024 | 0.3869 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 70, "v_min": 20} | 0.6692 | 0.0012 | 0.3352 |
| hsv_green | {"h_min": 30, "h_max": 95, "s_min": 70, "v_min": 40} | 0.6393 | 0.0011 | 0.3202 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 30, "v_min": 20} | 0.8141 | 0.2175 | 0.5158 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 30, "v_min": 40} | 0.787 | 0.2169 | 0.5019 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 50, "v_min": 20} | 0.7511 | 0.0 | 0.3756 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 50, "v_min": 40} | 0.7228 | 0.0 | 0.3614 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 70, "v_min": 20} | 0.6251 | 0.0 | 0.3126 |
| hsv_green | {"h_min": 35, "h_max": 85, "s_min": 70, "v_min": 40} | 0.5953 | 0.0 | 0.2976 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 30, "v_min": 20} | 0.8149 | 0.2216 | 0.5183 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 30, "v_min": 40} | 0.7878 | 0.221 | 0.5044 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 50, "v_min": 20} | 0.7518 | 0.0025 | 0.3771 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 50, "v_min": 40} | 0.7235 | 0.0024 | 0.3629 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 70, "v_min": 20} | 0.6255 | 0.0012 | 0.3133 |
| hsv_green | {"h_min": 35, "h_max": 95, "s_min": 70, "v_min": 40} | 0.5956 | 0.0011 | 0.2984 |
| kmeans | {"k": 2, "border_share": 0.1, "features": "Lab", "seed": 42} | 0.3251 | 0.2263 | 0.2757 |
| kmeans | {"k": 2, "border_share": 0.1, "features": "ab", "seed": 42} | 0.8862 | 0.5149 | 0.7006 |
| kmeans | {"k": 2, "border_share": 0.2, "features": "Lab", "seed": 42} | 0.4216 | 0.2964 | 0.359 |
| kmeans | {"k": 2, "border_share": 0.2, "features": "ab", "seed": 42} | 0.9031 | 0.5251 | 0.7141 |
| kmeans | {"k": 2, "border_share": 0.3, "features": "Lab", "seed": 42} | 0.4633 | 0.3296 | 0.3965 |
| kmeans | {"k": 2, "border_share": 0.3, "features": "ab", "seed": 42} | 0.9064 | 0.526 | 0.7162 |
| kmeans | {"k": 2, "border_share": 0.4, "features": "Lab", "seed": 42} | 0.4831 | 0.3448 | 0.4139 |
| kmeans | {"k": 2, "border_share": 0.4, "features": "ab", "seed": 42} | 0.91 | 0.5268 | 0.7184 |
| kmeans | {"k": 2, "border_share": 0.5, "features": "Lab", "seed": 42} | 0.4882 | 0.3492 | 0.4187 |
| kmeans | {"k": 2, "border_share": 0.5, "features": "ab", "seed": 42} | 0.91 | 0.5268 | 0.7184 |
| kmeans | {"k": 3, "border_share": 0.1, "features": "Lab", "seed": 42} | 0.4304 | 0.2988 | 0.3646 |
| kmeans | {"k": 3, "border_share": 0.1, "features": "ab", "seed": 42} | 0.9149 | 0.7421 | 0.8285 |
| kmeans | {"k": 3, "border_share": 0.2, "features": "Lab", "seed": 42} | 0.4784 | 0.3439 | 0.4112 |
| kmeans | {"k": 3, "border_share": 0.2, "features": "ab", "seed": 42} | 0.9209 | 0.7469 | 0.8339 |
| kmeans | {"k": 3, "border_share": 0.3, "features": "Lab", "seed": 42} | 0.5162 | 0.3995 | 0.4579 |
| kmeans | {"k": 3, "border_share": 0.3, "features": "ab", "seed": 42} | 0.9252 | 0.7485 | 0.8368 |
| kmeans | {"k": 3, "border_share": 0.4, "features": "Lab", "seed": 42} | 0.5554 | 0.5086 | 0.532 |
| kmeans | {"k": 3, "border_share": 0.4, "features": "ab", "seed": 42} | 0.9252 | 0.7485 | 0.8368 |
| kmeans | {"k": 3, "border_share": 0.5, "features": "Lab", "seed": 42} | 0.563 | 0.6276 | 0.5953 |
| kmeans | {"k": 3, "border_share": 0.5, "features": "ab", "seed": 42} | 0.9252 | 0.7485 | 0.8368 |
| grabcut | {"channel_name": "ExG", "iterations": 3} | 0.8824 | 0.6305 | 0.7564 |
| grabcut | {"channel_name": "ExG", "iterations": 5} | 0.8779 | 0.631 | 0.7545 |
| watershed | {"channel_name": "ExG", "erode": 5, "dilate": 5} | 0.8837 | 0.4093 | 0.6465 |
| watershed | {"channel_name": "ExG", "erode": 5, "dilate": 9} | 0.8939 | 0.4975 | 0.6957 |
| watershed | {"channel_name": "ExG", "erode": 5, "dilate": 15} | 0.8956 | 0.5588 | 0.7272 |
| watershed | {"channel_name": "ExG", "erode": 9, "dilate": 5} | 0.8651 | 0.3677 | 0.6164 |
| watershed | {"channel_name": "ExG", "erode": 9, "dilate": 9} | 0.8793 | 0.4523 | 0.6658 |
| watershed | {"channel_name": "ExG", "erode": 9, "dilate": 15} | 0.8853 | 0.5129 | 0.6991 |
| watershed | {"channel_name": "ExG", "erode": 15, "dilate": 5} | 0.8364 | 0.3309 | 0.5837 |
| watershed | {"channel_name": "ExG", "erode": 15, "dilate": 9} | 0.8562 | 0.4094 | 0.6328 |
| watershed | {"channel_name": "ExG", "erode": 15, "dilate": 15} | 0.8655 | 0.4655 | 0.6655 |
| postprocess | {"kernel": 3} | 0.904 | 0.6369 | 0.7705 |
| postprocess | {"kernel": 5} | 0.9035 | 0.6306 | 0.767 |
| postprocess | {"kernel": 7} | 0.9011 | 0.6227 | 0.7619 |
