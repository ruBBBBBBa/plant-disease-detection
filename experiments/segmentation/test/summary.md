# Segmentación — evaluación en `test`

`python scripts/evaluate_segmentation.py --split test --seed 42 --postprocess both` — 296 imágenes. Parámetros: `configs/segmentation.json` (ajustados con train). Referencia: `segmented` con umbral 22 y relleno de huecos. Cobertura no verde: H fuera de [30, 90] o S < 40 dentro de la referencia (NaN si no hay píxeles no verdes).

`postprocess=True`: apertura + cierre (núcleo 3), mayor componente conexa y relleno de huecos. Rejillas `overlay_<método>.jpg`: variante con postprocesado; fila 1, 8 casos cerca de la mediana de IoU; fila 2, los 8 peores. Amarillo = referencia, rojo = método.

## Global

| method | postprocess | iou_mean | iou_median | iou_p10 | dice_mean | dice_median | dice_p10 | non_green_cov_mean | non_green_cov_median | non_green_cov_p10 |
|---|---|---|---|---|---|---|---|---|---|---|
| grabcut | False | 0.902 | 0.934 | 0.826 | 0.945 | 0.966 | 0.905 | 0.728 | 0.822 | 0.323 |
| grabcut | True | 0.903 | 0.934 | 0.823 | 0.945 | 0.966 | 0.903 | 0.734 | 0.826 | 0.319 |
| hsv_green | False | 0.9 | 0.927 | 0.82 | 0.945 | 0.962 | 0.901 | 0.636 | 0.659 | 0.449 |
| hsv_green | True | 0.923 | 0.948 | 0.855 | 0.958 | 0.973 | 0.922 | 0.728 | 0.767 | 0.501 |
| kmeans | False | 0.926 | 0.956 | 0.834 | 0.959 | 0.977 | 0.909 | 0.824 | 0.922 | 0.487 |
| kmeans | True | 0.932 | 0.965 | 0.834 | 0.962 | 0.982 | 0.909 | 0.859 | 0.936 | 0.618 |
| otsu | False | 0.922 | 0.955 | 0.858 | 0.956 | 0.977 | 0.923 | 0.62 | 0.756 | 0.134 |
| otsu | True | 0.931 | 0.963 | 0.883 | 0.961 | 0.981 | 0.938 | 0.672 | 0.784 | 0.187 |
| watershed | False | 0.911 | 0.937 | 0.872 | 0.95 | 0.968 | 0.932 | 0.679 | 0.761 | 0.281 |
| watershed | True | 0.914 | 0.939 | 0.873 | 0.952 | 0.969 | 0.932 | 0.689 | 0.773 | 0.282 |

## Por `binary_label`

| binary_label | method | postprocess | iou_mean | iou_median | iou_p10 | dice_mean | dice_median | dice_p10 | non_green_cov_mean | non_green_cov_median | non_green_cov_p10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diseased | grabcut | False | 0.883 | 0.93 | 0.772 | 0.931 | 0.964 | 0.872 | 0.598 | 0.667 | 0.158 |
| diseased | grabcut | True | 0.885 | 0.933 | 0.771 | 0.932 | 0.965 | 0.871 | 0.61 | 0.696 | 0.155 |
| diseased | hsv_green | False | 0.894 | 0.932 | 0.772 | 0.941 | 0.965 | 0.871 | 0.621 | 0.633 | 0.381 |
| diseased | hsv_green | True | 0.911 | 0.951 | 0.82 | 0.951 | 0.975 | 0.901 | 0.694 | 0.743 | 0.424 |
| diseased | kmeans | False | 0.922 | 0.957 | 0.843 | 0.957 | 0.978 | 0.915 | 0.702 | 0.776 | 0.273 |
| diseased | kmeans | True | 0.932 | 0.969 | 0.843 | 0.961 | 0.984 | 0.915 | 0.768 | 0.857 | 0.429 |
| diseased | otsu | False | 0.883 | 0.927 | 0.736 | 0.931 | 0.962 | 0.848 | 0.37 | 0.345 | 0.059 |
| diseased | otsu | True | 0.9 | 0.949 | 0.796 | 0.94 | 0.974 | 0.887 | 0.468 | 0.509 | 0.064 |
| diseased | watershed | False | 0.887 | 0.934 | 0.751 | 0.934 | 0.966 | 0.858 | 0.51 | 0.547 | 0.118 |
| diseased | watershed | True | 0.892 | 0.937 | 0.773 | 0.937 | 0.968 | 0.872 | 0.531 | 0.588 | 0.117 |
| healthy | grabcut | False | 0.922 | 0.939 | 0.854 | 0.959 | 0.968 | 0.921 | 0.859 | 0.885 | 0.707 |
| healthy | grabcut | True | 0.921 | 0.939 | 0.854 | 0.958 | 0.968 | 0.921 | 0.858 | 0.885 | 0.706 |
| healthy | hsv_green | False | 0.905 | 0.919 | 0.843 | 0.949 | 0.958 | 0.915 | 0.651 | 0.664 | 0.566 |
| healthy | hsv_green | True | 0.935 | 0.946 | 0.907 | 0.966 | 0.973 | 0.951 | 0.762 | 0.78 | 0.62 |
| healthy | kmeans | False | 0.929 | 0.955 | 0.831 | 0.962 | 0.977 | 0.908 | 0.947 | 0.967 | 0.855 |
| healthy | kmeans | True | 0.932 | 0.961 | 0.832 | 0.964 | 0.98 | 0.908 | 0.95 | 0.97 | 0.865 |
| healthy | otsu | False | 0.961 | 0.966 | 0.941 | 0.98 | 0.983 | 0.97 | 0.87 | 0.908 | 0.733 |
| healthy | otsu | True | 0.963 | 0.968 | 0.941 | 0.981 | 0.984 | 0.969 | 0.877 | 0.915 | 0.758 |
| healthy | watershed | False | 0.936 | 0.941 | 0.908 | 0.967 | 0.97 | 0.952 | 0.848 | 0.875 | 0.689 |
| healthy | watershed | True | 0.936 | 0.942 | 0.909 | 0.967 | 0.97 | 0.953 | 0.846 | 0.873 | 0.685 |

## Por enfermedad — IoU media

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.949 | 0.95 | 0.94 | 0.947 | 0.968 | 0.971 | 0.939 | 0.95 | 0.945 | 0.945 |
| early_blight | 0.861 | 0.861 | 0.878 | 0.899 | 0.895 | 0.912 | 0.892 | 0.91 | 0.901 | 0.901 |
| healthy | 0.922 | 0.921 | 0.905 | 0.935 | 0.929 | 0.932 | 0.961 | 0.963 | 0.936 | 0.936 |
| late_blight | 0.713 | 0.725 | 0.742 | 0.788 | 0.826 | 0.834 | 0.667 | 0.697 | 0.685 | 0.704 |
| leaf_mold | 0.817 | 0.818 | 0.857 | 0.862 | 0.859 | 0.861 | 0.849 | 0.851 | 0.854 | 0.854 |
| septoria_leaf_spot | 0.912 | 0.913 | 0.918 | 0.928 | 0.958 | 0.967 | 0.897 | 0.909 | 0.913 | 0.912 |
| spider_mites | 0.944 | 0.944 | 0.935 | 0.948 | 0.958 | 0.961 | 0.961 | 0.965 | 0.938 | 0.938 |
| target_spot | 0.937 | 0.939 | 0.95 | 0.962 | 0.966 | 0.974 | 0.96 | 0.968 | 0.946 | 0.951 |
| yellow_leaf_curl_virus | 0.941 | 0.941 | 0.947 | 0.964 | 0.954 | 0.979 | 0.919 | 0.962 | 0.93 | 0.945 |

## Por enfermedad — IoU percentil 10

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.926 | 0.927 | 0.92 | 0.926 | 0.957 | 0.96 | 0.915 | 0.934 | 0.924 | 0.924 |
| early_blight | 0.756 | 0.753 | 0.806 | 0.823 | 0.831 | 0.829 | 0.839 | 0.829 | 0.845 | 0.823 |
| healthy | 0.854 | 0.854 | 0.843 | 0.907 | 0.831 | 0.832 | 0.941 | 0.941 | 0.908 | 0.909 |
| late_blight | 0.359 | 0.359 | 0.551 | 0.629 | 0.586 | 0.537 | 0.298 | 0.292 | 0.354 | 0.353 |
| leaf_mold | 0.615 | 0.615 | 0.605 | 0.605 | 0.665 | 0.665 | 0.609 | 0.609 | 0.614 | 0.613 |
| septoria_leaf_spot | 0.872 | 0.873 | 0.854 | 0.869 | 0.912 | 0.932 | 0.829 | 0.841 | 0.87 | 0.869 |
| spider_mites | 0.916 | 0.916 | 0.913 | 0.916 | 0.923 | 0.924 | 0.957 | 0.952 | 0.928 | 0.928 |
| target_spot | 0.902 | 0.902 | 0.937 | 0.947 | 0.952 | 0.959 | 0.944 | 0.953 | 0.929 | 0.937 |
| yellow_leaf_curl_virus | 0.899 | 0.899 | 0.916 | 0.945 | 0.926 | 0.969 | 0.858 | 0.939 | 0.897 | 0.921 |

## Por enfermedad — cobertura no verde media

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.746 | 0.746 | 0.696 | 0.747 | 0.762 | 0.807 | 0.424 | 0.535 | 0.672 | 0.668 |
| early_blight | 0.739 | 0.744 | 0.606 | 0.751 | 0.676 | 0.801 | 0.456 | 0.629 | 0.679 | 0.7 |
| healthy | 0.859 | 0.858 | 0.651 | 0.762 | 0.947 | 0.95 | 0.87 | 0.877 | 0.848 | 0.846 |
| late_blight | 0.387 | 0.459 | 0.51 | 0.577 | 0.806 | 0.892 | 0.125 | 0.288 | 0.225 | 0.337 |
| leaf_mold | 0.587 | 0.585 | 0.551 | 0.553 | 0.722 | 0.724 | 0.356 | 0.365 | 0.485 | 0.481 |
| septoria_leaf_spot | 0.655 | 0.663 | 0.594 | 0.653 | 0.806 | 0.863 | 0.36 | 0.453 | 0.583 | 0.581 |
| spider_mites | 0.566 | 0.562 | 0.701 | 0.751 | 0.773 | 0.782 | 0.586 | 0.593 | 0.518 | 0.513 |
| target_spot | 0.649 | 0.662 | 0.588 | 0.684 | 0.681 | 0.755 | 0.511 | 0.6 | 0.584 | 0.617 |
| yellow_leaf_curl_virus | 0.496 | 0.495 | 0.72 | 0.819 | 0.401 | 0.514 | 0.178 | 0.299 | 0.38 | 0.389 |

## Por enfermedad — cobertura no verde percentil 10

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.389 | 0.389 | 0.349 | 0.417 | 0.492 | 0.609 | 0.139 | 0.291 | 0.386 | 0.382 |
| early_blight | 0.537 | 0.553 | 0.46 | 0.635 | 0.45 | 0.562 | 0.252 | 0.433 | 0.502 | 0.496 |
| healthy | 0.707 | 0.706 | 0.566 | 0.62 | 0.855 | 0.865 | 0.733 | 0.758 | 0.689 | 0.685 |
| late_blight | 0.09 | 0.09 | 0.332 | 0.323 | 0.33 | 0.806 | 0.015 | 0.016 | 0.057 | 0.056 |
| leaf_mold | 0.083 | 0.082 | 0.052 | 0.051 | 0.223 | 0.223 | 0.061 | 0.061 | 0.08 | 0.079 |
| septoria_leaf_spot | 0.363 | 0.361 | 0.455 | 0.498 | 0.681 | 0.759 | 0.163 | 0.202 | 0.314 | 0.309 |
| spider_mites | 0.178 | 0.177 | 0.444 | 0.434 | 0.511 | 0.527 | 0.141 | 0.143 | 0.097 | 0.093 |
| target_spot | 0.384 | 0.381 | 0.475 | 0.56 | 0.47 | 0.528 | 0.147 | 0.249 | 0.312 | 0.307 |
| yellow_leaf_curl_virus | 0.075 | 0.073 | 0.604 | 0.642 | 0.034 | 0.075 | 0.0 | 0.035 | 0.042 | 0.042 |

## Diagnóstico: cobertura no verde baja

Imágenes con cobertura no verde < 0,5 y mediana de `non_green_rim_frac` (fracción de sus píxeles no verdes a ≤ 3 px del borde de la referencia). Si es alta, los píxeles no cubiertos son sobre todo el halo de fondo/sombra de la referencia, no lesiones interiores.

| method | postprocess | images | median_rim_frac |
|---|---|---|---|
| grabcut | False | 53 | 0.45 |
| grabcut | True | 51 | 0.52 |
| hsv_green | False | 46 | 0.29 |
| hsv_green | True | 29 | 0.37 |
| kmeans | False | 32 | 0.59 |
| kmeans | True | 19 | 0.87 |
| otsu | False | 98 | 0.5 |
| otsu | True | 73 | 0.63 |
| watershed | False | 66 | 0.55 |
| watershed | True | 61 | 0.69 |

## Tiempo medio por imagen (ms) — 13th Gen Intel(R) Core(TM) i7-1355U, Linux, un hilo de Python

| method | postprocess | ms |
|---|---|---|
| grabcut | False | 78.5 |
| grabcut | True | 79.0 |
| hsv_green | False | 0.1 |
| hsv_green | True | 0.5 |
| kmeans | False | 5.9 |
| kmeans | True | 6.3 |
| otsu | False | 0.2 |
| otsu | True | 0.6 |
| watershed | False | 0.9 |
| watershed | True | 1.3 |
