# Segmentación — evaluación en `val`

`python scripts/evaluate_segmentation.py --split val --seed 42 --postprocess both` — 296 imágenes. Parámetros: `configs/segmentation.json` (ajustados con train). Referencia: `segmented` con umbral 22 y relleno de huecos. Cobertura no verde: H fuera de [30, 90] o S < 40 dentro de la referencia (NaN si no hay píxeles no verdes).

`postprocess=True`: apertura + cierre (núcleo 3), mayor componente conexa y relleno de huecos. Rejillas `overlay_<método>.jpg`: variante con postprocesado; fila 1, 8 casos cerca de la mediana de IoU; fila 2, los 8 peores. Amarillo = referencia, rojo = método.

## Global

| method | postprocess | iou_mean | iou_median | iou_p10 | dice_mean | dice_median | dice_p10 | non_green_cov_mean | non_green_cov_median | non_green_cov_p10 |
|---|---|---|---|---|---|---|---|---|---|---|
| grabcut | False | 0.911 | 0.935 | 0.837 | 0.952 | 0.966 | 0.911 | 0.741 | 0.824 | 0.324 |
| grabcut | True | 0.91 | 0.935 | 0.837 | 0.951 | 0.967 | 0.911 | 0.741 | 0.825 | 0.337 |
| hsv_green | False | 0.904 | 0.921 | 0.81 | 0.948 | 0.959 | 0.895 | 0.645 | 0.643 | 0.481 |
| hsv_green | True | 0.927 | 0.952 | 0.848 | 0.961 | 0.975 | 0.918 | 0.743 | 0.769 | 0.556 |
| kmeans | False | 0.938 | 0.959 | 0.858 | 0.967 | 0.979 | 0.924 | 0.824 | 0.915 | 0.497 |
| kmeans | True | 0.943 | 0.966 | 0.872 | 0.97 | 0.983 | 0.932 | 0.854 | 0.932 | 0.579 |
| otsu | False | 0.935 | 0.959 | 0.844 | 0.965 | 0.979 | 0.916 | 0.642 | 0.723 | 0.132 |
| otsu | True | 0.94 | 0.965 | 0.871 | 0.968 | 0.982 | 0.931 | 0.687 | 0.776 | 0.273 |
| watershed | False | 0.922 | 0.94 | 0.864 | 0.959 | 0.969 | 0.927 | 0.692 | 0.761 | 0.264 |
| watershed | True | 0.924 | 0.942 | 0.875 | 0.959 | 0.97 | 0.934 | 0.703 | 0.761 | 0.287 |

## Por `binary_label`

| binary_label | method | postprocess | iou_mean | iou_median | iou_p10 | dice_mean | dice_median | dice_p10 | non_green_cov_mean | non_green_cov_median | non_green_cov_p10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diseased | grabcut | False | 0.9 | 0.935 | 0.79 | 0.945 | 0.966 | 0.883 | 0.621 | 0.688 | 0.214 |
| diseased | grabcut | True | 0.898 | 0.935 | 0.788 | 0.943 | 0.967 | 0.881 | 0.622 | 0.692 | 0.208 |
| diseased | hsv_green | False | 0.917 | 0.935 | 0.835 | 0.956 | 0.966 | 0.91 | 0.67 | 0.705 | 0.458 |
| diseased | hsv_green | True | 0.933 | 0.958 | 0.85 | 0.964 | 0.979 | 0.919 | 0.754 | 0.805 | 0.47 |
| diseased | kmeans | False | 0.945 | 0.96 | 0.893 | 0.971 | 0.98 | 0.944 | 0.704 | 0.775 | 0.349 |
| diseased | kmeans | True | 0.954 | 0.971 | 0.914 | 0.975 | 0.985 | 0.955 | 0.762 | 0.837 | 0.44 |
| diseased | otsu | False | 0.909 | 0.941 | 0.805 | 0.95 | 0.969 | 0.892 | 0.426 | 0.403 | 0.085 |
| diseased | otsu | True | 0.919 | 0.957 | 0.813 | 0.955 | 0.978 | 0.897 | 0.511 | 0.529 | 0.131 |
| diseased | watershed | False | 0.91 | 0.933 | 0.816 | 0.951 | 0.965 | 0.898 | 0.546 | 0.583 | 0.205 |
| diseased | watershed | True | 0.912 | 0.94 | 0.835 | 0.952 | 0.969 | 0.91 | 0.57 | 0.611 | 0.214 |
| healthy | grabcut | False | 0.922 | 0.935 | 0.864 | 0.958 | 0.966 | 0.927 | 0.861 | 0.907 | 0.683 |
| healthy | grabcut | True | 0.922 | 0.935 | 0.864 | 0.958 | 0.967 | 0.927 | 0.86 | 0.907 | 0.68 |
| healthy | hsv_green | False | 0.89 | 0.905 | 0.8 | 0.941 | 0.95 | 0.889 | 0.621 | 0.624 | 0.501 |
| healthy | hsv_green | True | 0.922 | 0.941 | 0.847 | 0.958 | 0.97 | 0.917 | 0.733 | 0.739 | 0.584 |
| healthy | kmeans | False | 0.931 | 0.957 | 0.843 | 0.963 | 0.978 | 0.915 | 0.944 | 0.975 | 0.864 |
| healthy | kmeans | True | 0.933 | 0.959 | 0.844 | 0.964 | 0.979 | 0.915 | 0.946 | 0.979 | 0.868 |
| healthy | otsu | False | 0.961 | 0.968 | 0.937 | 0.98 | 0.984 | 0.967 | 0.858 | 0.922 | 0.632 |
| healthy | otsu | True | 0.962 | 0.97 | 0.934 | 0.98 | 0.985 | 0.966 | 0.863 | 0.921 | 0.645 |
| healthy | watershed | False | 0.935 | 0.942 | 0.903 | 0.966 | 0.97 | 0.949 | 0.838 | 0.89 | 0.611 |
| healthy | watershed | True | 0.935 | 0.942 | 0.903 | 0.966 | 0.97 | 0.949 | 0.836 | 0.889 | 0.604 |

## Por enfermedad — IoU media

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.954 | 0.955 | 0.948 | 0.953 | 0.975 | 0.976 | 0.95 | 0.956 | 0.948 | 0.948 |
| early_blight | 0.873 | 0.872 | 0.87 | 0.893 | 0.905 | 0.936 | 0.869 | 0.898 | 0.886 | 0.891 |
| healthy | 0.922 | 0.922 | 0.89 | 0.922 | 0.931 | 0.933 | 0.961 | 0.962 | 0.935 | 0.935 |
| late_blight | 0.742 | 0.731 | 0.827 | 0.844 | 0.916 | 0.912 | 0.777 | 0.776 | 0.793 | 0.79 |
| leaf_mold | 0.911 | 0.911 | 0.95 | 0.955 | 0.926 | 0.927 | 0.937 | 0.938 | 0.934 | 0.933 |
| septoria_leaf_spot | 0.939 | 0.939 | 0.961 | 0.963 | 0.969 | 0.973 | 0.931 | 0.939 | 0.93 | 0.93 |
| spider_mites | 0.943 | 0.943 | 0.92 | 0.949 | 0.958 | 0.962 | 0.955 | 0.961 | 0.938 | 0.939 |
| target_spot | 0.916 | 0.916 | 0.938 | 0.957 | 0.961 | 0.966 | 0.961 | 0.965 | 0.945 | 0.947 |
| yellow_leaf_curl_virus | 0.936 | 0.936 | 0.94 | 0.965 | 0.956 | 0.977 | 0.916 | 0.941 | 0.922 | 0.938 |

## Por enfermedad — IoU percentil 10

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.932 | 0.932 | 0.914 | 0.921 | 0.966 | 0.97 | 0.901 | 0.915 | 0.918 | 0.918 |
| early_blight | 0.796 | 0.793 | 0.817 | 0.827 | 0.84 | 0.889 | 0.783 | 0.808 | 0.805 | 0.808 |
| healthy | 0.864 | 0.864 | 0.8 | 0.847 | 0.843 | 0.844 | 0.937 | 0.934 | 0.903 | 0.903 |
| late_blight | 0.573 | 0.542 | 0.755 | 0.767 | 0.844 | 0.857 | 0.684 | 0.613 | 0.714 | 0.614 |
| leaf_mold | 0.861 | 0.86 | 0.923 | 0.929 | 0.821 | 0.822 | 0.91 | 0.918 | 0.911 | 0.91 |
| septoria_leaf_spot | 0.899 | 0.9 | 0.924 | 0.923 | 0.919 | 0.933 | 0.852 | 0.86 | 0.873 | 0.872 |
| spider_mites | 0.91 | 0.91 | 0.871 | 0.907 | 0.934 | 0.941 | 0.927 | 0.922 | 0.91 | 0.908 |
| target_spot | 0.863 | 0.864 | 0.907 | 0.927 | 0.948 | 0.951 | 0.943 | 0.948 | 0.925 | 0.926 |
| yellow_leaf_curl_virus | 0.9 | 0.9 | 0.885 | 0.955 | 0.916 | 0.965 | 0.82 | 0.86 | 0.86 | 0.909 |

## Por enfermedad — cobertura no verde media

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.65 | 0.65 | 0.764 | 0.775 | 0.675 | 0.693 | 0.418 | 0.47 | 0.571 | 0.568 |
| early_blight | 0.723 | 0.729 | 0.568 | 0.767 | 0.477 | 0.729 | 0.248 | 0.49 | 0.549 | 0.594 |
| healthy | 0.861 | 0.86 | 0.621 | 0.733 | 0.944 | 0.946 | 0.858 | 0.863 | 0.838 | 0.836 |
| late_blight | 0.428 | 0.438 | 0.537 | 0.635 | 0.847 | 0.892 | 0.204 | 0.323 | 0.309 | 0.446 |
| leaf_mold | 0.629 | 0.628 | 0.707 | 0.714 | 0.664 | 0.663 | 0.444 | 0.458 | 0.589 | 0.579 |
| septoria_leaf_spot | 0.563 | 0.562 | 0.742 | 0.771 | 0.678 | 0.717 | 0.288 | 0.414 | 0.515 | 0.51 |
| spider_mites | 0.704 | 0.702 | 0.629 | 0.76 | 0.813 | 0.827 | 0.732 | 0.741 | 0.68 | 0.677 |
| target_spot | 0.826 | 0.825 | 0.737 | 0.806 | 0.886 | 0.898 | 0.771 | 0.787 | 0.76 | 0.759 |
| yellow_leaf_curl_virus | 0.491 | 0.488 | 0.716 | 0.809 | 0.615 | 0.671 | 0.378 | 0.443 | 0.449 | 0.466 |

## Por enfermedad — cobertura no verde percentil 10

| disease_name | grabcut | grabcut+post | hsv_green | hsv_green+post | kmeans | kmeans+post | otsu | otsu+post | watershed | watershed+post |
|---|---|---|---|---|---|---|---|---|---|---|
| bacterial_spot | 0.406 | 0.405 | 0.578 | 0.583 | 0.46 | 0.464 | 0.302 | 0.314 | 0.326 | 0.322 |
| early_blight | 0.432 | 0.444 | 0.476 | 0.641 | 0.198 | 0.517 | 0.032 | 0.369 | 0.331 | 0.426 |
| healthy | 0.683 | 0.68 | 0.501 | 0.584 | 0.864 | 0.868 | 0.632 | 0.645 | 0.611 | 0.604 |
| late_blight | 0.1 | 0.098 | 0.237 | 0.231 | 0.672 | 0.65 | 0.024 | 0.04 | 0.135 | 0.158 |
| leaf_mold | 0.117 | 0.113 | 0.465 | 0.469 | 0.063 | 0.06 | 0.061 | 0.061 | 0.196 | 0.191 |
| septoria_leaf_spot | 0.24 | 0.242 | 0.646 | 0.683 | 0.435 | 0.445 | 0.123 | 0.252 | 0.242 | 0.236 |
| spider_mites | 0.351 | 0.351 | 0.286 | 0.28 | 0.522 | 0.522 | 0.314 | 0.315 | 0.226 | 0.22 |
| target_spot | 0.651 | 0.649 | 0.625 | 0.718 | 0.745 | 0.755 | 0.524 | 0.535 | 0.497 | 0.491 |
| yellow_leaf_curl_virus | 0.173 | 0.161 | 0.606 | 0.636 | 0.302 | 0.307 | 0.117 | 0.124 | 0.117 | 0.105 |

## Diagnóstico: cobertura no verde baja

Imágenes con cobertura no verde < 0,5 y mediana de `non_green_rim_frac` (fracción de sus píxeles no verdes a ≤ 3 px del borde de la referencia). Si es alta, los píxeles no cubiertos son sobre todo el halo de fondo/sombra de la referencia, no lesiones interiores.

| method | postprocess | images | median_rim_frac |
|---|---|---|---|
| grabcut | False | 47 | 0.78 |
| grabcut | True | 47 | 0.78 |
| hsv_green | False | 35 | 0.25 |
| hsv_green | True | 20 | 0.81 |
| kmeans | False | 30 | 0.83 |
| kmeans | True | 19 | 0.95 |
| otsu | False | 97 | 0.54 |
| otsu | True | 73 | 0.69 |
| watershed | False | 64 | 0.74 |
| watershed | True | 59 | 0.78 |

## Tiempo medio por imagen (ms) — 13th Gen Intel(R) Core(TM) i7-1355U, Linux, un hilo de Python

| method | postprocess | ms |
|---|---|---|
| grabcut | False | 83.8 |
| grabcut | True | 84.3 |
| hsv_green | False | 0.1 |
| hsv_green | True | 0.5 |
| kmeans | False | 6.1 |
| kmeans | True | 6.5 |
| otsu | False | 0.2 |
| otsu | True | 0.6 |
| watershed | False | 1.0 |
| watershed | True | 1.4 |
