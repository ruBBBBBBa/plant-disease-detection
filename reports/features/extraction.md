# Extracción de características — registro

## subset

`python scripts/extract_features.py --subset-only` → `data/processed/features_v1_subset.csv` (no versionado), SHA-1 `04020f706325ec6bcfd1319dd4de754306be947f`.

- Imágenes por split: {'test': 296, 'train': 1406, 'val': 296}; características por imagen: 75.
- Tiempo medio por imagen (segmentación + características): 41.6 ms — 13th Gen Intel(R) Core(TM) i7-1355U, Linux, un hilo.
- Las características de `val` y `test` se calculan pero no se analizan (protocolo T04).

## test_full

`python scripts/extract_features.py --full-test` → `data/processed/features_v1_test_full.csv` (no versionado), SHA-1 `560c1f1e82de33a706eaa40f0307579e5054d453`.

- Imágenes por split: {'test': 1900}; características por imagen: 75.
- Tiempo medio por imagen (segmentación + características): 43.7 ms — 13th Gen Intel(R) Core(TM) i7-1355U, Linux, un hilo.
- Las características de `val` y `test` se calculan pero no se analizan (protocolo T04).
