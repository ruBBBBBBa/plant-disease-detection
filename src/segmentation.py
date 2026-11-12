"""Segmentación clásica de la hoja frente al fondo (T03).

Todos los métodos reciben una imagen BGR ``uint8`` y devuelven una máscara ``uint8`` 0/255 del
mismo alto y ancho (255 = hoja). Ninguno usa aprendizaje: son técnicas del Tema 5 (umbralización,
Otsu, K-means, watershed) y de la práctica de segmentación (GrabCut, watershed con marcadores).

Convenciones de OpenCV que hay que recordar: las imágenes son BGR; en HSV de 8 bits H va de
0 a 179 (grados / 2) y S, V de 0 a 255; en Lab de 8 bits a* y b* están desplazados +128.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- utilidades


def read_image(path: Path) -> np.ndarray:
    """Lee una imagen BGR y detiene el CLI con ayuda si falta o no se puede decodificar."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(
            f"Error: no se puede leer la imagen {path}. "
            "Comprueba que existe y no está corrupta. Descarga los datos color y segmented "
            "siguiendo data/README.md."
        )
    return image


def to_mask(binary: np.ndarray) -> np.ndarray:
    """Convierte cualquier matriz booleana o 0/1 en máscara ``uint8`` 0/255."""
    return np.where(binary, 255, 0).astype(np.uint8)


def ellipse_kernel(size: int) -> np.ndarray:
    """Elemento estructurante elíptico ``size × size`` (bordes de hoja redondeados)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def channel(image_bgr: np.ndarray, name: str) -> np.ndarray:
    """Devuelve un canal ``uint8`` candidato para umbralizar la hoja.

    - ``S``: saturación de HSV. El fondo de PlantVillage es gris (poco saturado); la hoja,
      verde, amarilla o marrón, es saturada.
    - ``a``: a* de Lab invertido (255 − a*), alto en verde y bajo en rojo/marrón.
    - ``L``: luminosidad L* de Lab.
    - ``ExG``: exceso de verde 2G − R − B, reescalado a 0–255.
    """
    if name == "S":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    if name == "a":
        return 255 - cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    if name == "L":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    if name == "ExG":
        b, g, r = cv2.split(image_bgr.astype(np.int16))  # int16: 2G − R − B no cabe en uint8
        exg = 2 * g - r - b
        return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    raise ValueError(f"Canal desconocido: {name!r}")


# --------------------------------------------------------------------------- métodos


def segment_hsv_green(
    image_bgr: np.ndarray,
    h_min: int = 25,
    h_max: int = 95,
    s_min: int = 40,
    v_min: int = 30,
) -> np.ndarray:
    """M1 — Umbral de color en HSV con ``cv2.inRange`` (baseline).

    Técnica: umbralización por rangos en un espacio de color (Tema 5 y práctica de color).
    Se queda con los píxeles de tono verde (H en [h_min, h_max], escala 0–179) suficientemente
    saturados y claros. Es el método «obvio» y sirve para medir lo que se pierde al suponer que
    la hoja es verde: las lesiones marrones y amarillas caen fuera del rango.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))


def segment_otsu(image_bgr: np.ndarray, channel_name: str = "S", blur: int = 5) -> np.ndarray:
    """M2 — Umbral global de Otsu sobre un canal (``cv2.threshold`` con ``THRESH_OTSU``).

    Técnica: umbralización global automática (Tema 5). Otsu elige el umbral que maximiza la
    varianza entre las dos clases del histograma, así que funciona si el canal es bimodal
    (hoja / fondo). El canal se elige con ``train`` (ver ``scripts/tune_segmentation.py``).
    Antes se suaviza con un filtro gaussiano para que el histograma sea menos ruidoso.
    En todos los canales de ``channel`` la hoja queda en la parte alta.
    """
    values = channel(image_bgr, channel_name)
    if blur > 1:
        values = cv2.GaussianBlur(values, (blur, blur), 0)
    _, mask = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_kmeans(
    image_bgr: np.ndarray,
    k: int = 3,
    border_share: float = 0.2,
    features: str = "ab",
    sample: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    """M3 — Clustering K-means de los píxeles en Lab (``cv2.kmeans``).

    Técnica: segmentación por clustering (Tema 5). Se agrupan los colores en ``k`` clústeres
    usando solo la cromaticidad (a*, b*) (``features="ab"``) o también la luminosidad
    (``features="Lab"``). Sin L*, la distancia no la domina el brillo: una hoja oscura no cae en el
    mismo clúster que la sombra y el fondo iluminado y el sombreado se parecen más (con ``train``:
    IoU 0,93 con ab frente a 0,52 con Lab; ver ``experiments/segmentation/tuning_train/``).
    Aun así, una sombra marcada con tinte propio puede formar su propio clúster y entrar como hoja.

    **Regla para decidir qué es hoja:** la hoja está centrada y el fondo rodea la imagen, así que
    es **fondo todo clúster que ocupe al menos ``border_share`` de los píxeles del borde** (y
    siempre el que más ocupe); el resto es hoja. Un color de lesión (marrón, amarillo) no se
    descarta por no ser verde. (Se probó antes «fondo = clúster de menor croma» y fallaba cuando
    el fondo se partía en dos clústeres.)
    Para ir rápido se ajusta con ``sample`` píxeles al azar (semilla fija) y luego se asigna
    cada píxel al centro más cercano.
    """
    height, width = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if features == "ab":
        lab = np.ascontiguousarray(lab[:, 1:])
    elif features != "Lab":
        raise ValueError(f"features debe ser 'ab' o 'Lab', no {features!r}")
    rng = np.random.default_rng(seed)
    fit = lab[rng.choice(len(lab), size=min(sample, len(lab)), replace=False)]
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(fit, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = np.argmin(((lab[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    labels = labels.reshape(height, width)
    border = np.concatenate([labels[0], labels[-1], labels[1:-1, 0], labels[1:-1, -1]])
    share = np.bincount(border, minlength=k) / len(border)
    background = (share >= border_share) | (share == share.max())
    return to_mask(~background[labels])


def segment_grabcut(
    image_bgr: np.ndarray,
    channel_name: str = "S",
    iterations: int = 3,
    border: int = 0,
) -> np.ndarray:
    """M4 — GrabCut inicializado con la máscara de Otsu (M2) (``cv2.grabCut``).

    Técnica: GrabCut de la práctica de segmentación (modelos de color por mezcla de gaussianas
    + corte de grafo). En vez de un rectángulo se inicializa con M2: sus píxeles de hoja son
    «probable hoja», el resto «probable fondo» y una franja de ``border`` píxeles en el borde de
    la imagen puede marcarse como «fondo seguro» (0 = no, porque hay hojas que tocan el borde).
    GrabCut refina el borde usando el color de toda la región.
    """
    init = segment_otsu(image_bgr, channel_name)
    mask = np.where(init > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    if border > 0:
        mask[:border, :] = mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = mask[:, -border:] = cv2.GC_BGD
    if not (mask == cv2.GC_PR_FGD).any():
        return np.zeros(image_bgr.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(image_bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    return to_mask((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))


def segment_watershed(
    image_bgr: np.ndarray, channel_name: str = "S", erode: int = 9, dilate: int = 9
) -> np.ndarray:
    """M5 — Watershed con marcadores (``cv2.watershed``), como en la práctica.

    Técnica: watershed (Tema 5) con marcadores obtenidos de Otsu (M2): «hoja segura» = máscara
    erosionada, «fondo seguro» = complemento de la máscara dilatada, y la franja intermedia queda
    sin etiquetar. Watershed inunda desde los marcadores y coloca la frontera en los bordes de
    la imagen, que es donde de verdad termina la hoja.
    """
    init = segment_otsu(image_bgr, channel_name)
    sure_fg = cv2.erode(init, ellipse_kernel(erode))
    sure_bg = cv2.dilate(init, ellipse_kernel(dilate)) == 0
    markers = np.zeros(init.shape, np.int32)
    markers[sure_bg] = 1
    markers[sure_fg > 0] = 2
    cv2.watershed(image_bgr, markers)  # las fronteras quedan con -1
    return to_mask(markers == 2)


METHODS = {
    "hsv_green": segment_hsv_green,
    "otsu": segment_otsu,
    "kmeans": segment_kmeans,
    "grabcut": segment_grabcut,
    "watershed": segment_watershed,
}


# --------------------------------------------------------------------------- postprocesado


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Rellena los agujeros interiores de una máscara 0/255.

    Se inunda el fondo desde una esquina con ``cv2.floodFill`` sobre una copia con un marco de
    1 píxel: lo que no se alcanza desde fuera y no es hoja es un agujero (p. ej. una lesión
    oscura que el umbral dejó fuera) y se añade a la máscara.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Se queda con la mayor componente conexa (8-vecindad) de la máscara."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return to_mask(labels == biggest)


def postprocess(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Postprocesado común: apertura, cierre, mayor componente conexa y relleno de huecos.

    Técnica: morfología matemática y análisis de componentes conexas (Tema 5). La apertura quita
    puntos sueltos de fondo; el cierre une pequeños cortes del borde; la mayor componente descarta
    trozos de sombra o fondo separados; el relleno recupera lesiones interiores.
    """
    if kernel > 1:
        k = ellipse_kernel(kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return fill_holes(largest_component(mask))


# --------------------------------------------------------------------------- referencia y métricas


def reference_mask(segmented_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Máscara de referencia a partir de la imagen ``segmented`` de PlantVillage.

    En ``segmented`` la hoja está sobre fondo negro. Por la compresión JPEG el fondo no es
    exactamente 0, así que es hoja todo píxel con ``max(B, G, R) > threshold`` (umbral elegido
    con el histograma de ``train``). Después se rellenan los huecos interiores: un píxel negro
    rodeado de hoja es una lesión muy oscura, no fondo. Las lesiones del **borde** que el
    procesado original recortó no se pueden recuperar (defecto documentado de la referencia).
    """
    return fill_holes(to_mask(segmented_bgr.max(axis=2) > threshold))


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """Intersección sobre unión |A∩B| / |A∪B| (1 si ambas máscaras están vacías)."""
    a, b = pred > 0, ref > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Coeficiente de Dice 2|A∩B| / (|A| + |B|) (1 si ambas están vacías)."""
    a, b = pred > 0, ref > 0
    total = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / total) if total else 1.0


def non_green_mask(
    image_bgr: np.ndarray, ref: np.ndarray, h_min: int = 30, h_max: int = 90, s_min: int = 40
) -> np.ndarray:
    """Píxeles de la hoja de referencia que **no** son verdes.

    «Verde» = H en [h_min, h_max] (escala OpenCV 0–179, ≈ 60°–180°) **y** S ≥ ``s_min``. Todo lo
    demás dentro de la referencia (amarillo, marrón, necrosis grisácea u oscura) cuenta como no
    verde: es donde suelen estar los síntomas que el clasificador necesitará.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    green = (h >= h_min) & (h <= h_max) & (s >= s_min)
    return to_mask((ref > 0) & ~green)


def non_green_coverage(pred: np.ndarray, non_green: np.ndarray) -> float:
    """Fracción de los píxeles no verdes de la referencia que incluye la máscara (NaN si no hay)."""
    total = (non_green > 0).sum()
    return float(((pred > 0) & (non_green > 0)).sum() / total) if total else float("nan")


def rim_fraction(non_green: np.ndarray, ref: np.ndarray, width: int = 3) -> float:
    """Fracción de los píxeles no verdes que están a ``width`` píxeles o menos del borde de la ref.

    Diagnóstico de la métrica: si casi todos los píxeles no verdes están en el borde, suelen ser
    el halo de fondo o sombra que la referencia incluye, no lesiones (NaN si no hay no verdes).
    """
    total = (non_green > 0).sum()
    if not total:
        return float("nan")
    inner = cv2.erode(ref, ellipse_kernel(2 * width + 1))
    return float(((non_green > 0) & (inner == 0)).sum() / total)


# --------------------------------------------------------------------------- datos


def segmented_path(color_path: Path) -> Path:
    """Ruta del fichero ``segmented`` que corresponde a una imagen ``color`` de PlantVillage.

    ``raw/color/<clase>/<nombre>.JPG`` → ``raw/segmented/<clase>/<nombre>_final_masked.jpg``.
    Caso especial visto en T01 (*Spider_mites*): el segmentado no lleva el prefijo UUID
    (``Com.G_SpM_FL 1378_final_masked.jpg``), así que si no existe se prueba sin el prefijo.
    """
    seg_dir = Path(str(color_path.parent).replace("/raw/color/", "/raw/segmented/"))
    candidate = seg_dir / f"{color_path.stem}_final_masked.jpg"
    if candidate.is_file() or "___" not in color_path.stem:
        return candidate
    return seg_dir / f"{color_path.stem.split('___')[-1]}_final_masked.jpg"


def select_images(
    metadata: pd.DataFrame, split: str, sample_per_class: int | None, seed: int
) -> pd.DataFrame:
    """Imágenes de PlantVillage del subconjunto (``in_subset``) en un split.

    Con ``sample_per_class`` se toma una muestra fija de ese tamaño por ``disease_name``
    (semilla declarada), útil para ajustar con ``train`` sin recorrerlo entero.
    """
    in_subset = metadata["in_subset"].astype(str) == "True"
    rows = metadata[
        (metadata["source"] == "plantvillage") & in_subset & (metadata["split"] == split)
    ]
    rows = rows.sort_values("image_path")
    if sample_per_class:
        picked = [
            group.sample(min(sample_per_class, len(group)), random_state=seed)
            for _, group in rows.groupby("disease_name")
        ]
        rows = pd.concat(picked)
    return rows.sort_values("image_path").reset_index(drop=True)
