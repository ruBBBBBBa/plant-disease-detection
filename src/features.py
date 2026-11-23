"""Características explícitas de color, textura y manchas de una hoja (T04).

Todas se calculan **solo dentro de la máscara de la hoja** (D017), que sale de la segmentación
congelada de T03 (K-means + postprocesado) y de la variante de sombra elegida con ``train``
(D019). Cada familia es una función que devuelve un diccionario ``{nombre: valor}``; el orden de
inserción es el orden de las columnas, así que los nombres y su orden son estables para una
configuración dada.

Rangos de OpenCV a recordar: H de 0 a 179 (grados / 2); S, V, L de 0 a 255; a* y b* desplazados
+128. Aquí se convierten a unidades interpretables (grados, 0–1, L* 0–100, a*/b* con signo).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

from src import segmentation as seg

DARK_V = 60  # «oscuro» para el criterio de sombras (fijado antes de ver resultados, ver T04)


# --------------------------------------------------------------------------- máscara


def leaf_mask(image_bgr: np.ndarray, seg_config: dict) -> np.ndarray:
    """Máscara de la hoja con el método congelado de T03 (``configs/segmentation.json``)."""
    method = seg_config["default_method"]
    mask = seg.METHODS[method](image_bgr, **seg_config["methods"][method])
    if seg_config["default_postprocess"]:
        mask = seg.postprocess(mask, seg_config["postprocess_kernel"])
    return mask


def erode_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Variante (b): erosiona la máscara ``n`` píxeles (núcleo elíptico de lado 2n + 1).

    Quita una franja del contorno, donde suelen quedar la sombra pegada a la hoja y el halo de
    fondo; a cambio también quita el borde real de la hoja (y sus lesiones de borde).
    """
    return cv2.erode(mask, seg.ellipse_kernel(2 * n + 1)) if n > 0 else mask.copy()


def remove_shadow(image_bgr: np.ndarray, mask: np.ndarray, v_max: int, s_max: int) -> np.ndarray:
    """Variante (c): quita de la máscara los píxeles de sombra conectados a su borde.

    Sombra candidata = píxel de la máscara **oscuro** (V < ``v_max``) y **poco saturado**
    (S < ``s_max``). Solo se quitan las componentes conexas de candidatos que **tocan el borde de
    la máscara**: una sombra proyectada entra desde fuera de la hoja, mientras que una lesión
    interior oscura está rodeada de hoja y se conserva. Técnica: umbralización en HSV y
    componentes conexas (Temas 2 y 5).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    candidates = (mask > 0) & (hsv[:, :, 2] < v_max) & (hsv[:, :, 1] < s_max)
    n, labels = cv2.connectedComponents(seg.to_mask(candidates), connectivity=8)
    rim = (mask > 0) & (cv2.erode(mask, seg.ellipse_kernel(3)) == 0)
    touching = np.unique(labels[rim & candidates])
    shadow = np.isin(labels, touching[touching > 0])
    return seg.to_mask((mask > 0) & ~shadow)


def prepare_mask(image_bgr: np.ndarray, mask: np.ndarray, variant: dict) -> np.ndarray:
    """Aplica la variante de máscara de ``configs/features.json`` (raw, erode o shadow)."""
    kind = variant["kind"]
    if kind == "raw":
        return mask.copy()
    if kind == "erode":
        return erode_mask(mask, variant["n"])
    if kind == "shadow":
        return remove_shadow(image_bgr, mask, variant["v_max"], variant["s_max"])
    raise ValueError(f"Variante de máscara desconocida: {kind!r}")


def dark_fraction(image_bgr: np.ndarray, mask: np.ndarray, v_dark: int = DARK_V) -> float:
    """Fracción de píxeles de la máscara con V < ``v_dark`` (medida del criterio de sombras)."""
    inside = mask > 0
    if not inside.any():
        return 0.0
    v = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float((v[inside] < v_dark).mean())


# --------------------------------------------------------------------------- color


def tone_masks(image_bgr: np.ndarray, mask: np.ndarray, tones: dict) -> dict[str, np.ndarray]:
    """Máscaras booleanas de tonos anómalos dentro de la hoja.

    Umbrales explícitos (``tones``), elegidos con las hojas **sanas** de ``train``:
    - ``dark``: V < ``v_dark``.
    - ``brown``: V ≥ ``v_hue_min``, S ≥ ``s_min`` y H < ``h_brown_yellow`` o H ≥ 160 (rojos;
      H es circular).
    - ``yellow``: V ≥ ``v_hue_min``, S ≥ ``s_min`` y ``h_brown_yellow`` ≤ H < ``h_green_min``.
    Con poco brillo o poca saturación el tono H es ruido (en ``train`` las hojas sanas muestran
    picos espurios de H en píxeles oscuros), por eso marrón y amarillo exigen V y S mínimos.
    No se supone que verde = sano: son proporciones que el clasificador ponderará.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    inside = mask > 0
    dark = inside & (v < tones["v_dark"])
    colored = inside & (v >= tones["v_hue_min"]) & (s >= tones["s_min"])
    brown = colored & ((h < tones["h_brown_yellow"]) | (h >= 160))
    yellow = colored & (h >= tones["h_brown_yellow"]) & (h < tones["h_green_min"])
    return {"yellow": yellow, "brown": brown, "dark": dark}


def color_features(image_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> dict[str, float]:
    """Color dentro de la hoja: estadísticas HSV y Lab, histogramas gruesos y tonos anómalos.

    Técnica: espacios de color e histogramas (Tema 2) y descriptores de color (Tema 4).
    - H es un ángulo: se usa la **media circular** (grados) y la **varianza circular** (0–1),
      porque una media aritmética de 179 y 1 daría 90 en vez de ~0.
    - S, V en 0–1; L* en 0–100; a* (verde − / rojo +) y b* (azul − / amarillo +) con signo.
    - Histogramas normalizados (suman 1): H en 12 intervalos de 30°; a* y b* en 8 intervalos dentro
      del rango p1–p99 de ``train`` (los valores fuera se acumulan en los extremos).
    - Fracciones de tonos amarillento, marrón y oscuro (``tone_masks``): los síntomas de clorosis,
      necrosis y tejido muerto suelen alejar el color del verde de una hoja sana.
    """
    inside = mask > 0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[inside].astype(np.float64)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[inside].astype(np.float64)
    angles = np.deg2rad(hsv[:, 0] * 2)
    mean_vector = complex(np.cos(angles).mean(), np.sin(angles).mean())
    s, v = hsv[:, 1] / 255, hsv[:, 2] / 255
    l_star, a_star, b_star = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128

    feats = {
        "h_circ_mean_deg": math.degrees(math.atan2(mean_vector.imag, mean_vector.real)) % 360,
        "h_circ_var": 1 - abs(mean_vector),
        "s_mean": s.mean(),
        "s_std": s.std(),
        "v_mean": v.mean(),
        "v_std": v.std(),
        "l_mean": l_star.mean(),
        "l_std": l_star.std(),
        "a_mean": a_star.mean(),
        "a_std": a_star.std(),
        "b_mean": b_star.mean(),
        "b_std": b_star.std(),
    }
    h_hist = np.histogram(hsv[:, 0], bins=12, range=(0, 180))[0]
    for i, value in enumerate(h_hist / h_hist.sum()):
        feats[f"h_hist_{i:02d}"] = value
    for name, values in (("a", a_star), ("b", b_star)):
        low, high = cfg["hist_ranges"][name]
        hist = np.histogram(np.clip(values, low, high), bins=8, range=(low, high))[0]
        for i, value in enumerate(hist / hist.sum()):
            feats[f"{name}_hist_{i:02d}"] = value
    area = inside.sum()
    for tone, tone_mask in tone_masks(image_bgr, mask, cfg["tones"]).items():
        feats[f"frac_{tone}"] = tone_mask.sum() / area
    return {k: float(v) for k, v in feats.items()}


# --------------------------------------------------------------------------- textura


def glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int, distance: int) -> dict:
    """GLCM solo con pares de píxeles de la hoja (Tema 4).

    **Exclusión del fondo:** el gris de la hoja se cuantiza a los niveles 1..``levels`` y el fondo
    queda en el nivel 0, reservado. Se calcula la GLCM con ``levels + 1`` niveles y se **eliminan
    la fila y la columna 0** antes de normalizar: así no cuenta ningún par hoja–fondo ni
    fondo–fondo. Propiedades (contraste, homogeneidad, energía, correlación) promediadas en 4
    ángulos (0°, 45°, 90°, 135°). Las lesiones rompen la textura regular de la hoja sana: más
    contraste y menos homogeneidad.
    """
    quantized = (1 + gray.astype(np.int32) * levels // 256).astype(np.uint8)
    quantized[mask == 0] = 0
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(quantized, [distance], angles, levels=levels + 1, symmetric=True)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)
    return {
        f"glcm_{prop}": float(graycoprops(glcm, prop).mean())
        for prop in ("contrast", "homogeneity", "energy", "correlation")
    }


def lbp_features(gray: np.ndarray, mask: np.ndarray, points: int, radius: int) -> dict:
    """Histograma LBP uniforme solo sobre la hoja (Tema 4).

    LBP compara cada píxel con ``points`` vecinos en un círculo de radio ``radius``; con
    ``method="uniform"`` los patrones uniformes (bordes, esquinas, zonas planas) tienen su propio
    código y el resto comparten uno (``points + 2`` códigos). **Exclusión del fondo:** el histograma
    solo usa píxeles de la máscara erosionada ``radius + 1`` píxeles, para que ningún vecindario
    toque el fondo (los vecinos del círculo se interpolan con los píxeles enteros que los rodean,
    que pueden estar a ``radius + 1``). Normalizado (suma 1).
    """
    codes = local_binary_pattern(gray, points, radius, method="uniform")
    inner = cv2.erode(mask, seg.ellipse_kernel(2 * (radius + 1) + 1)) > 0
    hist = np.bincount(codes[inner].astype(np.int64), minlength=points + 2)[: points + 2]
    hist = hist / hist.sum() if hist.sum() else hist.astype(np.float64)
    return {f"lbp_{i:02d}": float(value) for i, value in enumerate(hist)}


def texture_features(image_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> dict[str, float]:
    """GLCM + LBP sobre el gris de la imagen, solo con píxeles de la hoja."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tex = cfg["texture"]
    return glcm_features(gray, mask, tex["glcm_levels"], tex["glcm_distance"]) | lbp_features(
        gray, mask, tex["lbp_points"], tex["lbp_radius"]
    )


# --------------------------------------------------------------------------- manchas


def spot_mask(image_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Máscara de **regiones candidatas**: píxeles de la hoja con tono anómalo, limpiados.

    Unión de amarillento, marrón y oscuro (``tone_masks``), apertura 3×3 para quitar píxeles sueltos
    y componentes de menos de ``min_area`` píxeles fuera. Es **evidencia visual de zonas
    sospechosas**, no la zona que «mira» el clasificador (regla 8 de ``AGENTS.md``).
    """
    tones = tone_masks(image_bgr, mask, cfg["tones"])
    candidates = seg.to_mask(tones["yellow"] | tones["brown"] | tones["dark"])
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, seg.ellipse_kernel(3))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= cfg["spots"]["min_area"])
    return seg.to_mask(np.isin(labels, keep[keep > 0]))


def spot_features(image_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> dict[str, float]:
    """Número, tamaño y forma de las regiones candidatas (Tema 6: componentes y regionprops).

    Áreas como fracción del área de la hoja (independiente del tamaño de la hoja en la foto).
    Forma media: solidez (área / área de la envolvente convexa), excentricidad de la elipse
    equivalente y circularidad 4πA/P² (1 = círculo). Sin regiones, todo vale 0.
    Muchas manchas pequeñas y redondeadas encajan con, p. ej., mancha bacteriana o septoria; pocas
    y grandes e irregulares, con tizones. Son hipótesis, no reglas.
    """
    regions = regionprops(label(spot_mask(image_bgr, mask, cfg) > 0, connectivity=2))
    leaf_area = (mask > 0).sum()
    areas = np.array([r.area for r in regions], dtype=np.float64) / leaf_area
    circularity = [
        min(1.0, 4 * math.pi * r.area / r.perimeter**2) if r.perimeter > 0 else 0.0 for r in regions
    ]
    return {
        "spot_count": float(len(regions)),
        "spot_area_frac": float(areas.sum()),
        "spot_mean_area_frac": float(areas.mean()) if len(areas) else 0.0,
        "spot_max_area_frac": float(areas.max()) if len(areas) else 0.0,
        "spot_mean_solidity": float(np.mean([r.solidity for r in regions])) if regions else 0.0,
        "spot_mean_eccentricity": float(np.mean([r.eccentricity for r in regions]))
        if regions
        else 0.0,
        "spot_mean_circularity": float(np.mean(circularity)) if regions else 0.0,
    }


# --------------------------------------------------------------------------- calidad


def quality_features(mask: np.ndarray) -> dict[str, float]:
    """Calidad de la máscara (D017). Sin regla de «no concluyente» para PlantVillage (D021):
    la solidez se mantiene como característica normal; el criterio se retoma en la fase 5 con
    PlantDoc. Motivo: las reglas candidatas solo marcaban hojas enfermas cuya forma es el propio
    síntoma (necrosis perdida, bordes irregulares).

    - ``leaf_area_frac``: fracción de la imagen ocupada por la hoja.
    - ``border_touch_frac``: fracción de los píxeles del borde de la imagen que son hoja.
    - ``mask_solidity``: área / área de la envolvente convexa de la mayor región (una máscara
      deshilachada o con trozos de fondo tiene solidez baja).
    """
    inside = mask > 0
    border = np.concatenate([inside[0], inside[-1], inside[1:-1, 0], inside[1:-1, -1]])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(biggest))
        solidity = cv2.contourArea(biggest) / hull_area if hull_area > 0 else 0.0
    return {
        "leaf_area_frac": float(inside.mean()),
        "border_touch_frac": float(border.mean()),
        "mask_solidity": float(solidity),
    }


def sharpness_feature(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Nitidez de la captura dentro de la hoja: varianza del laplaciano del gris.

    Es una **columna de diagnóstico** (familia ``diagnostico``), no una característica de síntoma.
    Permite detectar sesgos de captura (p. ej. lote de sanas más nítido que el de enfermas).
    Se calcula **antes** de cualquier suavizado para medir la nitidez real de la adquisición.
    Se usa una máscara erosionada 2 píxeles para evitar efectos de borde con el fondo.
    """
    inside = cv2.erode(mask, seg.ellipse_kernel(5)) > 0
    if not inside.any():
        return {"sharpness_laplacian_var": 0.0}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return {"sharpness_laplacian_var": float(laplacian[inside].var())}


# --------------------------------------------------------------------------- todo junto


def _zero_features(cfg: dict) -> dict[str, float]:
    """Vector de ceros con la misma estructura que extract_features (para mask_empty=True)."""
    # quality (3)
    zero = {
        "leaf_area_frac": 0.0,
        "border_touch_frac": 0.0,
        "mask_solidity": 0.0,
    }
    # diagnostico (1)
    zero["sharpness_laplacian_var"] = 0.0
    # color (43): stats + histograms + tone fractions
    zero.update(
        {
            "h_circ_mean_deg": 0.0,
            "h_circ_var": 0.0,
            "s_mean": 0.0,
            "s_std": 0.0,
            "v_mean": 0.0,
            "v_std": 0.0,
            "l_mean": 0.0,
            "l_std": 0.0,
            "a_mean": 0.0,
            "a_std": 0.0,
            "b_mean": 0.0,
            "b_std": 0.0,
        }
    )
    for i in range(12):
        zero[f"h_hist_{i:02d}"] = 0.0
    for name in ("a", "b"):
        for i in range(8):
            zero[f"{name}_hist_{i:02d}"] = 0.0
    zero.update({"frac_yellow": 0.0, "frac_brown": 0.0, "frac_dark": 0.0})
    # texture (22): GLCM (4) + LBP (18)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        zero[f"glcm_{prop}"] = 0.0
    for i in range(18):
        zero[f"lbp_{i:02d}"] = 0.0
    # spots (7)
    zero.update(
        {
            "spot_count": 0.0,
            "spot_area_frac": 0.0,
            "spot_mean_area_frac": 0.0,
            "spot_max_area_frac": 0.0,
            "spot_mean_solidity": 0.0,
            "spot_mean_eccentricity": 0.0,
            "spot_mean_circularity": 0.0,
        }
    )
    return zero


def gray_world(image_bgr: np.ndarray) -> np.ndarray:
    """Normalización de color *gray world* (constancia de color clásica, variante B′ de T08).

    Supone que la media de la escena es gris neutro: escala cada canal B, G y R para que su media
    en la imagen completa sea la media de los tres canales, y recorta a [0, 255] en ``uint8``.
    No se aprende de los datos. Punto débil: si la hoja ocupa casi toda la imagen, la «escena»
    es verde y la corrección también altera el color de la hoja.
    """
    pixels = image_bgr.astype(np.float64)
    means = pixels.reshape(-1, 3).mean(axis=0)
    gains = np.divide(means.mean(), means, out=np.ones(3), where=means > 0)
    return np.clip(np.rint(pixels * gains), 0, 255).astype(np.uint8)


def extract_features(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    cfg: dict,
    feature_image: np.ndarray | None = None,
) -> dict[str, float]:
    """Vector completo: calidad (máscara T03) + color, textura, manchas y diagnóstico.

    La calidad y el diagnóstico (nitidez) usan la máscara de T03 tal cual; el resto,
    la máscara preparada con la variante de sombra (``cfg["mask_variant"]``).

    ``feature_image`` (B′, T08): imagen de la que se sacan color, textura y manchas, p. ej. tras
    ``gray_world``. La máscara, su preparación, ``mask_empty``, la calidad (de la que sale el «no
    concluyente») y la nitidez salen siempre de ``image_bgr``, así que B y B′ comparten máscara y
    cobertura. Por defecto es ``image_bgr`` (B, sin cambios).

    Devuelve ``mask_empty`` (bool): True si la máscara final no tiene píxeles. En ese caso
    las demás características valdrán 0; la fase 4 decide el tratamiento. Evita NaN silenciosos
    (P2 de Codex).
    """
    mask = prepare_mask(image_bgr, raw_mask, cfg["mask_variant"])
    mask_empty = not (mask > 0).any()
    if mask_empty:
        return {"mask_empty": True} | _zero_features(cfg)
    colour = image_bgr if feature_image is None else feature_image
    return (
        {"mask_empty": False}
        | quality_features(raw_mask)
        | sharpness_feature(image_bgr, raw_mask)
        | color_features(colour, mask, cfg)
        | texture_features(colour, mask, cfg)
        | spot_features(colour, mask, cfg)
    )


def is_inconclusive(quality: dict[str, float], rules: dict) -> bool:
    """«No concluyente» si la máscara incumple alguna de las reglas activas (fijadas con train).

    Reglas posibles: ``leaf_area_frac_min``, ``mask_solidity_min`` y ``border_touch_frac_max``.
    Solo se aplican las que están en ``rules`` (``configs/features.json``).

    Nota: Para PlantVillage no se usa (D021). Se mantiene por compatibilidad y para la fase 5
    (PlantDoc), donde se decidirá si se activa.
    """
    checks = {
        "leaf_area_frac_min": lambda t: quality["leaf_area_frac"] < t,
        "mask_solidity_min": lambda t: quality["mask_solidity"] < t,
        "border_touch_frac_max": lambda t: quality["border_touch_frac"] > t,
    }
    return any(checks[name](threshold) for name, threshold in rules.items())
