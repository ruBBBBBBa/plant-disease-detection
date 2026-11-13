"""Tests de las características (T04) con imágenes sintéticas; nunca usan los datos reales."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import features as ft

SIZE = 128
CONFIG = {
    "mask_variant": {"kind": "raw"},
    "tones": {"v_dark": 15, "s_min": 40, "v_hue_min": 50, "h_brown_yellow": 20, "h_green_min": 29},
    "hist_ranges": {"a": [-27.0, 5.0], "b": [1.0, 48.0]},
    "texture": {"glcm_levels": 32, "glcm_distance": 1, "lbp_points": 16, "lbp_radius": 2},
    "spots": {"min_area": 10},
}
GREEN = (40, 150, 50)  # BGR, H ≈ 62 (escala OpenCV)
BROWN = (30, 70, 120)  # BGR, H ≈ 10


def leaf(background: tuple[int, int, int] = (150, 150, 150), seed: int = 0):
    """Elipse verde con textura leve sobre un fondo dado. Devuelve (imagen, máscara 0/255)."""
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), background, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (64, 64), (44, 30), 15, 0, 360, 255, -1)
    texture = rng.integers(-10, 11, (SIZE, SIZE, 1))
    green = np.clip(np.array(GREEN, np.int16) + texture, 0, 255).astype(np.uint8)
    image[mask > 0] = green[mask > 0]
    return image, mask


def test_uniform_green_leaf_has_no_anomalous_tones_or_spots() -> None:
    """Una hoja verde: fracción de tonos anómalos ≈ 0 y ninguna región candidata."""
    image, mask = leaf()
    feats = ft.extract_features(image, mask, CONFIG)
    assert feats["frac_yellow"] + feats["frac_brown"] + feats["frac_dark"] < 0.01
    assert feats["spot_count"] == 0 and feats["spot_area_frac"] == 0


def test_brown_spot_of_known_area() -> None:
    """Una mancha marrón circular: una componente y fracción de área ≈ área conocida."""
    image, mask = leaf()
    spot = np.zeros_like(mask)
    cv2.circle(spot, (60, 64), 8, 255, -1)
    image[spot > 0] = BROWN
    feats = ft.extract_features(image, mask, CONFIG)
    expected = (spot > 0).sum() / (mask > 0).sum()
    assert feats["spot_count"] == 1
    assert feats["frac_brown"] == pytest.approx(expected, rel=0.05)
    assert feats["spot_area_frac"] == pytest.approx(expected, rel=0.15)
    assert feats["spot_mean_circularity"] > 0.7


def test_background_does_not_change_features() -> None:
    """Misma hoja y misma máscara sobre dos fondos distintos → mismo vector."""
    gray_bg, mask = leaf((150, 150, 150))
    blue_bg, _ = leaf((200, 60, 20))
    assert ft.extract_features(gray_bg, mask, CONFIG) == ft.extract_features(blue_bg, mask, CONFIG)


def test_feature_names_and_order_are_stable() -> None:
    """Dos imágenes distintas dan exactamente los mismos nombres en el mismo orden."""
    a = list(ft.extract_features(*leaf(seed=1), CONFIG))
    image, mask = leaf(seed=2)
    image[50:60, 50:60] = BROWN
    b = list(ft.extract_features(image, mask, CONFIG))
    assert a == b
    assert a[:3] == ["leaf_area_frac", "border_touch_frac", "mask_solidity"]
    assert a[-1] == "spot_mean_circularity"
    assert len(a) == 3 + 12 + 12 + 8 + 8 + 3 + 4 + 18 + 7


def test_shadow_variant_keeps_interior_lesion_and_removes_rim_shadow() -> None:
    """La sombra oscura que entra por el borde se quita; una lesión oscura interior se conserva."""
    image, mask = leaf()
    lesion = np.zeros_like(mask)
    cv2.circle(lesion, (64, 64), 6, 255, -1)
    shadow = np.zeros_like(mask)
    cv2.rectangle(shadow, (90, 40), (130, 90), 255, -1)
    shadow &= mask
    image[lesion > 0] = (30, 30, 30)  # oscuro y poco saturado
    image[shadow > 0] = (35, 35, 40)
    cleaned = ft.remove_shadow(image, mask, v_max=70, s_max=60)
    assert (cleaned[lesion > 0] == 255).all()
    assert (cleaned[shadow > 0] == 0).all()


def test_glcm_ignores_background_pairs() -> None:
    """Cambiar solo el fondo no cambia la GLCM (nivel 0 reservado y eliminado)."""
    image, mask = leaf()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    other = gray.copy()
    other[mask == 0] = 255
    assert ft.glcm_features(gray, mask, 32, 1) == ft.glcm_features(other, mask, 32, 1)


def test_hue_mean_is_circular() -> None:
    """Rojos a ambos lados de H = 0 dan una media circular cerca de 0°, no de 180°."""
    image = np.zeros((10, 10, 3), np.uint8)
    hsv = np.full((10, 10, 3), (2, 200, 200), np.uint8)
    hsv[:, 5:, 0] = 177
    image[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    feats = ft.color_features(image, np.full((10, 10), 255, np.uint8), CONFIG)
    angle = feats["h_circ_mean_deg"]
    assert min(angle, 360 - angle) < 10


def test_is_inconclusive_only_uses_active_rules() -> None:
    """Solo cuentan las reglas presentes en la configuración."""
    quality = {"leaf_area_frac": 0.1, "border_touch_frac": 0.5, "mask_solidity": 0.9}
    assert not ft.is_inconclusive(quality, {"mask_solidity_min": 0.65})
    assert ft.is_inconclusive(quality, {"mask_solidity_min": 0.65, "leaf_area_frac_min": 0.2})
