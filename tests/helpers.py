"""Rostros sintéticos para probar el pipeline sin cámara ni modelo."""
from __future__ import annotations

import numpy as np

from alertdrive.vision.landmarks import (
    LEFT_EYE_EAR,
    MOUTH_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    RIGHT_EYE_EAR,
    FaceObservation,
)


def synthetic_points(ear: float = 0.30, mar: float = 0.05) -> np.ndarray:
    """478 landmarks donde los ojos tienen el EAR pedido y la boca el MAR pedido."""
    pts = np.zeros((478, 2))
    h = ear * 5.0  # con ancho de ojo 10: EAR = 4h / 20
    for (p1, p2, p3, p4, p5, p6), x0 in ((RIGHT_EYE_EAR, 100.0), (LEFT_EYE_EAR, 200.0)):
        y0 = 100.0
        pts[p1] = (x0, y0)
        pts[p4] = (x0 + 10, y0)
        pts[p2] = (x0 + 3, y0 - h)
        pts[p3] = (x0 + 7, y0 - h)
        pts[p5] = (x0 + 7, y0 + h)
        pts[p6] = (x0 + 3, y0 + h)
    left, right = MOUTH_CORNERS
    pts[left] = (140.0, 200.0)
    pts[right] = (180.0, 200.0)
    v = mar * 40.0
    for (a, b), x in zip(MOUTH_VERTICAL_PAIRS, (150.0, 160.0, 170.0)):
        pts[a] = (x, 200.0 - v / 2)
        pts[b] = (x, 200.0 + v / 2)
    return pts


def synthetic_obs(t: float, ear: float = 0.30, mar: float = 0.05, transform: np.ndarray | None = None) -> FaceObservation:
    return FaceObservation(
        timestamp_s=t,
        face_present=True,
        landmarks_px=synthetic_points(ear, mar),
        transform=np.eye(4) if transform is None else transform,
        blendshapes={"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1, "jawOpen": mar},
        image_size=(640, 480),
    )
