"""Dibujo sobre el frame: contornos de ojos y boca, medidas y borde de estado."""
from __future__ import annotations

import math

import cv2
import numpy as np

from alertdrive.features.extractor import FrameFeatures
from alertdrive.vision.landmarks import (
    INNER_LIPS_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    FaceObservation,
)

# Colores BGR por nivel de alerta (mismos tonos que la interfaz).
LEVEL_BGR = {
    "calibrando": (192, 101, 21),
    "sin_rostro": (122, 110, 84),
    "ok": (50, 125, 46),
    "atencion": (37, 168, 249),
    "alarma": (40, 40, 198),
}
_GREEN = (80, 200, 80)
_RED = (60, 60, 230)
_ORANGE = (40, 160, 255)


def _poly(img: np.ndarray, pts: np.ndarray, idx: tuple[int, ...], color: tuple[int, int, int]) -> None:
    contour = pts[list(idx)].round().astype(np.int32).reshape(-1, 1, 2)
    thickness = 1 if img.shape[1] < 480 else 2
    cv2.polylines(img, [contour], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _fmt(v: float, nd: int = 2) -> str:
    return "--" if v is None or not math.isfinite(v) else f"{v:.{nd}f}"


def draw_overlay(frame_bgr: np.ndarray, obs: FaceObservation, ff: FrameFeatures, level: str) -> np.ndarray:
    """Devuelve una copia del frame con la malla relevante y un borde de color."""
    img = frame_bgr.copy()
    if obs.face_present and obs.landmarks_px is not None:
        pts = obs.landmarks_px
        eye_color = _RED if ff.eye_closed else _GREEN
        _poly(img, pts, RIGHT_EYE_CONTOUR, eye_color)
        _poly(img, pts, LEFT_EYE_CONTOUR, eye_color)
        _poly(img, pts, INNER_LIPS_CONTOUR, _ORANGE if ff.mouth_open else _GREEN)
        text = f"EAR {_fmt(ff.ear)}  MAR {_fmt(ff.mar)}  cabeza {_fmt(ff.pitch_dev, 0)} grados"
    else:
        text = "Rostro no detectado"

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 28), (w, h), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    border = max(6, w // 80)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT,
                              value=LEVEL_BGR.get(level, (128, 128, 128)))
