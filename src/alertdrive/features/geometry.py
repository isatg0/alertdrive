"""Medidas geométricas por frame: EAR, MAR y ángulos de la cabeza."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from alertdrive.vision.landmarks import (
    LEFT_EYE_EAR,
    MOUTH_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    RIGHT_EYE_EAR,
)

_EPS = 1e-6


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: Sequence[int]) -> float:
    """EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)  (Soukupová & Čech, 2016).

    ≈0.25–0.35 con el ojo abierto y cercano a 0 cerrado.
    """
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal < _EPS:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def eyes_ear(points: np.ndarray) -> tuple[float, float]:
    """Devuelve (EAR ojo derecho, EAR ojo izquierdo) del conductor."""
    return eye_aspect_ratio(points, RIGHT_EYE_EAR), eye_aspect_ratio(points, LEFT_EYE_EAR)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """MAR = promedio de las 3 aperturas verticales del labio interno / ancho de boca."""
    left, right = (points[i] for i in MOUTH_CORNERS)
    horizontal = _dist(left, right)
    if horizontal < _EPS:
        return 0.0
    vertical = np.mean([_dist(points[a], points[b]) for a, b in MOUTH_VERTICAL_PAIRS])
    return float(vertical / horizontal)


def rotation_to_euler_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convierte una matriz de rotación 3×3 en (pitch, yaw, roll) en grados.

    Convención R = Rz(roll) · Ry(yaw) · Rx(pitch): pitch = cabeceo (arriba/abajo),
    yaw = giro lateral, roll = inclinación hacia el hombro. El signo absoluto
    depende de la convención de MediaPipe; por eso el sistema trabaja con la
    desviación respecto a la línea base de cada conductor.
    """
    r = np.asarray(rotation, dtype=np.float64)[:3, :3]
    sy = -r[2, 0]
    sy = max(-1.0, min(1.0, sy))
    yaw = math.asin(sy)
    if abs(math.cos(yaw)) > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # bloqueo de cardán: se fija roll = 0
        pitch = math.atan2(-r[1, 2], r[1, 1])
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def euler_to_rotation_deg(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Inversa de :func:`rotation_to_euler_deg` (útil para pruebas)."""
    p, y, r = (math.radians(v) for v in (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx
