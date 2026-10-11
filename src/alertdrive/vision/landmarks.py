"""Índices de la malla facial de MediaPipe (478 puntos) y estructura de observación."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Orden [p1, p2, p3, p4, p5, p6] de la fórmula EAR de Soukupová & Čech (2016):
# p1/p4 = comisuras, p2/p3 = párpado superior, p6/p5 = párpado inferior.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Labio interno: comisuras y tres pares verticales (izquierda, centro, derecha).
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICAL_PAIRS = ((82, 87), (13, 14), (312, 317))

# Contornos para dibujar en la interfaz.
RIGHT_EYE_CONTOUR = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
LEFT_EYE_CONTOUR = (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382)
INNER_LIPS_CONTOUR = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)


@dataclass(frozen=True)
class FaceObservation:
    """Resultado de procesar un frame.

    Attributes:
        timestamp_s: tiempo del frame en segundos (video: índice/fps; webcam: reloj).
        face_present: si se detectó un rostro.
        landmarks_px: (478, 2) en píxeles del frame procesado, o None.
        transform: matriz 4×4 de transformación facial (pose de la cabeza), o None.
        blendshapes: coeficientes de expresión de MediaPipe (p. ej. 'eyeBlinkLeft').
        image_size: (ancho, alto) del frame procesado.
    """

    timestamp_s: float
    face_present: bool
    landmarks_px: np.ndarray | None = None
    transform: np.ndarray | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)

    @staticmethod
    def absent(timestamp_s: float, image_size: tuple[int, int] = (0, 0)) -> "FaceObservation":
        return FaceObservation(timestamp_s=timestamp_s, face_present=False, image_size=image_size)
