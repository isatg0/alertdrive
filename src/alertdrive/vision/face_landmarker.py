"""Envoltorio de MediaPipe FaceLandmarker (API Tasks).

Se usa la API Tasks porque la API antigua ``mp.solutions`` fue retirada de
MediaPipe 0.10.x recientes. FaceLandmarker además entrega la matriz de
transformación facial (pose de la cabeza) y blendshapes, sin necesidad de
resolver PnP con un modelo 3D genérico.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from alertdrive.config import LandmarkerConfig
from alertdrive.vision.landmarks import FaceObservation

logger = logging.getLogger(__name__)


class FaceLandmarkDetector:
    """Detecta el rostro y sus 478 landmarks en frames BGR de OpenCV.

    Usa el modo VIDEO de MediaPipe (síncrono, aprovecha el seguimiento entre
    frames). Una instancia por video o por sesión de webcam, porque los
    timestamps deben ser estrictamente crecientes.

    Ejemplo::

        with FaceLandmarkDetector() as detector:
            obs = detector.process(frame_bgr, timestamp_s=0.0)
    """

    def __init__(self, config: LandmarkerConfig | None = None) -> None:
        self.config = config or LandmarkerConfig()
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. "
                "Ejecuta: python scripts/download_model.py"
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=self.config.num_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.config.max_width and w > self.config.max_width:
            scale = self.config.max_width / w
            frame = cv2.resize(frame, (self.config.max_width, round(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> tuple[FaceObservation, np.ndarray]:
        """Procesa un frame y devuelve (observación, frame reescalado usado)."""
        frame = self._resize(frame_bgr)
        h, w = frame.shape[:2]

        ts_ms = int(round(timestamp_s * 1000))
        if ts_ms <= self._last_ts_ms:  # MediaPipe exige timestamps crecientes
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        if not result.face_landmarks:
            return FaceObservation.absent(timestamp_s, (w, h)), frame

        lms = result.face_landmarks[0]
        # Se convierte a píxeles para que las distancias no se distorsionen
        # por la relación de aspecto del frame.
        pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float64)
        transform = (
            np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
            if result.facial_transformation_matrixes
            else None
        )
        blendshapes = (
            {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
            if result.face_blendshapes
            else {}
        )
        obs = FaceObservation(
            timestamp_s=timestamp_s,
            face_present=True,
            landmarks_px=pts,
            transform=transform,
            blendshapes=blendshapes,
            image_size=(w, h),
        )
        return obs, frame

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
