"""Pipeline completo: video → frames → landmarks → características por ventana."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from alertdrive.config import FeatureConfig
from alertdrive.features.extractor import FrameFeatureExtractor
from alertdrive.features.window import WindowAggregator
from alertdrive.vision.face_landmarker import FaceLandmarkDetector

logger = logging.getLogger(__name__)


def read_frame_labels(path: str | Path) -> np.ndarray:
    """Lee anotaciones por frame en formato de dígitos (p. ej. NTHU-DDD: "0001110...")."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    digits = [int(c) for c in text if c.isdigit()]
    return np.asarray(digits, dtype=np.int8)


def video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if 1.0 <= fps <= 240.0 else 30.0


def iter_video_frames(
    path: str | Path, sample_fps: float | None = 10.0, max_seconds: float | None = None
) -> Iterator[tuple[int, float, float, np.ndarray]]:
    """Genera (índice_frame, tiempo_s, fps, frame_bgr) submuestreando a ``sample_fps``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"No se pudo abrir el video: {path}")
    fps = video_fps(cap)
    step = max(1, round(fps / sample_fps)) if sample_fps else 1
    idx = 0
    try:
        while cap.grab():
            t = idx / fps
            if max_seconds is not None and t > max_seconds:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, t, fps, frame
            idx += 1
    finally:
        cap.release()


def extract_video_windows(
    video_path: str | Path,
    config: FeatureConfig | None = None,
    sample_fps: float | None = 10.0,
    max_seconds: float | None = None,
    frame_labels: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Procesa un video completo y devuelve una fila de características por ventana.

    Si se entregan ``frame_labels`` (anotación por frame), cada ventana incluye
    ``label_fraction``: la fracción de frames anotados como positivos.
    """
    config = config or FeatureConfig()
    extractor = FrameFeatureExtractor(config)
    aggregator = WindowAggregator(config)
    rows: list[dict[str, float]] = []
    n_frames = 0

    with FaceLandmarkDetector(config.landmarker) as detector:
        for idx, t, fps, frame in iter_video_frames(video_path, sample_fps, max_seconds):
            obs, _ = detector.process(frame, t)
            window = aggregator.add(extractor.update(obs))
            n_frames += 1
            if window is None:
                continue
            if frame_labels is not None and frame_labels.size:
                lo = max(0, int(window["t_start"] * fps) + 1)
                hi = min(frame_labels.size, idx + 1)
                window["label_fraction"] = float(np.mean(frame_labels[lo:hi] > 0)) if hi > lo else np.nan
            rows.append(window)

    logger.info("%s: %d frames procesados, %d ventanas", Path(video_path).name, n_frames, len(rows))
    return rows
