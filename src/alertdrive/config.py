"""Configuración central del sistema.

Todos los umbrales viven aquí para que el pipeline de entrenamiento y el de
inferencia en vivo usen exactamente los mismos valores. Cambiar un umbral en un
solo sitio evita que el modelo evaluado difiera del desplegado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass(frozen=True)
class LandmarkerConfig:
    """Parámetros del detector de rostro y landmarks (MediaPipe FaceLandmarker)."""

    model_path: Path = DEFAULT_MODEL_PATH
    num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_width: int = 640  # se reescala el frame antes de detectar, por velocidad


@dataclass(frozen=True)
class EyeConfig:
    """Detección de cierre de ojos basada en EAR con línea base adaptativa.

    El EAR con ojos abiertos varía entre personas (≈0.25–0.35), por eso el
    umbral se expresa relativo a la línea base del propio conductor. Mientras
    no hay suficientes muestras se usa el umbral absoluto clásico.
    """

    abs_closed_ear: float = 0.20
    closed_ratio: float = 0.70  # cerrado si EAR < 0.70 × línea base
    baseline_percentile: float = 90.0
    baseline_window_s: float = 30.0
    baseline_min_samples: int = 30
    max_blink_s: float = 0.5  # cierres más largos cuentan como "cierre prolongado"
    microsleep_s: float = 1.0


@dataclass(frozen=True)
class MouthConfig:
    """Detección de bostezos basada en MAR (labio interno)."""

    yawn_mar: float = 0.55
    min_yawn_s: float = 1.2  # hablar produce aperturas cortas; un bostezo dura más
    merge_gap_s: float = 0.4  # une aperturas separadas por parpadeos del detector


@dataclass(frozen=True)
class HeadConfig:
    """Desviación de la pose de la cabeza respecto a la línea base del conductor."""

    baseline_window_s: float = 30.0
    baseline_min_samples: int = 30
    nod_deg: float = 15.0  # |pitch - base| mayor a esto = cabeza fuera de posición
    min_nod_s: float = 0.3
    off_road_yaw_deg: float = 25.0  # mirada fuera de la vía


@dataclass(frozen=True)
class WindowConfig:
    """Ventanas deslizantes sobre las que se calculan las características."""

    window_s: float = 15.0
    stride_s: float = 5.0
    min_face_ratio: float = 0.5  # ventanas con menos rostro visible se marcan inválidas


@dataclass(frozen=True)
class FeatureConfig:
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
