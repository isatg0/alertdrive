"""Motor de inferencia en vivo: frame → landmarks → características → predicción.

Usa exactamente los mismos componentes que ``pipeline.extract_video_windows``
(FaceLandmarkDetector, FrameFeatureExtractor, WindowAggregator), de modo que
la plataforma calcula las características igual que en el entrenamiento.
La única diferencia es que aquí la ventana se evalúa cada ``predict_every_s``
segundos con ``WindowAggregator.compute`` en lugar de cada ``stride_s``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alertdrive.classify.predictor import DrowsinessPredictor
from alertdrive.config import FeatureConfig
from alertdrive.features.extractor import FrameFeatureExtractor, FrameFeatures
from alertdrive.features.window import WindowAggregator
from alertdrive.vision.face_landmarker import FaceLandmarkDetector
from alertdrive.vision.landmarks import FaceObservation


def _single_thread(obj: Any, seen: set[int] | None = None, depth: int = 0) -> None:
    """Pone ``n_jobs = 1`` en todo estimador que lo tenga dentro de ``obj``.

    Con ``n_jobs=-1`` el Random Forest arranca un grupo de hilos en cada
    predicción. Para una sola ventana por segundo ese arranque cuesta mucho más
    que la predicción misma (en Windows, cientos de milisegundos), así que en
    inferencia se usa un solo hilo. El entrenamiento no cambia.
    """
    seen = set() if seen is None else seen
    if depth > 6 or id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, bool, type)):
        return
    if isinstance(obj, np.ndarray):
        return
    seen.add(id(obj))
    if hasattr(obj, "n_jobs") and not callable(getattr(obj, "n_jobs")):
        try:
            obj.n_jobs = 1
        except (AttributeError, TypeError):
            pass
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        return
    for child in children:
        _single_thread(child, seen, depth + 1)


def load_predictor(path: str) -> DrowsinessPredictor:
    """Carga un modelo de la Fase 2 listo para inferencia en vivo (un solo hilo)."""
    predictor = DrowsinessPredictor.load(path)
    _single_thread(predictor)
    return predictor


@dataclass
class WindowPrediction:
    """Última predicción sobre la ventana deslizante."""

    t: float = -math.inf
    ready: bool = False          # ya pasó una ventana completa desde el inicio
    valid: bool = False          # la ventana tuvo suficiente rostro visible
    is_drowsy: bool = False
    score: float = math.nan
    reasons: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class EngineOutput:
    """Resultado de procesar un frame."""

    t: float
    frame: np.ndarray            # frame reescalado que vio MediaPipe (para dibujar)
    obs: FaceObservation
    frame_features: FrameFeatures
    prediction: WindowPrediction
    new_prediction: bool         # True si en este frame se recalculó la ventana


class DrowsinessEngine:
    """Procesa frames uno a uno y mantiene el estado temporal de la sesión.

    Una instancia por sesión de webcam o por video: MediaPipe exige
    timestamps crecientes y las líneas base son de cada conductor.

    Ejemplo::

        engine = DrowsinessEngine("models/drowsiness_best.joblib", threshold=0.7)
        out = engine.process(frame_bgr, t)   # t en segundos, ~5 frames por segundo
        out.prediction.is_drowsy, out.prediction.reasons
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: FeatureConfig | None = None,
        predict_every_s: float = 1.0,
        predictor: Any | None = None,
        detector: Any | None = None,
        threshold: float | None = None,
    ) -> None:
        if predictor is None and model_path is None:
            raise ValueError("Se necesita model_path o un predictor ya cargado")
        self.config = config or FeatureConfig()
        self.predict_every_s = predict_every_s
        self.predictor = predictor or load_predictor(model_path)
        # Puntaje mínimo para considerar una predicción como fatiga. None: se
        # usa la decisión del modelo (su propio umbral de la Fase 2).
        self.threshold = threshold
        self.detector = detector or FaceLandmarkDetector(self.config.landmarker)
        self.extractor = FrameFeatureExtractor(self.config)
        self.aggregator = WindowAggregator(self.config)
        self._t0: float | None = None
        self._last = WindowPrediction()

    @property
    def window_s(self) -> float:
        return self.config.window.window_s

    def process(self, frame_bgr: np.ndarray, t: float) -> EngineOutput:
        """Procesa un frame tomado en el instante ``t`` (segundos, creciente)."""
        if self._t0 is None:
            self._t0 = t

        obs, frame = self.detector.process(frame_bgr, t)
        ff = self.extractor.update(obs)
        # add() devuelve una ventana cada stride_s (modo dataset); aquí se ignora
        # y se evalúa la ventana con más frecuencia mediante compute().
        self.aggregator.add(ff)

        new_prediction = (t - self._last.t) >= self.predict_every_s
        if new_prediction:
            self._last = self._predict(t)
        return EngineOutput(t, frame, obs, ff, self._last, new_prediction)

    def _predict(self, t: float) -> WindowPrediction:
        feats = self.aggregator.compute(t)
        res = self.predictor.predict(feats)
        assert self._t0 is not None
        valid = bool(getattr(res, "valid", feats.get("valid", 1.0)))
        score = float(res.score)
        if self.threshold is None:
            is_drowsy = bool(res.is_drowsy)
        else:
            is_drowsy = valid and math.isfinite(score) and score >= self.threshold
        return WindowPrediction(
            t=t,
            ready=(t - self._t0) >= self.window_s,
            valid=valid,
            is_drowsy=is_drowsy,
            score=score,
            reasons=tuple(res.reasons),
            features=feats,
        )

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "DrowsinessEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
