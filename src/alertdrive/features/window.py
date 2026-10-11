"""Agregación de características por ventana deslizante.

Un solo frame no basta para decidir si alguien está somnoliento (un parpadeo
cierra los ojos). Los indicadores usados en la literatura son temporales:
PERCLOS, frecuencia y duración de parpadeo, bostezos por minuto, cabeceos.
Este módulo resume los frames de los últimos ``window_s`` segundos en un
vector de características que consumen los clasificadores.
"""
from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Callable, Iterable

import numpy as np

from alertdrive.config import FeatureConfig
from alertdrive.features.extractor import FrameFeatures
from alertdrive.features.temporal import Event

# Orden fijo de columnas que usarán los modelos (Fase 2).
FEATURE_COLUMNS: tuple[str, ...] = (
    "face_present_ratio",
    "ear_mean", "ear_std", "ear_min", "ear_rel_mean", "ear_rel_min",
    "perclos", "blink_rate_per_min", "blink_dur_mean_s",
    "long_closure_count", "microsleep_count", "max_closure_s",
    "mar_mean", "mar_max", "mouth_open_fraction", "yawn_count",
    "pitch_dev_mean", "pitch_dev_std", "yaw_dev_std", "roll_abs_mean",
    "head_off_fraction", "looking_away_fraction", "nod_count",
    "bs_eye_blink_mean", "bs_jaw_open_max",
)
META_COLUMNS: tuple[str, ...] = ("t_start", "t_end", "n_frames", "valid")


def _stat(values: Iterable[float], fn: Callable[[np.ndarray], float]) -> float:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(fn(arr))


class WindowAggregator:
    """Mantiene los últimos frames y calcula las características de ventana.

    - ``add(frame)`` devuelve un diccionario cada ``stride_s`` segundos una vez
      llena la primera ventana (modo dataset).
    - ``compute(t_end)`` calcula la ventana en cualquier instante (modo en vivo).
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._frames: deque[FrameFeatures] = deque()
        self._events: deque[Event] = deque()
        self._first_t: float | None = None
        self._next_emit: float | None = None

    def add(self, frame: FrameFeatures) -> dict[str, float] | None:
        self.add_events(frame.events)
        self._frames.append(frame)
        if self._first_t is None:
            self._first_t = frame.t
            self._next_emit = frame.t + self.config.window.window_s
        self._prune(frame.t)

        assert self._next_emit is not None
        if frame.t >= self._next_emit:
            self._next_emit += self.config.window.stride_s
            # Si hubo un salto grande de tiempo, se realinea el siguiente disparo.
            if self._next_emit <= frame.t:
                self._next_emit = frame.t + self.config.window.stride_s
            return self.compute(frame.t)
        return None

    def add_events(self, events: Iterable[Event]) -> None:
        self._events.extend(events)

    def _prune(self, t: float) -> None:
        horizon = t - self.config.window.window_s
        while self._frames and self._frames[0].t <= horizon:
            self._frames.popleft()
        while self._events and self._events[0].end_s <= horizon:
            self._events.popleft()

    def compute(self, t_end: float) -> dict[str, float]:
        w = self.config.window
        eye, mouth = self.config.eye, self.config.mouth
        t0 = t_end - w.window_s
        frames = [f for f in self._frames if t0 < f.t <= t_end]
        events = [e for e in self._events if t0 < e.end_s <= t_end]
        present = [f for f in frames if f.face_present]
        n, n_present = len(frames), len(present)

        span_s = min(w.window_s, t_end - frames[0].t) if frames else 0.0
        span_min = max(span_s, 1e-6) / 60.0

        closures = [e for e in events if e.kind == "eye_closure"]
        blinks = [e for e in closures if e.duration_s <= eye.max_blink_s]
        long_closures = [e for e in closures if e.duration_s > eye.max_blink_s]
        ongoing_closure = frames[-1].closure_ongoing_s if frames else 0.0
        ongoing_yawn = frames[-1].yawn_ongoing_s if frames else 0.0

        def frac(pred: Callable[[FrameFeatures], bool]) -> float:
            return sum(1 for f in present if pred(f)) / n_present if n_present else math.nan

        face_ratio = n_present / n if n else 0.0
        feats: dict[str, float] = {
            "t_start": t0,
            "t_end": t_end,
            "n_frames": float(n),
            "valid": float(face_ratio >= w.min_face_ratio),
            "face_present_ratio": face_ratio,
            "ear_mean": _stat((f.ear for f in present), np.mean),
            "ear_std": _stat((f.ear for f in present), np.std),
            "ear_min": _stat((f.ear for f in present), np.min),
            "ear_rel_mean": _stat((f.ear_rel for f in present), np.mean),
            "ear_rel_min": _stat((f.ear_rel for f in present), np.min),
            "perclos": frac(lambda f: f.eye_closed),
            "blink_rate_per_min": len(blinks) / span_min,
            "blink_dur_mean_s": _stat((e.duration_s for e in blinks), np.mean) if blinks else 0.0,
            "long_closure_count": float(len(long_closures) + (ongoing_closure > eye.max_blink_s)),
            "microsleep_count": float(
                sum(e.duration_s >= eye.microsleep_s for e in closures) + (ongoing_closure >= eye.microsleep_s)
            ),
            "max_closure_s": max([e.duration_s for e in closures] + [ongoing_closure]),
            "mar_mean": _stat((f.mar for f in present), np.mean),
            "mar_max": _stat((f.mar for f in present), np.max),
            "mouth_open_fraction": frac(lambda f: f.mouth_open),
            "yawn_count": float(
                sum(e.kind == "yawn" for e in events) + (ongoing_yawn >= mouth.min_yawn_s)
            ),
            "pitch_dev_mean": _stat((f.pitch_dev for f in present), np.mean),
            "pitch_dev_std": _stat((f.pitch_dev for f in present), np.std),
            "yaw_dev_std": _stat((f.yaw_dev for f in present), np.std),
            "roll_abs_mean": _stat((abs(f.roll) for f in present), np.mean),
            "head_off_fraction": frac(lambda f: f.head_off),
            "looking_away_fraction": frac(lambda f: f.looking_away),
            "nod_count": float(sum(e.kind == "nod" for e in events)),
            "bs_eye_blink_mean": _stat((f.bs_eye_blink for f in present), np.mean),
            "bs_jaw_open_max": _stat((f.bs_jaw_open for f in present), np.max),
        }
        # Protege contra olvidar una columna al modificar el código.
        assert set(FEATURE_COLUMNS) <= feats.keys()
        return feats
