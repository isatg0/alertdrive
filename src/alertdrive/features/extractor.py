"""Extractor de características por frame.

Recibe observaciones del detector (una por frame) y mantiene el estado
temporal necesario: líneas base adaptativas y detectores de eventos. Es el
mismo código para procesar datasets (entrenamiento) y la webcam (inferencia),
lo que garantiza que el modelo vea en producción las mismas características
con las que fue evaluado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from alertdrive.config import FeatureConfig
from alertdrive.features.geometry import eyes_ear, mouth_aspect_ratio, rotation_to_euler_deg
from alertdrive.features.temporal import AdaptiveBaseline, Event, EventDetector
from alertdrive.vision.landmarks import FaceObservation

NAN = math.nan


@dataclass(frozen=True)
class FrameFeatures:
    t: float
    face_present: bool
    ear: float = NAN
    ear_right: float = NAN
    ear_left: float = NAN
    ear_rel: float = NAN  # EAR / línea base del conductor
    ear_threshold: float = NAN
    eye_closed: bool = False
    closure_ongoing_s: float = 0.0
    mar: float = NAN
    mouth_open: bool = False
    yawn_ongoing_s: float = 0.0
    pitch: float = NAN
    yaw: float = NAN
    roll: float = NAN
    pitch_dev: float = NAN
    yaw_dev: float = NAN
    head_off: bool = False  # cabeceo: |pitch − base| > umbral
    looking_away: bool = False  # |yaw − base| > umbral
    bs_eye_blink: float = NAN  # blendshape de MediaPipe (promedio ambos ojos)
    bs_jaw_open: float = NAN
    events: tuple[Event, ...] = field(default_factory=tuple)


class FrameFeatureExtractor:
    """Convierte observaciones faciales en características por frame con estado."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        eye, head, mouth = self.config.eye, self.config.head, self.config.mouth
        self._ear_base = AdaptiveBaseline(eye.baseline_window_s, eye.baseline_percentile, eye.baseline_min_samples)
        self._pitch_base = AdaptiveBaseline(head.baseline_window_s, 50.0, head.baseline_min_samples)
        self._yaw_base = AdaptiveBaseline(head.baseline_window_s, 50.0, head.baseline_min_samples)
        self._closure = EventDetector("eye_closure")
        self._yawn = EventDetector("yawn", merge_gap_s=mouth.merge_gap_s)
        self._nod = EventDetector("nod", merge_gap_s=0.2)

    def update(self, obs: FaceObservation) -> FrameFeatures:
        t = obs.timestamp_s
        if not obs.face_present or obs.landmarks_px is None:
            events = self._collect_events(t, closed=False, mouth_open=False, head_off=False)
            return FrameFeatures(t=t, face_present=False, events=events)

        eye_cfg, mouth_cfg, head_cfg = self.config.eye, self.config.mouth, self.config.head
        pts = obs.landmarks_px

        # --- Ojos -------------------------------------------------------
        ear_r, ear_l = eyes_ear(pts)
        ear = (ear_r + ear_l) / 2.0
        self._ear_base.update(t, ear)
        base = self._ear_base.value
        threshold = eye_cfg.closed_ratio * base if base else eye_cfg.abs_closed_ear
        eye_closed = ear < threshold
        ear_rel = ear / base if base else NAN

        # --- Boca -------------------------------------------------------
        mar = mouth_aspect_ratio(pts)
        mouth_open = mar > mouth_cfg.yawn_mar

        # --- Cabeza -----------------------------------------------------
        pitch = yaw = roll = pitch_dev = yaw_dev = NAN
        head_off = looking_away = False
        if obs.transform is not None:
            pitch, yaw, roll = rotation_to_euler_deg(obs.transform)
            self._pitch_base.update(t, pitch)
            self._yaw_base.update(t, yaw)
            if self._pitch_base.value is not None and self._yaw_base.value is not None:
                pitch_dev = pitch - self._pitch_base.value
                yaw_dev = yaw - self._yaw_base.value
                head_off = abs(pitch_dev) > head_cfg.nod_deg
                looking_away = abs(yaw_dev) > head_cfg.off_road_yaw_deg

        events = self._collect_events(t, eye_closed, mouth_open, head_off)
        bs = obs.blendshapes
        return FrameFeatures(
            t=t,
            face_present=True,
            ear=ear,
            ear_right=ear_r,
            ear_left=ear_l,
            ear_rel=ear_rel,
            ear_threshold=threshold,
            eye_closed=eye_closed,
            closure_ongoing_s=self._closure.ongoing_duration(t),
            mar=mar,
            mouth_open=mouth_open,
            yawn_ongoing_s=self._yawn.ongoing_duration(t),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            pitch_dev=pitch_dev,
            yaw_dev=yaw_dev,
            head_off=head_off,
            looking_away=looking_away,
            bs_eye_blink=(bs.get("eyeBlinkLeft", NAN) + bs.get("eyeBlinkRight", NAN)) / 2.0,
            bs_jaw_open=bs.get("jawOpen", NAN),
            events=events,
        )

    def _collect_events(self, t: float, closed: bool, mouth_open: bool, head_off: bool) -> tuple[Event, ...]:
        out: list[Event] = []
        if (ev := self._closure.update(t, closed)) is not None:
            out.append(ev)
        if (ev := self._yawn.update(t, mouth_open)) is not None and ev.duration_s >= self.config.mouth.min_yawn_s:
            out.append(ev)
        if (ev := self._nod.update(t, head_off)) is not None and ev.duration_s >= self.config.head.min_nod_s:
            out.append(ev)
        return tuple(out)

    def flush(self, t: float) -> tuple[Event, ...]:
        """Cierra eventos abiertos al final de un video."""
        out: list[Event] = []
        if (ev := self._closure.flush(t)) is not None:
            out.append(ev)
        if (ev := self._yawn.flush(t)) is not None and ev.duration_s >= self.config.mouth.min_yawn_s:
            out.append(ev)
        if (ev := self._nod.flush(t)) is not None and ev.duration_s >= self.config.head.min_nod_s:
            out.append(ev)
        return tuple(out)
