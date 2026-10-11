"""Utilidades temporales: línea base adaptativa y detección de eventos.

Un "evento" es un intervalo continuo en el que una condición se cumple
(ojos cerrados, boca muy abierta, cabeza desviada). A partir de los eventos se
obtienen parpadeos, cierres prolongados, bostezos y cabeceos.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    kind: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class AdaptiveBaseline:
    """Percentil móvil de una señal en una ventana de tiempo.

    Ejemplo: el percentil 90 del EAR en los últimos 30 s aproxima el EAR del
    conductor con los ojos abiertos, sin calibración manual.
    """

    def __init__(self, window_s: float, percentile: float, min_samples: int) -> None:
        self.window_s = window_s
        self.percentile = percentile
        self.min_samples = min_samples
        self._buf: deque[tuple[float, float]] = deque()
        self._value: float | None = None

    def update(self, t: float, value: float) -> None:
        self._buf.append((t, value))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        if len(self._buf) >= self.min_samples:
            self._value = float(np.percentile([v for _, v in self._buf], self.percentile))

    @property
    def value(self) -> float | None:
        return self._value


class EventDetector:
    """Convierte una señal booleana por frame en eventos con inicio y fin.

    ``merge_gap_s`` une activaciones separadas por huecos cortos (p. ej. un
    frame en que el detector falla a mitad de un bostezo).
    """

    def __init__(self, kind: str, merge_gap_s: float = 0.0) -> None:
        self.kind = kind
        self.merge_gap_s = merge_gap_s
        self._start: float | None = None
        self._first_inactive: float | None = None

    @property
    def active(self) -> bool:
        return self._start is not None

    def update(self, t: float, is_active: bool) -> Event | None:
        if is_active:
            if self._start is None:
                self._start = t
            self._first_inactive = None
            return None
        if self._start is None:
            return None
        if self._first_inactive is None:
            self._first_inactive = t
        if t - self._first_inactive >= self.merge_gap_s:
            return self._close()
        return None

    def ongoing_duration(self, t: float) -> float:
        """Duración del evento en curso (0 si no hay)."""
        if self._start is None:
            return 0.0
        end = self._first_inactive if self._first_inactive is not None else t
        return end - self._start

    def flush(self, t: float) -> Event | None:
        """Cierra el evento en curso (al terminar un video)."""
        if self._start is None:
            return None
        if self._first_inactive is None:
            self._first_inactive = t
        return self._close()

    def _close(self) -> Event:
        assert self._start is not None and self._first_inactive is not None
        event = Event(self.kind, self._start, self._first_inactive)
        self._start = None
        self._first_inactive = None
        return event
