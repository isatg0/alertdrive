"""Lógica de alertas: convierte predicciones sueltas en alertas estables.

El modelo predice cada segundo sobre una ventana de 15 s. Dos predicciones
seguidas comparten 14 s de datos, así que una sola predicción positiva no
basta para sonar: se exige que se repita ``persistence`` veces. Una vez en
alarma, el sonido se repite como máximo cada ``cooldown_s`` segundos.
"""
from __future__ import annotations

import io
import math
import struct
import sys
import threading
import wave
from dataclasses import dataclass, field

# Niveles posibles, de menor a mayor gravedad.
LEVELS = ("calibrando", "sin_rostro", "ok", "atencion", "alarma")


@dataclass(frozen=True)
class AlertDecision:
    level: str                       # uno de LEVELS
    new_alarm: bool                  # True: sonar y registrar el evento ahora
    streak: int = 0                  # predicciones somnolientas consecutivas
    reasons: tuple[str, ...] = field(default_factory=tuple)


class AlertManager:
    def __init__(self, persistence: int = 3, cooldown_s: float = 8.0) -> None:
        if persistence < 1:
            raise ValueError("persistence debe ser >= 1")
        self.persistence = persistence
        self.cooldown_s = cooldown_s
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self._last_alarm_t = -math.inf

    def update(
        self,
        t: float,
        ready: bool,
        valid: bool,
        is_drowsy: bool,
        reasons: list[str] | tuple[str, ...] | None = None,
    ) -> AlertDecision:
        """Actualiza el estado con una nueva predicción.

        Args:
            t: tiempo de la predicción en segundos.
            ready: si ya hay una ventana completa (fin de la calibración).
            valid: si la ventana tuvo suficiente rostro visible.
            is_drowsy: salida del clasificador.
            reasons: reglas activadas que explican la predicción.
        """
        reasons = tuple(reasons or ())
        if not ready:
            return AlertDecision("calibrando", False)
        if not valid:
            self.streak = 0
            return AlertDecision("sin_rostro", False)

        self.streak = self.streak + 1 if is_drowsy else 0
        if self.streak >= self.persistence:
            new_alarm = (t - self._last_alarm_t) >= self.cooldown_s
            if new_alarm:
                self._last_alarm_t = t
            return AlertDecision("alarma", new_alarm, self.streak, reasons)
        if self.streak >= 1:
            return AlertDecision("atencion", False, self.streak, reasons)
        return AlertDecision("ok", False, 0, ())


def make_beep_wav(
    freq: int = 1000, dur_s: float = 0.35, repeats: int = 3, rate: int = 22050, volume: float = 0.6
) -> bytes:
    """Genera en memoria un WAV con ``repeats`` pitidos (para el navegador)."""
    beep = [int(volume * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(int(rate * dur_s))]
    silence = [0] * int(rate * 0.15)
    samples = (beep + silence) * repeats
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def play_system_beep(freq: int = 1000, dur_ms: int = 350, repeats: int = 3) -> bool:
    """Suena en los parlantes del computador que corre la app (sólo Windows).

    Corre en un hilo para no congelar el video. Devuelve False si no hay
    ``winsound`` (Linux/macOS); en ese caso la app usa el audio del navegador.
    """
    if sys.platform != "win32":
        return False
    import winsound

    def _run() -> None:
        for _ in range(repeats):
            winsound.Beep(freq, dur_ms)

    threading.Thread(target=_run, daemon=True).start()
    return True
