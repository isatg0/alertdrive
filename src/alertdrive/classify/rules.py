"""Algoritmo A: clasificador por reglas (sistema experto con escala de puntos).

Cada regla compara una característica de la ventana con un umbral tomado de la
literatura de detección de somnolencia y suma puntos si se cumple. La ventana
se clasifica como somnolienta si el puntaje total alcanza ``min_score``.

- Los umbrales y los puntos de cada regla son FIJOS (conocimiento experto); no
  se aprenden de los datos.
- Lo único que se ajusta con datos es ``min_score``, y sólo con el conjunto de
  entrenamiento de cada partición, para que la comparación con el algoritmo B
  sea justa y no haya fuga de información.
- Cada predicción puede explicarse con las reglas que se activaron
  (``explain``), útil para mostrar el motivo de la alerta en la plataforma.

Referencias para los umbrales (verificar la cita exacta antes del informe):
- PERCLOS ≥ 0.15: Wierwille et al. (1994); Dinges y Grace (1998), FHWA-MCRT-98-006.
- Duración de parpadeo elevada con la fatiga: Schleicher et al. (2008), Ergonomics 51(7).
- Microsueño como cierre ≥ 1 s y bostezos/cabeceos como signos tardíos de fatiga.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class Rule:
    feature: str
    threshold: float
    points: int
    description: str

    def fires(self, value: float) -> bool:
        return value is not None and not (isinstance(value, float) and math.isnan(value)) and value >= self.threshold


# Reglas por ventana de 15 s. Algunas son escalonadas (p. ej. PERCLOS suma 2
# puntos al pasar 0.15 y 1 más al pasar 0.30).
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("perclos", 0.15, 2, "PERCLOS ≥ 15 %: ojos cerrados una fracción anormal del tiempo"),
    Rule("perclos", 0.30, 1, "PERCLOS ≥ 30 %: cierre ocular severo"),
    Rule("microsleep_count", 1, 3, "Microsueño: cierre de ojos ≥ 1 s"),
    Rule("long_closure_count", 1, 1, "Cierre prolongado (> 0,5 s)"),
    Rule("blink_dur_mean_s", 0.30, 1, "Parpadeos lentos (duración media ≥ 0,3 s)"),
    Rule("yawn_count", 1, 1, "Bostezo detectado"),
    Rule("yawn_count", 2, 1, "Bostezos repetidos"),
    Rule("nod_count", 1, 2, "Cabeceo detectado"),
    Rule("head_off_fraction", 0.30, 1, "Cabeza fuera de posición ≥ 30 % del tiempo"),
)
MAX_SCORE = sum(r.points for r in DEFAULT_RULES)


class RuleBasedClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador por reglas con interfaz de scikit-learn.

    Args:
        min_score: puntaje mínimo para declarar somnolencia. Si es ``None``, se
            elige en ``fit`` el valor de ``candidate_scores`` que maximiza F1 en
            los datos de entrenamiento.
        candidate_scores: valores de ``min_score`` que se prueban en ``fit``.
    """

    def __init__(self, min_score: int | None = None, candidate_scores: tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
        self.min_score = min_score
        self.candidate_scores = candidate_scores

    # ------------------------------------------------------------ puntaje
    @staticmethod
    def score_frame(X: pd.DataFrame) -> np.ndarray:
        """Puntaje entero de cada ventana (0 a ``MAX_SCORE``)."""
        total = np.zeros(len(X), dtype=float)
        for rule in DEFAULT_RULES:
            values = pd.to_numeric(X[rule.feature], errors="coerce").to_numpy(dtype=float)
            fired = np.nan_to_num(values, nan=-np.inf) >= rule.threshold
            total += rule.points * fired
        return total

    @staticmethod
    def explain(features: dict[str, float]) -> list[str]:
        """Reglas activadas para una ventana (para mostrar el motivo de la alerta)."""
        return [r.description for r in DEFAULT_RULES if r.fires(features.get(r.feature, math.nan))]

    # ------------------------------------------------------------ sklearn
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleBasedClassifier:
        self.classes_ = np.array([0, 1])
        if self.min_score is not None:
            self.min_score_ = int(self.min_score)
            return self
        scores = self.score_frame(X)
        y = np.asarray(y)
        best = max(
            self.candidate_scores,
            key=lambda s: (f1_score(y, (scores >= s).astype(int), zero_division=0), -s),
        )
        self.min_score_ = int(best)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Puntaje normalizado a [0, 1]. No es una probabilidad calibrada."""
        return self.score_frame(X) / MAX_SCORE

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = self.decision_function(X)
        return np.column_stack([1 - s, s])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score_frame(X) >= self.min_score_).astype(int)

    @property
    def threshold_(self) -> float:
        """Umbral equivalente sobre ``decision_function``."""
        return self.min_score_ / MAX_SCORE
