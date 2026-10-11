"""Uso del modelo entrenado sobre una ventana de características (webcam o video).

El archivo ``.joblib`` guarda, junto al modelo, la lista y el orden de las
columnas, el umbral y los datos de entrenamiento. Así la plataforma no puede
alimentar el modelo con columnas en otro orden sin que se note.

Uso (Fase 3)::

    predictor = DrowsinessPredictor.load("models/drowsiness_rf.joblib")
    window = aggregator.compute(t_now)          # diccionario de la Fase 1
    result = predictor.predict(window)
    if result.is_drowsy: ...                     # disparar alerta
    result.reasons                               # motivos legibles para la interfaz
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from alertdrive.classify.models import continuous_score
from alertdrive.classify.rules import RuleBasedClassifier
from alertdrive.features.window import FEATURE_COLUMNS

BUNDLE_VERSION = 1


@dataclass(frozen=True)
class Prediction:
    is_drowsy: bool
    score: float  # probabilidad (RF), distancia al hiperplano (SVM) o puntaje normalizado (reglas)
    threshold: float
    valid: bool  # False si la ventana tuvo poco rostro visible
    reasons: list[str] = field(default_factory=list)


def save_bundle(path: str | Path, model: Any, name: str, params: dict, trained_on: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, RuleBasedClassifier):
        threshold, score_kind = model.threshold_, "rule_score"
    elif hasattr(model, "predict_proba"):
        threshold, score_kind = 0.5, "probability"
    else:
        threshold, score_kind = 0.0, "decision_function"
    joblib.dump({
        "bundle_version": BUNDLE_VERSION,
        "name": name,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": float(threshold),
        "score_kind": score_kind,
        "params": params,
        "trained_on": trained_on,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, path)
    return path


class DrowsinessPredictor:
    def __init__(self, bundle: dict[str, Any]) -> None:
        if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Las columnas del modelo no coinciden con las de la Fase 1 actual. "
                "Vuelve a extraer características y a entrenar."
            )
        self.bundle = bundle
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.name = bundle["name"]

    @classmethod
    def load(cls, path: str | Path) -> DrowsinessPredictor:
        return cls(joblib.load(path))

    def predict(self, window: dict[str, float]) -> Prediction:
        valid = float(window.get("valid", 1.0)) >= 1.0
        row = {c: window.get(c, math.nan) for c in FEATURE_COLUMNS}
        X = pd.DataFrame([row], columns=list(FEATURE_COLUMNS)).astype(float)
        score = float(np.asarray(continuous_score(self.model, X))[0])
        is_drowsy = bool(self.model.predict(X)[0]) and valid
        return Prediction(
            is_drowsy=is_drowsy,
            score=score,
            threshold=self.threshold,
            valid=valid,
            reasons=RuleBasedClassifier.explain(row) if valid else ["Rostro no visible la mayor parte de la ventana"],
        )
