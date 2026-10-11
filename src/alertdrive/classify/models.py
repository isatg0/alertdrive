"""Catálogo de clasificadores que se comparan en la Fase 2.

- ``rules``: algoritmo A, sistema de reglas con umbrales de la literatura.
- ``rf``: algoritmo B, Random Forest. Robusto a escalas distintas, captura
  interacciones no lineales entre indicadores y da importancia de variables.
- ``svm``: algoritmo B alternativo, SVM con kernel RBF. Requiere estandarizar.

Los modelos de ML se envuelven en un ``Pipeline`` (imputación → escalado →
clasificador). Así la imputación y el escalado se ajustan sólo con los datos
de entrenamiento de cada partición, y el mismo objeto se usa en la webcam.

Ambos usan ``class_weight="balanced"`` porque las clases pueden quedar
desbalanceadas (sobre todo en YawDD y NTHU-DDD).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from alertdrive.classify.rules import RuleBasedClassifier


@dataclass
class ModelSpec:
    name: str
    display_name: str
    build: Any  # callable(random_state) -> estimador sin entrenar
    param_grid: dict[str, list] = field(default_factory=dict)
    fast_grid: dict[str, list] = field(default_factory=dict)


def _rules(random_state: int) -> BaseEstimator:
    return RuleBasedClassifier()


def _rf(random_state: int) -> BaseEstimator:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state,
        )),
    ])


def _svm(random_state: int) -> BaseEstimator:
    # probability=False: el entrenamiento es ~5 veces más rápido. Para ROC se usa
    # decision_function, que ordena las muestras igual que una probabilidad.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=random_state)),
    ])


MODELS: dict[str, ModelSpec] = {
    "rules": ModelSpec("rules", "A · Reglas expertas", _rules),
    "rf": ModelSpec(
        "rf", "B · Random Forest", _rf,
        param_grid={
            "clf__max_depth": [None, 12],
            "clf__min_samples_leaf": [1, 5],
            "clf__max_features": ["sqrt", 0.5],
        },
        fast_grid={"clf__min_samples_leaf": [1, 5]},
    ),
    "svm": ModelSpec(
        "svm", "B' · SVM (kernel RBF)", _svm,
        param_grid={"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.03, 0.1]},
        fast_grid={"clf__C": [1, 10]},
    ),
}


def continuous_score(model: BaseEstimator, X) -> Any:
    """Puntaje continuo para ROC-AUC: probabilidad si existe, si no decision_function."""
    if isinstance(model, RuleBasedClassifier) or not hasattr(model, "predict_proba"):
        return model.decision_function(X)
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:  # SVC sin probability=True
        return model.decision_function(X)
