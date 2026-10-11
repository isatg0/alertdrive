"""Modelos entrenados en la Fase 2 y sus nombres para la interfaz y el histórico."""
from __future__ import annotations

from pathlib import Path

MODEL_FILES: dict[str, str] = {
    "Mejor modelo de la Fase 2": "models/drowsiness_best.joblib",
    "Random Forest": "models/drowsiness_rf.joblib",
    "SVM (kernel RBF)": "models/drowsiness_svm.joblib",
    "Reglas expertas": "models/drowsiness_rules.joblib",
}


def model_label(path: str | Path) -> str:
    """Nombre legible de un archivo de modelo (el mismo en la app y en la consola)."""
    name = Path(path).name
    for label, file in MODEL_FILES.items():
        if Path(file).name == name:
            return label
    return Path(path).stem


def available_models() -> dict[str, str]:
    return {label: path for label, path in MODEL_FILES.items() if Path(path).exists()}
