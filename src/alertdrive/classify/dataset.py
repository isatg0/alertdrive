"""Carga de los CSV de características (Fase 1) y construcción del objetivo binario.

El problema se plantea como clasificación binaria por ventana de 15 s:
``1 = signos de somnolencia/fatiga`` y ``0 = conductor alerta``.

Mapeo de etiquetas por dataset (declararlo en el informe):

- UTA-RLDD: ``drowsy`` → 1, ``alert`` → 0. ``low_vigilant`` es un estado
  intermedio; por defecto se descarta porque mezclarlo con cualquiera de las
  dos clases vuelve ambigua la frontera. Se puede cambiar con ``low_vigilant``.
- YawDD: ``yawning`` → 1, ``normal`` y ``talking`` → 0. Ojo: YawDD mide bostezo,
  no somnolencia; sirve como indicador de fatiga, no como verdad de sueño.
- Datasets con anotación por frame (p. ej. NTHU-DDD): si la fila trae
  ``label_fraction``, la ventana es positiva cuando al menos la mitad de sus
  frames está anotada como somnolienta, sin importar la etiqueta del video.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alertdrive.features.window import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

POSITIVE_LABELS = frozenset({"drowsy", "yawning", "sleepy", "1"})
NEGATIVE_LABELS = frozenset({"alert", "normal", "talking", "nonsleepy", "0"})
INTERMEDIATE_LABELS = frozenset({"low_vigilant"})
LOW_VIGILANT_POLICIES = ("drop", "positive", "negative")
FRAME_LABEL_THRESHOLD = 0.5


@dataclass
class FeatureDataset:
    """Datos listos para entrenar: X (características), y (0/1), groups (sujeto)."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame  # video_path, subject_id, label, dataset, t_end

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        n_pos = int(self.y.sum())
        n_subj = len(np.unique(self.groups))
        n_vid = self.meta["video_path"].nunique()
        return (
            f"{len(self)} ventanas, {n_vid} videos, {n_subj} sujetos | "
            f"positivas: {n_pos} ({n_pos / max(len(self), 1):.1%}), negativas: {len(self) - n_pos}"
        )


def map_label(label: str, low_vigilant: str = "drop") -> float:
    """Devuelve 1.0, 0.0 o NaN (descartar) para una etiqueta de video."""
    if low_vigilant not in LOW_VIGILANT_POLICIES:
        raise ValueError(f"low_vigilant debe ser uno de {LOW_VIGILANT_POLICIES}")
    key = str(label).strip().lower()
    if key in POSITIVE_LABELS:
        return 1.0
    if key in NEGATIVE_LABELS:
        return 0.0
    if key in INTERMEDIATE_LABELS:
        return {"drop": np.nan, "positive": 1.0, "negative": 0.0}[low_vigilant]
    return np.nan


def build_target(df: pd.DataFrame, low_vigilant: str = "drop") -> pd.Series:
    """Objetivo binario por fila; NaN significa que la fila se descarta."""
    y = df["label"].map(lambda s: map_label(s, low_vigilant)).astype(float)
    if "label_fraction" in df.columns:
        frac = pd.to_numeric(df["label_fraction"], errors="coerce")
        has_frames = frac.notna()
        y = y.where(~has_frames, (frac >= FRAME_LABEL_THRESHOLD).astype(float))
    return y


def load_features(
    paths: Iterable[str | Path],
    low_vigilant: str = "drop",
    only_valid: bool = True,
) -> FeatureDataset:
    """Lee uno o varios CSV de ``build_features`` y los deja listos para los modelos."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No se entregó ningún CSV de características")
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in ("subject_id", "label", "video_path", *FEATURE_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV de características: {missing}")

    n0 = len(df)
    if only_valid and "valid" in df.columns:
        df = df[df["valid"].astype(float) >= 1.0]
    n_invalid = n0 - len(df)

    y = build_target(df, low_vigilant)
    unknown = sorted(df.loc[y.isna() & ~df["label"].str.lower().isin(INTERMEDIATE_LABELS), "label"].unique())
    if unknown:
        logger.warning("Etiquetas sin mapeo (se descartan): %s", unknown)
    keep = y.notna()
    df, y = df[keep].reset_index(drop=True), y[keep].to_numpy(dtype=int)

    X = df[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    meta_cols = [c for c in ("video_path", "subject_id", "label", "dataset", "t_end") if c in df.columns]
    data = FeatureDataset(X=X, y=y, groups=df["subject_id"].astype(str).to_numpy(), meta=df[meta_cols].copy())

    logger.info(
        "Cargadas %d filas; descartadas %d inválidas (poco rostro) y %d sin etiqueta binaria",
        n0, n_invalid, int((~keep).sum()),
    )
    if len(np.unique(data.y)) < 2:
        raise ValueError("El conjunto quedó con una sola clase; revisa las etiquetas y la opción low_vigilant")
    return data


def subsample_windows(data: FeatureDataset, idx: np.ndarray, every: int) -> np.ndarray:
    """Conserva una de cada ``every`` ventanas por video dentro de ``idx``.

    Las ventanas se solapan (15 s con paso de 5 s), así que las consecutivas son
    casi idénticas. Submuestrear el entrenamiento reduce el costo sin perder
    información útil. Nunca se aplica al conjunto de prueba.
    """
    if every <= 1:
        return idx
    videos = data.meta["video_path"].to_numpy()[idx]
    order = pd.Series(np.arange(len(idx))).groupby(videos).cumcount().to_numpy()
    return idx[order % every == 0]
