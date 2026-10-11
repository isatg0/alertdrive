"""Evaluación comparativa de los algoritmos con validación cruzada POR SUJETO.

Protocolo (defendible en el informe):

1. Validación cruzada externa ``StratifiedGroupKFold`` agrupando por sujeto:
   ninguna persona aparece a la vez en entrenamiento y en prueba. Si se
   partiera por ventana, ventanas casi idénticas del mismo video caerían en
   ambos lados y las métricas saldrían infladas.
2. Dentro de cada partición de entrenamiento, búsqueda de hiperparámetros con
   ``GroupKFold`` interno (validación cruzada anidada). El conjunto de prueba
   nunca se usa para elegir nada, tampoco el umbral de las reglas.
3. Métricas por partición (media ± desviación) y sobre las predicciones
   fuera de partición agrupadas (matriz de confusión y curva ROC).
4. Diferencia de F1 entre algoritmos con intervalo de confianza por bootstrap
   de SUJETOS: las ventanas de un mismo sujeto están correlacionadas, así que
   remuestrear ventanas sueltas daría intervalos falsamente estrechos.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from alertdrive.classify.dataset import FeatureDataset, subsample_windows
from alertdrive.classify.models import MODELS, continuous_score

logger = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensibilidad
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score) if both else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@dataclass
class ModelEvaluation:
    name: str
    display_name: str
    fold_metrics: list[dict] = field(default_factory=list)
    best_params: list[dict] = field(default_factory=list)
    oof_pred: np.ndarray | None = None
    oof_score: np.ndarray | None = None
    latency_ms: list[float] = field(default_factory=list)
    fit_seconds: list[float] = field(default_factory=list)
    importance: pd.DataFrame | None = None

    def summary(self) -> dict[str, dict[str, float]]:
        df = pd.DataFrame(self.fold_metrics)
        return {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))} for m in METRIC_NAMES}


def _fit(name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, fast: bool,
         random_state: int, n_jobs: int, inner_splits: int = 3):
    spec = MODELS[name]
    estimator = spec.build(random_state)
    grid = spec.fast_grid if fast else spec.param_grid
    n_groups = len(np.unique(groups))
    if not grid or n_groups < inner_splits:
        estimator.fit(X, y)
        # Las reglas eligen su puntaje mínimo en fit; se registra para el informe.
        return estimator, ({"min_score": estimator.min_score_} if hasattr(estimator, "min_score_") else {})
    search = GridSearchCV(
        estimator, grid, scoring="f1", cv=GroupKFold(n_splits=inner_splits), n_jobs=n_jobs, refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def fit_model(name: str, data: FeatureDataset, subsample: int = 1, fast: bool = False,
              random_state: int = 42, n_jobs: int = -1):
    """Entrena un modelo con TODOS los datos (modelo final para desplegar)."""
    idx = subsample_windows(data, np.arange(len(data)), subsample)
    return _fit(name, data.X.iloc[idx], data.y[idx], data.groups[idx], fast, random_state, n_jobs)


def _outer_splitter(data: FeatureDataset, n_splits: int, random_state: int):
    n_groups = len(np.unique(data.groups))
    if n_groups < 2:
        raise ValueError("Se necesitan al menos 2 sujetos para evaluar por sujeto")
    if n_groups < n_splits:
        logger.warning("Sólo hay %d sujetos; se usan %d particiones en lugar de %d", n_groups, n_groups, n_splits)
        n_splits = n_groups
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # aviso de clases escasas en algún grupo
        return list(splitter.split(data.X, data.y, data.groups))


def cross_validate(
    data: FeatureDataset,
    model_names: list[str],
    n_splits: int = 5,
    subsample: int = 1,
    fast: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
    importance_for: tuple[str, ...] = ("rf",),
) -> tuple[dict[str, ModelEvaluation], np.ndarray]:
    """Evalúa cada modelo con las MISMAS particiones por sujeto.

    Devuelve las evaluaciones y un arreglo con la partición de prueba de cada fila.
    """
    splits = _outer_splitter(data, n_splits, random_state)
    fold_of = np.full(len(data), -1)
    results = {
        n: ModelEvaluation(n, MODELS[n].display_name,
                           oof_pred=np.zeros(len(data), dtype=int), oof_score=np.zeros(len(data)))
        for n in model_names
    }
    importances: dict[str, list[pd.Series]] = {n: [] for n in model_names}

    for k, (train_idx, test_idx) in enumerate(splits):
        fold_of[test_idx] = k
        train_idx = subsample_windows(data, train_idx, subsample)
        X_tr, y_tr, g_tr = data.X.iloc[train_idx], data.y[train_idx], data.groups[train_idx]
        X_te, y_te = data.X.iloc[test_idx], data.y[test_idx]
        logger.info(
            "Partición %d/%d: %d sujetos de prueba, %d ventanas de entrenamiento, %d de prueba",
            k + 1, len(splits), len(np.unique(data.groups[test_idx])), len(train_idx), len(test_idx),
        )
        for name in model_names:
            res = results[name]
            t0 = time.perf_counter()
            model, params = _fit(name, X_tr, y_tr, g_tr, fast, random_state, n_jobs)
            res.fit_seconds.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            pred = model.predict(X_te)
            res.latency_ms.append(1000 * (time.perf_counter() - t0) / max(len(test_idx), 1))
            score = continuous_score(model, X_te)

            res.oof_pred[test_idx], res.oof_score[test_idx] = pred, score
            res.best_params.append(params)
            res.fold_metrics.append({"fold": k, **classification_metrics(y_te, pred, score)})
            logger.info("  %-6s F1=%.3f recall=%.3f AUC=%.3f (%.1f s)", name,
                        res.fold_metrics[-1]["f1"], res.fold_metrics[-1]["recall"],
                        res.fold_metrics[-1]["roc_auc"], res.fit_seconds[-1])

            if name in importance_for and len(np.unique(y_te)) == 2:
                pi = permutation_importance(model, X_te, y_te, scoring="f1", n_repeats=5,
                                            random_state=random_state, n_jobs=n_jobs)
                importances[name].append(pd.Series(pi.importances_mean, index=data.X.columns))

    for name, series in importances.items():
        if series:
            imp = pd.concat(series, axis=1)
            results[name].importance = pd.DataFrame(
                {"mean": imp.mean(axis=1), "std": imp.std(axis=1, ddof=1)}
            ).sort_values("mean", ascending=False)
    return results, fold_of


def pooled_metrics(data: FeatureDataset, res: ModelEvaluation) -> dict[str, float]:
    """Métricas sobre todas las predicciones fuera de partición juntas."""
    return classification_metrics(data.y, res.oof_pred, res.oof_score)


def video_level_metrics(data: FeatureDataset, res: ModelEvaluation, vote: float = 0.5) -> dict[str, float]:
    """Métricas por video: el video es positivo si al menos ``vote`` de sus ventanas lo son.

    Las etiquetas de UTA-RLDD y YawDD son por video, así que esta vista es la que
    corresponde exactamente a la verdad del dataset.
    """
    df = pd.DataFrame({"video": data.meta["video_path"].to_numpy(), "y": data.y,
                       "pred": res.oof_pred, "score": res.oof_score})
    agg = df.groupby("video").agg(y=("y", "mean"), frac=("pred", "mean"), score=("score", "mean"))
    y_true = (agg["y"] >= 0.5).astype(int).to_numpy()
    y_pred = (agg["frac"] >= vote).astype(int).to_numpy()
    out = classification_metrics(y_true, y_pred, agg["score"].to_numpy())
    out["n_videos"] = len(agg)
    return out


def _subject_counts(data: FeatureDataset, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"g": data.groups, "y": data.y, "p": pred})
    return df.assign(tp=(df.y == 1) & (df.p == 1), fp=(df.y == 0) & (df.p == 1),
                     fn=(df.y == 1) & (df.p == 0)).groupby("g")[["tp", "fp", "fn"]].sum()


def bootstrap_f1_difference(
    data: FeatureDataset, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """IC 95 % de F1(B) − F1(A) remuestreando sujetos con reemplazo."""
    ca = _subject_counts(data, pred_a)
    cb = _subject_counts(data, pred_b).reindex(ca.index)
    A, B = ca.to_numpy(dtype=float), cb.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(n_boot, len(A)))

    def f1(counts: np.ndarray) -> np.ndarray:
        tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)

    diffs = f1(B[idx].sum(axis=1)) - f1(A[idx].sum(axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta_f1": float(f1(B.sum(axis=0)) - f1(A.sum(axis=0))),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "p_b_not_better": float(np.mean(diffs <= 0)),
        "n_subjects": len(A), "n_boot": n_boot,
    }
