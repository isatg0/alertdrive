"""Reporte de la evaluación: JSON, tablas en Markdown y gráficas PNG.

Las tablas de ``resumen.md`` están pensadas para copiarse al informe.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from alertdrive.classify.dataset import FeatureDataset  # noqa: E402
from alertdrive.classify.evaluate import METRIC_NAMES, ModelEvaluation  # noqa: E402

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
LINESTYLES = ["-", "--", ":", "-."]
METRIC_LABELS = {
    "accuracy": "Exactitud", "balanced_accuracy": "Exactitud balanceada", "precision": "Precisión",
    "recall": "Recall (sensibilidad)", "specificity": "Especificidad", "f1": "F1", "roc_auc": "ROC-AUC",
}


def to_markdown(df: pd.DataFrame) -> str:
    """Tabla Markdown sin depender de ``tabulate``."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_roc(data: FeatureDataset, results: dict[str, ModelEvaluation], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data.y, res.oof_score)
        auc = res.summary()["roc_auc"]
        ax.plot(fpr, tpr, color=COLORS[i % 4], linestyle=LINESTYLES[i % 4], linewidth=2,
                label=f"{res.display_name} (AUC {auc['mean']:.2f} ± {auc['std']:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title("Curvas ROC, validación cruzada por sujeto", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(data: FeatureDataset, results: dict[str, ModelEvaluation], out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results.values()):
        cm = pd.crosstab(pd.Series(data.y, name="real"), pd.Series(res.oof_pred, name="pred")) \
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}\n({norm[r, c]:.0%})", ha="center", va="center",
                    color="white" if norm[r, c] > 0.5 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["Alerta", "Somnoliento"])
        ax.set_yticks([0, 1], ["Alerta", "Somnoliento"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(res.display_name, fontsize=9, fontweight="bold")
    fig.suptitle("Matrices de confusión (ventanas fuera de partición)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(res: ModelEvaluation, out: Path, top: int = 15) -> None:
    imp = res.importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"].fillna(0), color=COLORS[0], capsize=2)
    ax.set_xlabel("Caída media de F1 al permutar la característica")
    ax.set_title(f"Importancia por permutación · {res.display_name}", fontweight="bold", fontsize=10)
    ax.axvline(0, color="grey", linewidth=0.8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def metrics_table(results: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for res in results.values():
        s = res.summary()
        rows.append({"Algoritmo": res.display_name,
                     **{METRIC_LABELS[m]: f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f}" for m in METRIC_NAMES},
                     "Latencia (ms/ventana)": f"{np.mean(res.latency_ms):.3f}"})
    return pd.DataFrame(rows)


def _fmt_row(d: dict, keys) -> dict:
    return {METRIC_LABELS.get(k, k): (f"{d[k]:.3f}" if isinstance(d[k], float) else d[k]) for k in keys}


def write_report(
    out_dir: Path,
    data: FeatureDataset,
    results: dict[str, ModelEvaluation],
    pooled: dict[str, dict],
    video: dict[str, dict],
    comparisons: dict[str, dict],
    settings: dict,
    best_model: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(data, results, out_dir / "roc.png")
    plot_confusion(data, results, out_dir / "confusion.png")
    for name, res in results.items():
        if res.importance is not None:
            plot_importance(res, out_dir / f"importancia_{name}.png")
            res.importance.to_csv(out_dir / f"importancia_{name}.csv")

    per_fold = pd.concat([pd.DataFrame(r.fold_metrics).assign(model=n) for n, r in results.items()])
    per_fold.to_csv(out_dir / "metricas_por_particion.csv", index=False)

    payload = {
        "settings": settings,
        "dataset": data.summary(),
        "best_model": best_model,
        "models": {
            n: {"display_name": r.display_name, "cv": r.summary(), "pooled": pooled[n], "video_level": video[n],
                "best_params_per_fold": r.best_params, "latency_ms": float(np.mean(r.latency_ms)),
                "fit_seconds_mean": float(np.mean(r.fit_seconds))}
            for n, r in results.items()
        },
        "comparisons_vs_rules": comparisons,
    }
    (out_dir / "metricas.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    lines = [
        "# Fase 2 · Comparación de algoritmos",
        "",
        f"**Datos:** {data.summary()}",
        "",
        f"**Protocolo:** validación cruzada de {settings['folds']} particiones agrupadas por sujeto "
        f"(StratifiedGroupKFold), búsqueda de hiperparámetros anidada con GroupKFold, "
        f"low_vigilant = `{settings['low_vigilant']}`, submuestreo de entrenamiento = {settings['subsample']}.",
        "",
        "## Métricas por ventana (media ± desviación entre particiones)",
        "",
        to_markdown(metrics_table(results)),
        "",
        "## Métricas por video (voto de mayoría de sus ventanas)",
        "",
        to_markdown(pd.DataFrame([{"Algoritmo": results[n].display_name, **_fmt_row(video[n], keys), "Videos": video[n]["n_videos"]}
                      for n in results])),
        "",
    ]
    if comparisons:
        lines += ["## Diferencia de F1 frente al algoritmo A (bootstrap por sujeto)", ""]
        lines.append(to_markdown(pd.DataFrame([
            {"Algoritmo": results[n].display_name, "ΔF1": f"{c['delta_f1']:+.3f}",
             "IC 95 %": f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]",
             "Sujetos": c["n_subjects"]} for n, c in comparisons.items()
        ])))
        lines += ["", "Si el intervalo no contiene 0, la diferencia es significativa al 95 %.", ""]
    lines += [f"**Modelo seleccionado para la plataforma:** {results[best_model].display_name} "
              "(mayor F1 medio por sujeto).", ""]
    (out_dir / "resumen.md").write_text("\n".join(lines), encoding="utf-8")
