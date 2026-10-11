"""Analiza las trazas de ``platform_eval`` y escribe el reporte de la Fase 5.

Reproduce la lógica de alertas de la plataforma (``AlertManager``) sobre las
predicciones guardadas, para varios valores de persistencia, sin volver a
procesar los videos. Mide lo que importa en uso real:

- En videos somnolientos: si hubo al menos una alarma (detección) y cuánto
  tardó la primera desde que terminó la calibración (latencia).
- En videos alerta: falsas alarmas por hora y videos sin ninguna alarma.
- Por segundo: exactitud, precisión, recall, especificidad, F1 y ROC-AUC.
- Rendimiento: milisegundos por frame y frames por segundo alcanzables.
- Umbral de puntaje: cómo cambian las alarmas si se exige un puntaje mayor
  para contar un segundo como fatiga, y qué umbral se recomienda por modelo.

    python -m alertdrive.evaluation.analyze --out reports/fase5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alertdrive.web.alerts import AlertManager

logger = logging.getLogger("alertdrive.evaluation")
WINDOW_S = 15.0
SAMPLE_FPS = 5.0


# ----------------------------------------------------------------- cálculo
def video_metrics(trace: pd.DataFrame, persistence: int, cooldown_s: float,
                  window_s: float = WINDOW_S) -> dict:
    """Aplica la lógica de alertas a la traza de un video y un modelo."""
    trace = trace.sort_values("t")
    am = AlertManager(persistence=persistence, cooldown_s=cooldown_s)
    levels, alarm_times = [], []
    for r in trace.itertuples(index=False):
        d = am.update(r.t, bool(r.ready), bool(r.valid), bool(r.is_drowsy))
        levels.append(d.level)
        if d.new_alarm:
            alarm_times.append(r.t)
    levels = np.asarray(levels)
    monitored = levels != "calibrando"
    n_mon = int(monitored.sum())
    hours = n_mon / 3600.0
    first = alarm_times[0] - window_s if alarm_times else math.nan
    return {
        "monitored_s": n_mon,
        "n_alarms": len(alarm_times),
        "alarms_per_hour": len(alarm_times) / hours if hours else math.nan,
        "first_alarm_s": max(first, 0.0) if alarm_times else math.nan,
        "frac_alarm": float((levels[monitored] == "alarma").mean()) if n_mon else math.nan,
        "frac_no_face": float((levels[monitored] == "sin_rostro").mean()) if n_mon else math.nan,
    }


def per_video_table(traces: pd.DataFrame, persistences: list[int], cooldown_s: float) -> pd.DataFrame:
    rows = []
    for (key, model), g in traces.groupby(["video_key", "model"], sort=False):
        for p in persistences:
            m = video_metrics(g, p, cooldown_s)
            rows.append({"video_key": key, "subject_id": g["subject_id"].iloc[0], "y": int(g["y"].iloc[0]),
                         "model": model, "persistence": p, **m})
    return pd.DataFrame(rows)


def summarize_platform(videos: pd.DataFrame) -> pd.DataFrame:
    """Resumen por modelo y persistencia (nivel video)."""
    rows = []
    for (model, p), g in videos.groupby(["model", "persistence"]):
        dz, al = g[g["y"] == 1], g[g["y"] == 0]
        detected = dz["n_alarms"] > 0
        al_hours = al["monitored_s"].sum() / 3600
        sens = detected.mean() if len(dz) else math.nan
        spec = (al["n_alarms"] == 0).mean() if len(al) else math.nan
        rows.append({
            "model": model, "persistence": p,
            "n_drowsy": len(dz), "detection_rate": sens,
            "latency_median_s": dz.loc[detected, "first_alarm_s"].median() if detected.any() else math.nan,
            "drowsy_time_in_alarm": dz["frac_alarm"].mean() if len(dz) else math.nan,
            "n_alert": len(al), "alert_videos_without_alarm": spec,
            "false_alarms_per_hour": al["n_alarms"].sum() / al_hours if al_hours else math.nan,
            "balanced_accuracy": np.nanmean([sens, spec]) if len(dz) and len(al) else math.nan,
        })
    return pd.DataFrame(rows)


def per_second_metrics(traces: pd.DataFrame) -> pd.DataFrame:
    """Predicción cruda del modelo cada segundo (sin persistencia), sólo segundos listos y válidos."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                 roc_auc_score)

    rows = []
    for model, g in traces[traces["ready"] & traces["valid"]].groupby("model"):
        y, yhat = g["y"].astype(int), g["is_drowsy"].astype(int)
        tn = int(((y == 0) & (yhat == 0)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        auc = roc_auc_score(y, g["score"]) if y.nunique() == 2 else math.nan
        rows.append({
            "model": model, "seconds": len(g),
            "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else math.nan,
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": auc,
        })
    return pd.DataFrame(rows)


def performance_table(perf: list[dict], sample_fps: float = SAMPLE_FPS) -> pd.DataFrame:
    if not perf:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "frames": p["frames"], "detect": p["detect_ms"]["mean"], "features": p["features_ms"]["mean"],
        "frame": p["frame_ms"]["mean"], "frame_p95": p["frame_ms"]["p95"], "model": p["model_ms"]["mean"],
    } for p in perf])
    w = df["frames"] / df["frames"].sum()
    frame_ms = float((df["frame"] * w).sum())
    return pd.DataFrame([{
        "videos": len(df), "frames": int(df["frames"].sum()),
        "detect_ms": float((df["detect"] * w).sum()),
        "features_ms": float((df["features"] * w).sum()),
        "frame_ms": frame_ms,
        "frame_ms_p95_max": float(df["frame_p95"].max()),
        "model_ms": float(df["model"].mean()),
        "max_fps": 1000.0 / frame_ms if frame_ms else math.nan,
        "realtime_margin": (1000.0 / frame_ms) / sample_fps if frame_ms else math.nan,
    }])


def threshold_sweep(traces: pd.DataFrame, persistence: int, cooldown_s: float,
                    quantiles: np.ndarray | None = None) -> pd.DataFrame:
    """Recalcula las alertas exigiendo ``score >= umbral`` en lugar de la decisión del modelo.

    Los umbrales candidatos son cuantiles del puntaje de cada modelo, así sirve
    igual para probabilidades (Random Forest, SVM) y para puntos (reglas).
    No vuelve a procesar videos: usa la columna ``score`` de las trazas.
    """
    quantiles = np.linspace(0.05, 0.95, 19) if quantiles is None else quantiles
    rows = []
    for model, g in traces.groupby("model"):
        scores = g.loc[g["ready"] & g["valid"], "score"].dropna()
        if scores.empty:
            continue
        for thr in sorted(set(np.round(np.quantile(scores, quantiles), 4))):
            gg = g.assign(is_drowsy=g["valid"] & (g["score"] >= thr))
            summ = summarize_platform(per_video_table(gg, [persistence], cooldown_s)).iloc[0].to_dict()
            sec = gg[gg["ready"] & gg["valid"]]
            pos, neg = sec[sec["y"] == 1], sec[sec["y"] == 0]
            summ.update({
                "threshold": float(thr),
                "recall_s": float(pos["is_drowsy"].mean()) if len(pos) else math.nan,
                "specificity_s": float((~neg["is_drowsy"]).mean()) if len(neg) else math.nan,
            })
            rows.append(summ)
    return pd.DataFrame(rows)


def recommend_thresholds(sweep: pd.DataFrame, min_detection: float = 0.9) -> pd.DataFrame:
    """Por modelo: el umbral con menos falsas alarmas que aún detecta ``min_detection`` de los videos."""
    out = []
    for model, g in sweep.groupby("model"):
        ok = g[g["detection_rate"] >= min_detection - 1e-9]
        if ok.empty:  # ningún umbral cumple: el de mejor exactitud balanceada
            best = g.sort_values(["balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]
        else:
            best = ok.sort_values(["false_alarms_per_hour", "threshold"]).iloc[0]
        out.append(best)
    return pd.DataFrame(out).reset_index(drop=True)


# ----------------------------------------------------------------- reporte
def _pct(v: float) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.0%}"


def _num(v: float, nd: int = 1) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for model, g in summary.groupby("model"):
        g = g.sort_values("persistence")
        axes[0].plot(g["persistence"], g["detection_rate"], marker="o", label=model)
        axes[1].plot(g["persistence"], g["latency_median_s"], marker="o", label=model)
        axes[2].plot(g["persistence"], g["false_alarms_per_hour"], marker="o", label=model)
    titles = ("Videos somnolientos detectados", "Latencia mediana de la primera alarma",
              "Falsas alarmas por hora (videos alerta)")
    ylabels = ("Proporción", "Segundos", "Alarmas por hora")
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Persistencia (segundos seguidos con fatiga)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax.set_xticks(sorted(summary["persistence"].unique()))
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_threshold_figure(sweep: pd.DataFrame, rec: pd.DataFrame, baseline: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (model, g) in enumerate(sweep.groupby("model")):
        color = f"C{i}"
        g = g.sort_values("false_alarms_per_hour")
        axes[0].plot(g["false_alarms_per_hour"], g["detection_rate"], marker="o", ms=3, color=color, label=model)
        axes[1].plot(g["false_alarms_per_hour"], g["latency_median_s"], marker="o", ms=3, color=color, label=model)
        r = rec[rec["model"] == model]
        b = baseline[baseline["model"] == model]
        for ax, col in ((axes[0], "detection_rate"), (axes[1], "latency_median_s")):
            if not r.empty:
                ax.scatter(r["false_alarms_per_hour"], r[col], s=120, facecolors="none", edgecolors=color, lw=2)
            if not b.empty:
                ax.scatter(b["false_alarms_per_hour"], b[col], marker="x", s=70, color=color)
    axes[0].set_title("Detección vs. falsas alarmas", fontsize=11)
    axes[0].set_ylabel("Videos somnolientos detectados")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("Latencia vs. falsas alarmas", fontsize=11)
    axes[1].set_ylabel("Latencia mediana de la primera alarma (s)")
    for ax in axes:
        ax.set_xlabel("Falsas alarmas por hora (videos alerta)")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=9, title="Círculo: recomendado · X: criterio del modelo", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out: Path, traces: pd.DataFrame, summary: pd.DataFrame, seconds: pd.DataFrame,
                 perf: pd.DataFrame, cooldown_s: float, default_p: int,
                 rec: pd.DataFrame | None = None, min_detection: float = 0.9) -> str:
    vids = traces.drop_duplicates("video_key")
    lines = [
        "# Fase 5 · Evaluación de la plataforma completa",
        "",
        f"**Datos:** {len(vids)} videos, {vids['subject_id'].nunique()} sujetos "
        f"(alerta: {int((vids['y'] == 0).sum())}, somnoliento: {int((vids['y'] == 1).sum())}), "
        f"{traces.groupby('video_key')['t'].max().sum() / 3600:.1f} h analizadas por modelo.",
        "",
        f"**Protocolo:** muestreo a {SAMPLE_FPS:.0f} fps, predicción cada segundo sobre la ventana de "
        f"{WINDOW_S:.0f} s, los primeros {WINDOW_S:.0f} s de cada video son calibración y no cuentan. "
        f"Enfriamiento entre alarmas: {cooldown_s:.0f} s.",
        "",
        "## Alertas por video",
        "",
        f"Con la persistencia por defecto de la plataforma ({default_p} s). Un video somnoliento se considera "
        "detectado si tiene al menos una alarma; la latencia se mide desde el fin de la calibración.",
        "",
        "| Modelo | Videos somnolientos detectados | Latencia mediana (s) | Tiempo en alarma (somnolientos) "
        "| Videos alerta sin alarmas | Falsas alarmas por hora | Exactitud balanceada |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary[summary["persistence"] == default_p].itertuples():
        lines.append(f"| {r.model} | {_pct(r.detection_rate)} ({r.n_drowsy}) | {_num(r.latency_median_s, 0)} | "
                     f"{_pct(r.drowsy_time_in_alarm)} | {_pct(r.alert_videos_without_alarm)} ({r.n_alert}) | "
                     f"{_num(r.false_alarms_per_hour)} | {_pct(r.balanced_accuracy)} |")
    lines += [
        "",
        "## Efecto de la persistencia",
        "",
        "Más persistencia reduce las falsas alarmas a costa de alarmar más tarde. Ver `persistencia.png`.",
        "",
        "| Modelo | Persistencia (s) | Detectados | Latencia mediana (s) | Falsas alarmas por hora |",
        "|---|---|---|---|---|",
    ]
    for r in summary.sort_values(["model", "persistence"]).itertuples():
        lines.append(f"| {r.model} | {r.persistence} | {_pct(r.detection_rate)} | {_num(r.latency_median_s, 0)} | "
                     f"{_num(r.false_alarms_per_hour)} |")
    lines += [
        "",
        "## Predicción por segundo (sin lógica de alertas)",
        "",
        "Sólo segundos con ventana completa y rostro visible. La etiqueta de cada segundo es la del video.",
        "",
        "| Modelo | Segundos | Exactitud | Precisión | Recall | Especificidad | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in seconds.itertuples():
        lines.append(f"| {r.model} | {r.seconds} | {_num(r.accuracy, 3)} | {_num(r.precision, 3)} | "
                     f"{_num(r.recall, 3)} | {_num(r.specificity, 3)} | {_num(r.f1, 3)} | {_num(r.roc_auc, 3)} |")
    if not perf.empty:
        p = perf.iloc[0]
        lines += [
            "",
            "## Rendimiento en este computador",
            "",
            "| Etapa | ms por frame |",
            "|---|---|",
            f"| Detección de rostro y landmarks (MediaPipe) | {p.detect_ms:.1f} |",
            f"| Características por frame y ventana | {p.features_ms:.2f} |",
            f"| Total por frame (media) | {p.frame_ms:.1f} |",
            f"| Total por frame (percentil 95, peor video) | {p.frame_ms_p95_max:.1f} |",
            f"| Clasificador (una predicción por segundo) | {p.model_ms:.2f} |",
            "",
            f"Frames por segundo alcanzables: **{p.max_fps:.0f}**. La plataforma necesita {SAMPLE_FPS:.0f}, "
            f"así que tiene un margen de {p.realtime_margin:.1f} veces sobre el tiempo real.",
        ]
    if rec is not None and not rec.empty:
        base = summary[summary["persistence"] == default_p].set_index("model")
        lines += [
            "",
            "## Umbral de puntaje",
            "",
            f"En lugar de la decisión del modelo, se exige un puntaje mínimo para contar un segundo como fatiga "
            f"(persistencia {default_p} s). Para cada modelo se recomienda el umbral con menos falsas alarmas "
            f"que todavía detecta al menos el {min_detection:.0%} de los videos somnolientos. Ver `umbral.png`.",
            "",
            "| Modelo | Umbral | Detectados | Latencia mediana (s) | Videos alerta sin alarmas "
            "| Falsas alarmas por hora | Antes (criterio del modelo) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rec.itertuples():
            before = _num(base.loc[r.model, "false_alarms_per_hour"]) if r.model in base.index else "--"
            lines.append(f"| {r.model} | **{r.threshold:.3g}** | {_pct(r.detection_rate)} | "
                         f"{_num(r.latency_median_s, 0)} | {_pct(r.alert_videos_without_alarm)} | "
                         f"{_num(r.false_alarms_per_hour)} | {before} falsas alarmas por hora |")
        lines += [
            "",
            "El umbral se eligió con estos mismos sujetos, así que estas cifras son optimistas: preséntalas como "
            "análisis de sensibilidad, no como resultado de prueba. Además se calcularon con los modelos de "
            "`models_part1`; los reentrenados con más sujetos producen puntajes en una escala parecida pero no "
            "idéntica, así que el valor es un punto de partida que se confirma en las pruebas reales.",
            "",
            "Para usarlo: en la plataforma, «Umbral de puntaje» en la barra lateral; en consola, `--threshold`.",
        ]
    lines += [
        "",
        "## Archivos",
        "",
        "- `por_video.csv`: métricas de cada video, modelo y persistencia.",
        "- `umbral.csv`, `umbral.png`: todos los umbrales probados y el compromiso detección/falsas alarmas.",
        "- `resumen_plataforma.csv`, `por_segundo.csv`, `rendimiento.csv`: las tablas de este reporte.",
        "- `persistencia.png`: detección, latencia y falsas alarmas según la persistencia.",
        "",
    ]
    text = "\n".join(lines)
    (out / "resumen.md").write_text(text, encoding="utf-8")
    return text


def load_traces(trace_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    csvs = sorted(trace_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No hay trazas en {trace_dir}. Corre primero alertdrive.evaluation.platform_eval")
    traces = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    traces["ready"] = traces["ready"].astype(bool)
    traces["valid"] = traces["valid"].astype(bool)
    traces["is_drowsy"] = traces["is_drowsy"].astype(bool)
    perf = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(trace_dir.glob("*.json"))]
    return traces, perf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/fase5", help="Misma carpeta usada en platform_eval")
    ap.add_argument("--persistence", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 10])
    ap.add_argument("--default-persistence", type=int, default=3, help="La que usa la plataforma")
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--min-detection", type=float, default=0.9,
                    help="Proporción mínima de videos somnolientos detectados al recomendar el umbral")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = Path(args.out)
    traces, perf_raw = load_traces(out / "trazas")
    persistences = sorted(set(args.persistence) | {args.default_persistence})
    videos = per_video_table(traces, persistences, args.cooldown)
    summary = summarize_platform(videos)
    seconds = per_second_metrics(traces)
    perf = performance_table(perf_raw)

    videos.to_csv(out / "por_video.csv", index=False)
    summary.to_csv(out / "resumen_plataforma.csv", index=False)
    seconds.to_csv(out / "por_segundo.csv", index=False)
    perf.to_csv(out / "rendimiento.csv", index=False)
    write_figure(summary, out / "persistencia.png")

    sweep = threshold_sweep(traces, args.default_persistence, args.cooldown)
    rec = recommend_thresholds(sweep, args.min_detection) if not sweep.empty else pd.DataFrame()
    sweep.to_csv(out / "umbral.csv", index=False)
    if not sweep.empty:
        write_threshold_figure(sweep, rec, summary[summary["persistence"] == args.default_persistence],
                               out / "umbral.png")
    print(write_report(out, traces, summary, seconds, perf, args.cooldown, args.default_persistence,
                       rec, args.min_detection))
    logger.info("Reporte en %s", (out / "resumen.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
