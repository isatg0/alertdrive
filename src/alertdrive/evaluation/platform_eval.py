"""Evalúa la plataforma completa sobre videos etiquetados.

A diferencia de la Fase 2, que mide el clasificador ventana por ventana, aquí
se reproduce lo que hace la plataforma en vivo: muestreo a 5 fps, predicción
cada segundo sobre los últimos 15 s y, después (en ``analyze``), la lógica de
alertas con persistencia y enfriamiento.

La visión (MediaPipe + características) se calcula una sola vez por video y
se aplica a todos los modelos, porque es lo costoso. Por cada video se guarda:

- ``trazas/<video>.csv``: una fila por segundo y modelo (estado listo/válido,
  predicción, puntaje, motivos).
- ``trazas/<video>.json``: tiempos de procesamiento por frame.

Es reanudable: los videos ya procesados se saltan.

    python -m alertdrive.evaluation.platform_eval --manifest manifests/uta_part2.csv \
        --models models_part1/drowsiness_rf.joblib models_part1/drowsiness_svm.joblib \
                 models_part1/drowsiness_rules.joblib --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("alertdrive.evaluation")

_ALERT = {"0", "alert", "alerta", "awake", "normal", "no_drowsy", "nodrowsy", "not_drowsy"}
_DROWSY = {"1", "10", "drowsy", "somnoliento", "sleepy", "fatigue"}
_LOW = {"5", "low_vigilant", "low-vigilant", "lowvigilant", "low vigilant", "baja_vigilancia"}


def binary_label(value: object) -> int | str | None:
    """Etiqueta del manifiesto → 0 (alerta), 1 (somnoliento), 'low' o None si no se reconoce."""
    text = str(value).strip().lower()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    if text in _ALERT:
        return 0
    if text in _DROWSY:
        return 1
    if text in _LOW:
        return "low"
    return None


def load_manifests(paths: list[str], low_vigilant: str = "drop") -> pd.DataFrame:
    """Une los manifiestos y deja sólo los videos con etiqueta binaria utilizable."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"video_path", "subject_id", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{p}: faltan columnas {sorted(missing)}")
        base = Path(p).resolve().parent
        df["video_path"] = [
            str(v) if Path(v).exists() or not (base / v).exists() else str(base / v) for v in df["video_path"]
        ]
        if "dataset" not in df.columns:
            df["dataset"] = Path(p).stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df["label"].map(binary_label)

    unknown = sorted(df.loc[df["y"].isna(), "label"].astype(str).unique())
    if unknown:
        logger.warning("Etiquetas no reconocidas (se descartan): %s", unknown)
    if low_vigilant == "positive":
        df.loc[df["y"] == "low", "y"] = 1
    elif low_vigilant == "negative":
        df.loc[df["y"] == "low", "y"] = 0
    df = df[df["y"].isin([0, 1])].copy()
    df["y"] = df["y"].astype(int)

    df["video_key"] = [
        re.sub(r"[^A-Za-z0-9_.-]+", "_",
               f"{s}_{Path(v).stem}" if str(s).startswith(str(d)) else f"{d}_{s}_{Path(v).stem}")
        for d, s, v in zip(df["dataset"], df["subject_id"], df["video_path"])
    ]
    if df["video_key"].duplicated().any():  # mismo sujeto y nombre en dos rutas distintas
        df["video_key"] = df["video_key"] + "_" + df.groupby("video_key").cumcount().astype(str)
    return df.reset_index(drop=True)


def _model_name(path: str) -> str:
    from alertdrive.web.catalog import model_label

    return model_label(path)


def process_video(row: dict, model_paths: list[str], sample_fps: float, max_seconds: float | None,
                  predict_every_s: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Corre visión + características una vez y predice con todos los modelos cada segundo."""
    from alertdrive.config import FeatureConfig
    from alertdrive.features.extractor import FrameFeatureExtractor
    from alertdrive.features.window import WindowAggregator
    from alertdrive.pipeline import iter_video_frames
    from alertdrive.vision.face_landmarker import FaceLandmarkDetector
    from alertdrive.web.engine import load_predictor

    config = FeatureConfig()
    window_s = config.window.window_s
    predictors = [(_model_name(p), load_predictor(p)) for p in model_paths]
    extractor = FrameFeatureExtractor(config)
    aggregator = WindowAggregator(config)

    rows: list[dict] = []
    detect_ms, feat_ms, model_ms = [], [], []
    t0, last_pred, t = None, -math.inf, 0.0
    with FaceLandmarkDetector(config.landmarker) as detector:
        for _, t, _, frame in iter_video_frames(row["video_path"], sample_fps, max_seconds):
            if t0 is None:
                t0 = t
            a = time.perf_counter()
            obs, _ = detector.process(frame, t)
            b = time.perf_counter()
            aggregator.add(extractor.update(obs))
            c = time.perf_counter()
            detect_ms.append((b - a) * 1000)
            feat_ms.append((c - b) * 1000)
            if t - last_pred < predict_every_s:
                continue
            last_pred = t
            feats = aggregator.compute(t)
            ready = (t - t0) >= window_s
            for name, pred in predictors:
                m = time.perf_counter()
                res = pred.predict(feats)
                model_ms.append((time.perf_counter() - m) * 1000)
                rows.append({
                    "video_key": row["video_key"], "dataset": row["dataset"], "subject_id": row["subject_id"],
                    "y": row["y"], "model": name, "t": round(t, 3), "ready": ready,
                    "valid": bool(getattr(res, "valid", feats.get("valid", 1.0))),
                    "is_drowsy": bool(res.is_drowsy), "score": float(res.score),
                    "reasons": "; ".join(res.reasons), "perclos": feats.get("perclos"),
                })

    def stats(v: list[float]) -> dict:
        return {"mean": float(np.mean(v)) if v else math.nan, "p95": float(np.percentile(v, 95)) if v else math.nan}

    perf = {
        "video_key": row["video_key"], "video_path": row["video_path"], "frames": len(detect_ms),
        "seconds": float(t), "detect_ms": stats(detect_ms), "features_ms": stats(feat_ms),
        "model_ms": stats(model_ms),
        "frame_ms": stats([d + f for d, f in zip(detect_ms, feat_ms)]),
    }
    return pd.DataFrame(rows), perf


def _run_one(args: tuple) -> tuple[str, str | None]:
    row, model_paths, sample_fps, max_seconds, out_dir = args
    try:
        traces, perf = process_video(row, model_paths, sample_fps, max_seconds)
        out = Path(out_dir)
        traces.to_csv(out / f"{row['video_key']}.csv", index=False)
        (out / f"{row['video_key']}.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
        return row["video_key"], None
    except Exception as exc:  # un video dañado no detiene la corrida
        return row["video_key"], f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, help="Uno o varios manifiestos CSV")
    ap.add_argument("--models", nargs="+", default=["models/drowsiness_rf.joblib", "models/drowsiness_svm.joblib",
                                                    "models/drowsiness_rules.joblib"])
    ap.add_argument("--out", default="reports/fase5", help="Carpeta de salida (las trazas van en <out>/trazas)")
    ap.add_argument("--sample-fps", type=float, default=5.0, help="Igual que en build_features y la app")
    ap.add_argument("--max-seconds", type=float, default=None, help="Analiza sólo el inicio de cada video")
    ap.add_argument("--low-vigilant", choices=["drop", "positive", "negative"], default="drop")
    ap.add_argument("--limit", type=int, default=None, help="Procesa sólo los primeros N videos (prueba)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="Reprocesa videos ya hechos")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for m in args.models:
        if not Path(m).exists():
            logger.error("No existe el modelo %s", m)
            return 1
    videos = load_manifests(args.manifest, args.low_vigilant)
    if args.limit:
        videos = videos.head(args.limit)
    out_dir = Path(args.out) / "trazas"
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [r for r in videos.to_dict("records")
               if args.overwrite or not (out_dir / f"{r['video_key']}.csv").exists()]
    logger.info("Videos: %d (%d sujetos) | alerta: %d, somnoliento: %d | pendientes: %d | modelos: %s",
                len(videos), videos["subject_id"].nunique(), int((videos["y"] == 0).sum()),
                int((videos["y"] == 1).sum()), len(pending), ", ".join(_model_name(m) for m in args.models))
    missing = [r["video_path"] for r in pending if not Path(r["video_path"]).exists()]
    if missing:
        logger.warning("%d videos no existen y se saltan, p. ej. %s", len(missing), missing[0])
        pending = [r for r in pending if Path(r["video_path"]).exists()]

    jobs = [(r, args.models, args.sample_fps, args.max_seconds, str(out_dir)) for r in pending]
    start, failures = time.perf_counter(), 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                key, err = fut.result()
                failures += err is not None
                logger.info("[%d/%d] %s %s", i, len(jobs), key, f"ERROR {err}" if err else "listo")
    else:
        for i, job in enumerate(jobs, 1):
            key, err = _run_one(job)
            failures += err is not None
            logger.info("[%d/%d] %s %s", i, len(jobs), key, f"ERROR {err}" if err else "listo")

    logger.info("Terminado en %.1f min (%d con error). Siguiente paso: "
                "python -m alertdrive.evaluation.analyze --out %s", (time.perf_counter() - start) / 60,
                failures, args.out)
    return 1 if failures and failures == len(jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
