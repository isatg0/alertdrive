"""Ejecuta el motor de la Fase 3 sobre un video, sin interfaz.

Sirve para comprobar el motor antes de abrir la app y para la Fase 5
(cuántas alarmas da en un video alerta vs. uno somnoliento). Con ``--save`` la
sesión queda en el histórico y aparece en la página Historial.

    python scripts/monitor_video.py --video RUTA\\video.mp4
    python scripts/monitor_video.py --video RUTA\\video.mp4 --model models/drowsiness_rf.joblib --max-seconds 120
    python scripts/monitor_video.py --video RUTA\\10.mp4 --save
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from alertdrive.pipeline import iter_video_frames
from alertdrive.storage.history import DEFAULT_DB_PATH, HistoryDB, SessionLogger
from alertdrive.web.alerts import AlertManager
from alertdrive.web.catalog import model_label
from alertdrive.web.engine import DrowsinessEngine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/drowsiness_best.joblib")
    ap.add_argument("--sample-fps", type=float, default=5.0, help="Igual que en build_features")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--persistence", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Puntaje mínimo para contar fatiga (por defecto, el criterio del modelo)")
    ap.add_argument("--save", action="store_true", help="Guardar la sesión en el histórico (SQLite)")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Archivo de la base de datos")
    ap.add_argument("--label", default=None, help="Nombre de la fuente en el histórico (por defecto, la ruta)")
    args = ap.parse_args(argv)

    if not Path(args.video).exists():
        print(f"No existe el video: {args.video}", file=sys.stderr)
        return 1

    alerts = AlertManager(args.persistence, args.cooldown)
    logger = None
    if args.save:
        video = Path(args.video)
        logger = SessionLogger(
            HistoryDB(args.db), args.label or (f"{video.parent.name}/{video.name}" if video.parent.name else video.name), "video", model_label(args.model),
            {"persistence": args.persistence, "cooldown_s": args.cooldown, "sample_fps": args.sample_fps, "threshold": args.threshold,
             "origen": "monitor_video.py"},
        )
    counts: dict[str, int] = {}
    n_alarms, n_frames, t = 0, 0, 0.0
    start = time.perf_counter()
    with DrowsinessEngine(args.model, threshold=args.threshold) as engine:
        for _, t, _, frame in iter_video_frames(args.video, args.sample_fps, args.max_seconds):
            out = engine.process(frame, t)
            n_frames += 1
            if not out.new_prediction:
                continue
            p = out.prediction
            dec = alerts.update(t, p.ready, p.valid, p.is_drowsy, p.reasons)
            if logger is not None:
                logger.log(t, dec, p)
            counts[dec.level] = counts.get(dec.level, 0) + 1
            if dec.new_alarm:
                n_alarms += 1
                print(f"[{t:7.1f} s] ALARMA  puntaje={p.score:.2f}  motivos: {', '.join(dec.reasons) or '-'}")

    if logger is not None:
        logger.close()
    elapsed = time.perf_counter() - start
    total = sum(counts.values()) or 1
    print(f"\nVideo: {Path(args.video).name} | {t:.0f} s analizados | {n_frames} frames "
          f"| {n_frames / max(elapsed, 1e-6):.1f} frames/s de procesamiento")
    print(f"Alarmas: {n_alarms}")
    for level in ("calibrando", "sin_rostro", "ok", "atencion", "alarma"):
        if level in counts:
            print(f"  {level:<11} {counts[level] / total:6.1%} del tiempo")
    if logger is not None:
        print(f"Sesión guardada en {args.db} con el número {logger.session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
