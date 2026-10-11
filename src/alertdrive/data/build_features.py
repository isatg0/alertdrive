"""Construye el CSV de características por ventana a partir de un manifiesto.

Uso::

    python -m alertdrive.data.build_features --manifest manifests/uta.csv \\
        --out data/features/uta.csv --sample-fps 5 --workers 4

- Reanudable: si se interrumpe, al volver a ejecutarlo salta los videos ya
  procesados en el CSV de salida.
- Paralelo por video (``--workers``): cada proceso crea su propio detector.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from alertdrive.config import FeatureConfig, WindowConfig
from alertdrive.data.manifest import ManifestEntry, read_manifest
from alertdrive.features.window import FEATURE_COLUMNS, META_COLUMNS
from alertdrive.pipeline import extract_video_windows, read_frame_labels

logger = logging.getLogger(__name__)

ID_COLUMNS = ("video_path", "subject_id", "label", "dataset")
OUTPUT_COLUMNS = ID_COLUMNS + META_COLUMNS + FEATURE_COLUMNS + ("label_fraction",)


def _process(
    entry: ManifestEntry, sample_fps: float, max_seconds: float | None, window_s: float, stride_s: float
) -> tuple[ManifestEntry, list[dict], str | None]:
    config = replace(FeatureConfig(), window=replace(WindowConfig(), window_s=window_s, stride_s=stride_s))
    try:
        labels = read_frame_labels(entry.frame_labels_path) if entry.frame_labels_path else None
        rows = extract_video_windows(entry.video_path, config, sample_fps, max_seconds, labels)
        return entry, rows, None
    except Exception as exc:  # un video corrupto no debe detener todo el lote
        return entry, [], f"{type(exc).__name__}: {exc}"


def _already_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    with out.open(newline="", encoding="utf-8") as fh:
        return {row["video_path"] for row in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae características por ventana de cada video.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Frames por segundo a procesar")
    parser.add_argument("--max-seconds", type=float, default=None, help="Recorta cada video (pruebas rápidas)")
    parser.add_argument("--window-s", type=float, default=WindowConfig.window_s)
    parser.add_argument("--stride-s", type=float, default=WindowConfig.stride_s)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Procesa sólo los primeros N videos")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    entries = read_manifest(args.manifest)
    done = _already_done(args.out)
    pending = [e for e in entries if e.video_path not in done][: args.limit]
    logger.info("%d videos en el manifiesto, %d ya procesados, %d pendientes", len(entries), len(done), len(pending))
    if not pending:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    started = time.perf_counter()
    failures: list[str] = []

    with args.out.open("a", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        futures = [
            pool.submit(_process, e, args.sample_fps, args.max_seconds, args.window_s, args.stride_s)
            for e in pending
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            entry, rows, error = fut.result()
            if error:
                failures.append(f"{entry.video_path}: {error}")
                logger.error("[%d/%d] FALLÓ %s — %s", i, len(pending), entry.video_path, error)
                continue
            ids = {k: v for k, v in asdict(entry).items() if k in ID_COLUMNS}
            for row in rows:
                writer.writerow({**ids, **row})
            fh.flush()
            logger.info("[%d/%d] %s → %d ventanas", i, len(pending), Path(entry.video_path).name, len(rows))

    elapsed = time.perf_counter() - started
    logger.info("Terminado en %.1f min. Fallos: %d", elapsed / 60, len(failures))
    for f in failures:
        logger.info("  %s", f)


if __name__ == "__main__":
    main()
