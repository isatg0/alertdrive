"""Fase 2: evalúa los algoritmos por sujeto, genera el reporte y guarda los modelos.

Uso::

    # Prueba rápida del flujo (datos sintéticos o subconjunto real)
    python -m alertdrive.classify.train --features data/features/uta.csv --fast

    # Corrida completa
    python -m alertdrive.classify.train --features data/features/uta.csv \\
        --models rules rf svm --folds 5 --subsample 3

Salidas:
- ``reports/fase2/``: resumen.md, metricas.json, metricas_por_particion.csv,
  roc.png, confusion.png, importancia_rf.png
- ``models/drowsiness_<modelo>.joblib``: modelos entrenados con todos los datos
- ``models/drowsiness_best.joblib``: copia del mejor por F1 medio
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from alertdrive.classify.dataset import LOW_VIGILANT_POLICIES, load_features
from alertdrive.classify.evaluate import (
    bootstrap_f1_difference,
    cross_validate,
    fit_model,
    pooled_metrics,
    video_level_metrics,
)
from alertdrive.classify.models import MODELS
from alertdrive.classify.predictor import save_bundle
from alertdrive.classify.report import write_report
from alertdrive.config import PROJECT_ROOT

logger = logging.getLogger("alertdrive.classify")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Compara el algoritmo de reglas contra ML, por sujeto.")
    parser.add_argument("--features", type=Path, nargs="+", required=True, help="Uno o varios CSV de la Fase 1")
    parser.add_argument("--models", nargs="+", default=["rules", "rf", "svm"], choices=sorted(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1,
                        help="Usar 1 de cada N ventanas por video al entrenar (3 = sin solapamiento)")
    parser.add_argument("--low-vigilant", choices=LOW_VIGILANT_POLICIES, default="drop")
    parser.add_argument("--fast", action="store_true", help="Rejilla de hiperparámetros reducida")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "fase2")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--no-final", action="store_true", help="No entrenar ni guardar los modelos finales")
    parser.add_argument("--jobs", type=int, default=-1, help="Procesos para scikit-learn (-1 = todos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = load_features(args.features, low_vigilant=args.low_vigilant)
    logger.info("Datos: %s", data.summary())

    t0 = time.perf_counter()
    results, _ = cross_validate(data, args.models, n_splits=args.folds, subsample=args.subsample,
                                fast=args.fast, random_state=args.seed, n_jobs=args.jobs)
    pooled = {n: pooled_metrics(data, r) for n, r in results.items()}
    video = {n: video_level_metrics(data, r) for n, r in results.items()}
    comparisons = {}
    if "rules" in results:
        comparisons = {n: bootstrap_f1_difference(data, results["rules"].oof_pred, r.oof_pred, seed=args.seed)
                       for n, r in results.items() if n != "rules"}
    best = max(results, key=lambda n: results[n].summary()["f1"]["mean"])

    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    settings["features"] = [str(p) for p in args.features]
    write_report(args.out_dir, data, results, pooled, video, comparisons, settings, best)
    logger.info("Evaluación terminada en %.1f min. Reporte en %s", (time.perf_counter() - t0) / 60, args.out_dir)

    if not args.no_final:
        trained_on = {"features": settings["features"], "summary": data.summary(),
                      "low_vigilant": args.low_vigilant, "subsample": args.subsample}
        for name in args.models:
            model, params = fit_model(name, data, args.subsample, args.fast, args.seed, args.jobs)
            path = save_bundle(args.models_dir / f"drowsiness_{name}.joblib", model, name, params, trained_on)
            logger.info("Modelo final guardado: %s %s", path, params or "")
        shutil.copyfile(args.models_dir / f"drowsiness_{best}.joblib", args.models_dir / "drowsiness_best.joblib")

    print("\n" + (args.out_dir / "resumen.md").read_text(encoding="utf-8"))
    return {"results": results, "best": best, "data": data,
            "f1": {n: float(np.mean([m["f1"] for m in r.fold_metrics])) for n, r in results.items()}}


if __name__ == "__main__":
    main()
