"""Datos SINTÉTICOS con el mismo formato que ``build_features``.

Sirven sólo para probar el flujo de la Fase 2 mientras llega el dataset real.
Las métricas que salgan con estos datos NO significan nada y no van al informe.

Cada sujeto tiene una línea base propia (EAR, frecuencia de parpadeo) y cada
video un nivel latente de somnolencia que varía ventana a ventana. Los videos
"alert" también tienen momentos de cansancio y los "drowsy" momentos de
alerta, como pasa con las etiquetas por video de UTA-RLDD.

Uso::

    python -m alertdrive.classify.synthetic --out data/features/synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alertdrive.features.window import FEATURE_COLUMNS, META_COLUMNS

VIDEO_LEVELS = {"alert": (1.5, 7.0), "low_vigilant": (2.5, 4.0), "drowsy": (3.5, 3.0)}  # Beta(a, b)


def make_synthetic_features(n_subjects: int = 24, windows_per_video: int = 60,
                            stride_s: float = 5.0, window_s: float = 15.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        base_ear = rng.normal(0.29, 0.03)
        base_blink = max(6.0, rng.normal(16, 5))
        base_pitch = abs(rng.normal(3, 1.5))
        for label, (a, b) in VIDEO_LEVELS.items():
            level = rng.beta(a, b)
            d = level
            for w in range(windows_per_video):
                # Nivel latente con autocorrelación entre ventanas consecutivas.
                d = float(np.clip(0.8 * d + 0.2 * rng.beta(a, b) + rng.normal(0, 0.05), 0, 1))
                micro = rng.poisson(1.6 * d ** 2)
                perclos = float(np.clip(0.03 + 0.35 * d ** 1.5 + 0.05 * micro + rng.normal(0, 0.03), 0, 1))
                yawns = rng.poisson(0.08 + 0.7 * d)
                ear_mean = base_ear * (1 - 0.5 * perclos) + rng.normal(0, 0.01)
                feats = {
                    "face_present_ratio": rng.uniform(0.85, 1.0),
                    "ear_mean": ear_mean,
                    "ear_std": abs(rng.normal(0.03 + 0.05 * perclos, 0.01)),
                    "ear_min": max(0.02, ear_mean - rng.uniform(0.12, 0.2)),
                    "ear_rel_mean": ear_mean / base_ear,
                    "ear_rel_min": max(0.05, (ear_mean - 0.15) / base_ear),
                    "perclos": perclos,
                    "blink_rate_per_min": max(0.0, base_blink * (1 + 0.4 * d) + rng.normal(0, 3)),
                    "blink_dur_mean_s": max(0.08, 0.15 + 0.22 * d + rng.normal(0, 0.04)),
                    "long_closure_count": rng.poisson(0.2 + 1.5 * d),
                    "microsleep_count": micro,
                    "max_closure_s": 1.0 + rng.exponential(0.5) if micro else rng.uniform(0.15, 0.9),
                    "mar_mean": max(0.0, rng.normal(0.08 + 0.05 * yawns, 0.02)),
                    "mar_max": max(0.1, rng.normal(0.3 + 0.35 * min(yawns, 1), 0.08)),
                    "mouth_open_fraction": float(np.clip(0.02 + 0.12 * yawns + rng.normal(0, 0.02), 0, 1)),
                    "yawn_count": yawns,
                    "pitch_dev_mean": abs(rng.normal(base_pitch + 6 * d, 1.5)),
                    "pitch_dev_std": abs(rng.normal(2 + 4 * d, 1)),
                    "yaw_dev_std": abs(rng.normal(4, 1.5)),
                    "roll_abs_mean": abs(rng.normal(3, 1.5)),
                    "head_off_fraction": float(np.clip(0.02 + 0.35 * d ** 2 + rng.normal(0, 0.03), 0, 1)),
                    "looking_away_fraction": float(np.clip(rng.normal(0.05, 0.03), 0, 1)),
                    "nod_count": rng.poisson(0.05 + 0.9 * d ** 2),
                    "bs_eye_blink_mean": float(np.clip(0.1 + 0.5 * perclos + rng.normal(0, 0.03), 0, 1)),
                    "bs_jaw_open_max": float(np.clip(0.15 + 0.45 * min(yawns, 1) + rng.normal(0, 0.05), 0, 1)),
                }
                t_end = window_s + w * stride_s
                rows.append({
                    "video_path": f"synthetic/{s:02d}/{label}.mp4",
                    "subject_id": f"syn_{s:02d}",
                    "label": label,
                    "dataset": "synthetic",
                    "t_start": t_end - window_s,
                    "t_end": t_end,
                    "n_frames": int(window_s * 5),
                    "valid": float(rng.random() > 0.02),
                    **feats,
                    "label_fraction": np.nan,
                })
    cols = ["video_path", "subject_id", "label", "dataset", *META_COLUMNS, *FEATURE_COLUMNS, "label_fraction"]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un CSV de características sintético.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--windows", type=int, default=60, help="Ventanas por video")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    df = make_synthetic_features(args.subjects, args.windows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"CSV sintético escrito en {args.out}: {len(df)} filas, {df.subject_id.nunique()} sujetos")


if __name__ == "__main__":
    main()
