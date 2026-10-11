"""Vista previa local para validar la extracción de características.

Muestra la webcam (o un video) con los contornos de ojos y boca y los
indicadores en vivo. No clasifica todavía: sirve para verificar que EAR, MAR,
PERCLOS, parpadeos, bostezos y pose se comportan como se espera antes de
entrenar modelos.

Uso::

    python scripts/live_preview.py              # webcam 0
    python scripts/live_preview.py --source 1   # otra webcam
    python scripts/live_preview.py --source video.mp4

Teclas: q = salir.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertdrive.config import FeatureConfig  # noqa: E402
from alertdrive.features.extractor import FrameFeatureExtractor  # noqa: E402
from alertdrive.features.window import WindowAggregator  # noqa: E402
from alertdrive.vision.face_landmarker import FaceLandmarkDetector  # noqa: E402
from alertdrive.vision.landmarks import (  # noqa: E402
    INNER_LIPS_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
)

GREEN, RED, YELLOW, WHITE = (80, 200, 80), (60, 60, 230), (0, 200, 230), (240, 240, 240)


def draw_contour(img: np.ndarray, pts: np.ndarray, idx: tuple[int, ...], color: tuple[int, int, int]) -> None:
    poly = np.array([pts[i] for i in idx], dtype=np.int32)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def put_lines(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 22
    for text, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Índice de webcam o ruta de video")
    args = parser.parse_args()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    is_file = isinstance(source, str)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = FeatureConfig()
    extractor = FrameFeatureExtractor(config)
    aggregator = WindowAggregator(config)
    t0 = time.monotonic()
    idx = 0
    window: dict[str, float] = {}
    last_window_t = -1.0

    with FaceLandmarkDetector(config.landmarker) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not is_file:
                frame = cv2.flip(frame, 1)  # efecto espejo en webcam
            t = idx / fps if is_file else time.monotonic() - t0
            idx += 1

            obs, shown = detector.process(frame, t)
            ff = extractor.update(obs)
            aggregator.add(ff)
            if t - last_window_t >= 0.5:  # recalcula la ventana dos veces por segundo
                window = aggregator.compute(t)
                last_window_t = t

            shown = shown.copy()
            if obs.face_present and obs.landmarks_px is not None:
                eye_color = RED if ff.eye_closed else GREEN
                draw_contour(shown, obs.landmarks_px, RIGHT_EYE_CONTOUR, eye_color)
                draw_contour(shown, obs.landmarks_px, LEFT_EYE_CONTOUR, eye_color)
                draw_contour(shown, obs.landmarks_px, INNER_LIPS_CONTOUR, YELLOW if ff.mouth_open else GREEN)
                lines = [
                    (f"EAR {ff.ear:.3f}  umbral {ff.ear_threshold:.3f}  {'CERRADO' if ff.eye_closed else 'abierto'}",
                     RED if ff.eye_closed else WHITE),
                    (f"MAR {ff.mar:.2f}  cierre actual {ff.closure_ongoing_s:.1f}s", WHITE),
                    (f"pitch {ff.pitch:+.0f}  yaw {ff.yaw:+.0f}  roll {ff.roll:+.0f}  (dev pitch {ff.pitch_dev:+.0f})",
                     YELLOW if ff.head_off else WHITE),
                ]
            else:
                lines = [("SIN ROSTRO", RED)]

            if window:
                w = config.window.window_s
                lines += [
                    (f"--- ventana {w:.0f}s ---", WHITE),
                    (f"PERCLOS {window['perclos']:.2f}  parpadeos/min {window['blink_rate_per_min']:.0f}", WHITE),
                    (f"cierre max {window['max_closure_s']:.1f}s  microsuenos {window['microsleep_count']:.0f}", WHITE),
                    (f"bostezos {window['yawn_count']:.0f}  cabeceos {window['nod_count']:.0f}", WHITE),
                    (f"rostro visible {window['face_present_ratio']:.0%}", WHITE),
                ]
            put_lines(shown, lines)
            cv2.imshow("AlertDrive - vista previa (q para salir)", shown)
            if cv2.waitKey(1 if not is_file else max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
