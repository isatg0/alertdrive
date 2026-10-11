"""Pruebas del motor con un detector y un predictor falsos (no usan MediaPipe)."""
from types import SimpleNamespace

import numpy as np

from alertdrive.vision.landmarks import FaceObservation
from alertdrive.web.engine import DrowsinessEngine


class FakeDetector:
    def __init__(self):
        self.closed = False

    def process(self, frame_bgr, timestamp_s):
        return FaceObservation.absent(timestamp_s, (frame_bgr.shape[1], frame_bgr.shape[0])), frame_bgr

    def close(self):
        self.closed = True


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, feats):
        self.calls.append(feats)
        return SimpleNamespace(valid=feats["valid"] > 0.5, is_drowsy=False, score=0.1, reasons=[])


def _run(seconds, fps=5.0):
    det, pred = FakeDetector(), FakePredictor()
    engine = DrowsinessEngine(predictor=pred, detector=det, predict_every_s=1.0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    outs = [engine.process(frame, i / fps) for i in range(int(seconds * fps))]
    engine.close()
    return engine, det, pred, outs


def test_predice_una_vez_por_segundo():
    _, _, pred, outs = _run(seconds=10)
    assert len(pred.calls) == 10
    assert sum(o.new_prediction for o in outs) == 10


def test_calibracion_dura_una_ventana():
    engine, _, _, outs = _run(seconds=20)
    ready_times = [o.t for o in outs if o.new_prediction and o.prediction.ready]
    assert ready_times and min(ready_times) >= engine.window_s


def test_sin_rostro_la_ventana_no_es_valida():
    _, det, _, outs = _run(seconds=20)
    assert all(not o.prediction.valid for o in outs if o.new_prediction)
    assert det.closed


class ScorePredictor:
    """Devuelve el puntaje fijado y una decisión propia que el umbral debe ignorar."""

    def __init__(self, score):
        self.score = score

    def predict(self, feats):
        return SimpleNamespace(valid=True, is_drowsy=False, score=self.score, reasons=[])


def _last_prediction(threshold, score):
    engine = DrowsinessEngine(predictor=ScorePredictor(score), detector=FakeDetector(), threshold=threshold)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = [engine.process(frame, i / 5) for i in range(10)][-1]
    return out.prediction


def test_umbral_reemplaza_la_decision_del_modelo():
    assert not _last_prediction(None, 0.9).is_drowsy        # sin umbral decide el modelo
    assert _last_prediction(0.7, 0.9).is_drowsy
    assert not _last_prediction(0.7, 0.6).is_drowsy


def test_carga_en_un_solo_hilo():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from alertdrive.web.engine import _single_thread

    rf = RandomForestClassifier(n_estimators=5, n_jobs=-1)
    holder = SimpleNamespace(model=Pipeline([("clf", rf)]), extra={"otro": [RandomForestClassifier(n_jobs=4)]})
    _single_thread(holder)
    assert rf.n_jobs == 1
    assert holder.extra["otro"][0].n_jobs == 1
