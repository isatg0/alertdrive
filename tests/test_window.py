"""Escenario sintético de 40 s a 10 fps con parpadeos, un microsueño y un bostezo."""
import numpy as np
import pytest

from alertdrive.config import FeatureConfig
from alertdrive.features.extractor import FrameFeatureExtractor
from alertdrive.features.geometry import euler_to_rotation_deg
from alertdrive.features.window import FEATURE_COLUMNS, WindowAggregator
from alertdrive.vision.landmarks import FaceObservation
from helpers import synthetic_obs

FPS = 10


def scenario(t: float) -> FaceObservation:
    ear, mar, pitch = 0.30, 0.05, 0.0
    if int(round(t * FPS)) % 30 in (0, 1):  # parpadeo de 0.2 s cada 3 s
        ear = 0.05
    if 25.0 <= t < 26.5:  # microsueño de 1.5 s
        ear = 0.05
    if 30.0 <= t < 32.0:  # bostezo de 2 s
        mar = 0.8
    if 34.0 <= t < 35.0:  # cabeceo de 1 s
        pitch = 25.0
    tf = np.eye(4)
    tf[:3, :3] = euler_to_rotation_deg(pitch, 0, 0)
    return synthetic_obs(t, ear=ear, mar=mar, transform=tf)


def run(seconds: float = 40.0):
    config = FeatureConfig()
    ext, agg = FrameFeatureExtractor(config), WindowAggregator(config)
    emitted = []
    for i in range(int(seconds * FPS)):
        if (w := agg.add(ext.update(scenario(i / FPS)))) is not None:
            emitted.append(w)
    return agg, emitted


def test_windows_are_emitted_every_stride():
    _, emitted = run()
    ends = [w["t_end"] for w in emitted]
    assert ends[0] == pytest.approx(15.0)
    assert np.diff(ends) == pytest.approx([5.0] * (len(ends) - 1))
    assert all(set(FEATURE_COLUMNS) <= w.keys() for w in emitted)


def test_calm_window_has_normal_blinks_and_no_events():
    _, emitted = run()
    calm = next(w for w in emitted if w["t_end"] == pytest.approx(20.0))
    assert calm["blink_rate_per_min"] == pytest.approx(20, abs=4)  # 1 cada 3 s
    assert calm["microsleep_count"] == 0
    assert calm["yawn_count"] == 0
    assert calm["perclos"] < 0.15
    assert calm["valid"] == 1.0


def test_drowsy_window_detects_microsleep_yawn_and_nod():
    agg, _ = run()
    w = agg.compute(39.9)
    assert w["microsleep_count"] == 1
    assert w["max_closure_s"] == pytest.approx(1.5, abs=0.15)
    assert w["yawn_count"] == 1
    assert w["nod_count"] == 1
    assert w["head_off_fraction"] > 0


def test_absent_face_lowers_presence_ratio():
    config = FeatureConfig()
    ext, agg = FrameFeatureExtractor(config), WindowAggregator(config)
    for i in range(200):
        t = i / FPS
        obs = FaceObservation.absent(t) if i % 2 else synthetic_obs(t)
        agg.add(ext.update(obs))
    w = agg.compute(19.9)
    assert w["face_present_ratio"] == pytest.approx(0.5, abs=0.02)
