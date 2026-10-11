"""Prueba de extremo a extremo con MediaPipe real.

Requiere models/face_landmarker.task y una foto frontal en tests/assets/face.jpg
(cualquier foto de un rostro). Si falta alguno, la prueba se omite.
"""
from pathlib import Path

import cv2
import pytest

from alertdrive.config import DEFAULT_MODEL_PATH, FeatureConfig, WindowConfig
from alertdrive.pipeline import extract_video_windows

ASSET = Path(__file__).parent / "assets" / "face.jpg"

pytestmark = pytest.mark.skipif(
    not (ASSET.exists() and DEFAULT_MODEL_PATH.exists()), reason="falta el modelo o tests/assets/face.jpg"
)


def test_video_pipeline_on_still_face(tmp_path):
    img = cv2.imread(str(ASSET))
    h, w = img.shape[:2]
    video = tmp_path / "face.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for _ in range(60):  # 6 s a 10 fps
        writer.write(img)
    writer.release()

    config = FeatureConfig(window=WindowConfig(window_s=3.0, stride_s=1.0))
    rows = extract_video_windows(video, config, sample_fps=10)
    assert len(rows) >= 2
    last = rows[-1]
    assert last["face_present_ratio"] == pytest.approx(1.0)
    assert 0.15 < last["ear_mean"] < 0.45  # ojos abiertos
    assert last["yawn_count"] == 0
