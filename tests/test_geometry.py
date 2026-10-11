import math

import numpy as np
import pytest

from alertdrive.features.geometry import (
    euler_to_rotation_deg,
    eyes_ear,
    mouth_aspect_ratio,
    rotation_to_euler_deg,
)
from helpers import synthetic_points


@pytest.mark.parametrize("ear", [0.0, 0.1, 0.3])
def test_ear_matches_construction(ear):
    r, l = eyes_ear(synthetic_points(ear=ear))
    assert r == pytest.approx(ear)
    assert l == pytest.approx(ear)


@pytest.mark.parametrize("mar", [0.0, 0.3, 0.8])
def test_mar_matches_construction(mar):
    assert mouth_aspect_ratio(synthetic_points(mar=mar)) == pytest.approx(mar)


def test_degenerate_eye_returns_zero():
    assert eyes_ear(np.zeros((478, 2))) == (0.0, 0.0)


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, 0, 0), (0, -30, 0), (0, 0, 15), (-12, 25, 8)])
def test_euler_round_trip(angles):
    got = rotation_to_euler_deg(euler_to_rotation_deg(*angles))
    assert got == pytest.approx(angles, abs=1e-6)


def test_accepts_4x4_transform():
    m = np.eye(4)
    m[:3, :3] = euler_to_rotation_deg(10, 5, -3)
    assert rotation_to_euler_deg(m) == pytest.approx((10, 5, -3), abs=1e-6)
    assert not any(math.isnan(v) for v in rotation_to_euler_deg(m))
