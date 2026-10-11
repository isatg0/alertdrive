import pytest

from alertdrive.features.temporal import AdaptiveBaseline, EventDetector


def test_baseline_waits_for_min_samples_and_forgets_old_values():
    b = AdaptiveBaseline(window_s=5.0, percentile=50, min_samples=3)
    b.update(0.0, 1.0)
    b.update(0.1, 1.0)
    assert b.value is None
    b.update(0.2, 1.0)
    assert b.value == 1.0
    for i in range(10):  # 10 s después sólo quedan valores nuevos
        b.update(10.0 + i * 0.1, 3.0)
    assert b.value == 3.0


def test_event_duration_at_10fps():
    d = EventDetector("eye_closure")
    signal = [False] * 5 + [True] * 3 + [False] * 3
    events = [d.update(i / 10, s) for i, s in enumerate(signal)]
    events = [e for e in events if e]
    assert len(events) == 1
    assert events[0].duration_s == pytest.approx(0.3)


def test_merge_gap_joins_flicker():
    d = EventDetector("yawn", merge_gap_s=0.25)
    signal = [True] * 10 + [False] + [True] * 10 + [False] * 5
    events = [e for i, s in enumerate(signal) if (e := d.update(i / 10, s))]
    assert len(events) == 1
    assert events[0].duration_s == pytest.approx(2.1)


def test_ongoing_and_flush():
    d = EventDetector("eye_closure")
    for i in range(12):
        d.update(i / 10, True)
    assert d.ongoing_duration(1.1) == pytest.approx(1.1)
    ev = d.flush(1.2)
    assert ev is not None and ev.duration_s == pytest.approx(1.2)
    assert not d.active
