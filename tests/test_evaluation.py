"""Pruebas de la evaluación de la Fase 5 (sin videos ni MediaPipe)."""
import math

import pandas as pd

from alertdrive.evaluation.analyze import per_second_metrics, per_video_table, summarize_platform, video_metrics
from alertdrive.evaluation.platform_eval import binary_label, load_manifests
from alertdrive.evaluation.real_tests import SCENARIOS, write_template


def _trace(key, y, drowsy_from=None, seconds=120, model="SVM"):
    """Traza sintética: 1 fila por segundo; somnolencia desde drowsy_from (None = nunca)."""
    return pd.DataFrame({
        "video_key": key, "subject_id": key[:2], "y": y, "model": model,
        "t": range(seconds), "ready": [t >= 15 for t in range(seconds)], "valid": True,
        "is_drowsy": [drowsy_from is not None and t >= drowsy_from for t in range(seconds)],
        "score": [0.9 if drowsy_from is not None and t >= drowsy_from else 0.1 for t in range(seconds)],
    })


def test_etiquetas_del_manifiesto():
    assert binary_label("0") == 0 and binary_label("alert") == 0
    assert binary_label(10) == 1 and binary_label("10.0") == 1 and binary_label("drowsy") == 1
    assert binary_label("5") == "low" and binary_label("low_vigilant") == "low"
    assert binary_label("???") is None


def test_manifiesto_descarta_baja_vigilancia(tmp_path):
    m = tmp_path / "m.csv"
    pd.DataFrame({"video_path": ["a/0.mp4", "a/5.mp4", "a/10.mp4"], "subject_id": "uta_01",
                  "label": [0, 5, 10], "dataset": "uta"}).to_csv(m, index=False)
    df = load_manifests([str(m)])
    assert df["y"].tolist() == [0, 1]
    assert df["video_key"].is_unique
    assert len(load_manifests([str(m)], low_vigilant="positive")) == 3


def test_latencia_y_alarmas():
    m = video_metrics(_trace("01_10", 1, drowsy_from=40), persistence=3, cooldown_s=8)
    # Primera alarma en t=42 (3 segundos seguidos desde 40) → 27 s después de calibrar
    assert m["first_alarm_s"] == 27
    assert m["n_alarms"] == math.ceil((119 - 42 + 1) / 8)
    assert m["monitored_s"] == 105


def test_video_alerta_sin_falsas_alarmas():
    m = video_metrics(_trace("01_0", 0), persistence=3, cooldown_s=8)
    assert m["n_alarms"] == 0 and math.isnan(m["first_alarm_s"]) and m["frac_alarm"] == 0


def test_resumen_por_modelo():
    traces = pd.concat([_trace("01_10", 1, 40), _trace("01_0", 0), _trace("02_0", 0, drowsy_from=100)])
    s = summarize_platform(per_video_table(traces, [1, 3], cooldown_s=8)).set_index("persistence")
    assert s.loc[3, "detection_rate"] == 1.0
    assert s.loc[3, "alert_videos_without_alarm"] == 0.5
    assert s.loc[1, "false_alarms_per_hour"] > 0
    sec = per_second_metrics(traces).iloc[0]
    assert 0 < sec["recall"] <= 1 and 0 < sec["specificity"] <= 1


def test_plantilla_de_escenarios(tmp_path):
    p = tmp_path / "esc.csv"
    write_template(p)
    df = pd.read_csv(p, encoding="utf-8-sig")
    assert len(df) == len(SCENARIOS)
    assert set(df["alarma_esperada"]) == {"si", "no"}


def test_barrido_de_umbral_reduce_falsas_alarmas():
    from alertdrive.evaluation.analyze import recommend_thresholds, threshold_sweep

    def noisy(key, y, base, seconds=200):
        # Puntaje alto en somnolientos (0.9) y medio en alerta (0.6): el modelo marca ambos (umbral 0.5).
        t = _trace(key, y, drowsy_from=0, seconds=seconds)
        t["score"] = base
        return t

    traces = pd.concat([noisy("01_10", 1, 0.9), noisy("02_10", 1, 0.9),
                        noisy("01_0", 0, 0.6), noisy("02_0", 0, 0.6)])
    sweep = threshold_sweep(traces, persistence=3, cooldown_s=8)
    low = sweep.set_index("threshold").loc[0.6]
    assert low["false_alarms_per_hour"] > 0                  # el umbral bajo alarma en videos alerta
    rec = recommend_thresholds(sweep, min_detection=0.9).iloc[0]
    assert 0.6 < rec["threshold"] <= 0.9                     # el más bajo que elimina las falsas alarmas
    assert rec["detection_rate"] == 1.0 and rec["false_alarms_per_hour"] == 0
