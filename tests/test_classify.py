"""Pruebas de la Fase 2: etiquetas, reglas, partición por sujeto, reporte y predictor."""
import math

import numpy as np
import pandas as pd
import pytest

from alertdrive.classify.dataset import build_target, load_features, map_label, subsample_windows
from alertdrive.classify.evaluate import _outer_splitter, bootstrap_f1_difference, cross_validate
from alertdrive.classify.predictor import DrowsinessPredictor
from alertdrive.classify.rules import MAX_SCORE, RuleBasedClassifier
from alertdrive.classify.synthetic import make_synthetic_features
from alertdrive.classify.train import main as train_main
from alertdrive.features.window import FEATURE_COLUMNS

CALM = {c: 0.0 for c in FEATURE_COLUMNS} | {"perclos": 0.04, "blink_dur_mean_s": 0.15, "valid": 1.0}
DROWSY = CALM | {"perclos": 0.35, "microsleep_count": 2, "yawn_count": 1, "nod_count": 1}


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("feat") / "syn.csv"
    make_synthetic_features(n_subjects=10, windows_per_video=24, seed=1).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ etiquetas
def test_label_mapping_and_low_vigilant_policy():
    assert map_label("drowsy") == 1 and map_label("Yawning") == 1
    assert map_label("alert") == 0 and map_label("talking") == 0
    assert math.isnan(map_label("low_vigilant"))
    assert map_label("low_vigilant", "positive") == 1
    assert math.isnan(map_label("desconocida"))


def test_frame_labels_override_video_label():
    df = pd.DataFrame({"label": ["alert", "alert", "drowsy"], "label_fraction": [0.8, np.nan, 0.1]})
    assert build_target(df).tolist() == [1.0, 0.0, 0.0]


def test_load_drops_invalid_and_intermediate(synthetic_csv):
    raw = pd.read_csv(synthetic_csv)
    data = load_features([synthetic_csv])
    expected = raw[(raw.valid >= 1) & (raw.label != "low_vigilant")]
    assert len(data) == len(expected)
    assert list(data.X.columns) == list(FEATURE_COLUMNS)
    assert set(np.unique(data.y)) == {0, 1}


def test_load_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"subject_id": ["a"], "label": ["alert"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_features([p])


def test_subsample_keeps_one_of_n_per_video(synthetic_csv):
    data = load_features([synthetic_csv])
    idx = subsample_windows(data, np.arange(len(data)), 3)
    per_video = data.meta.iloc[idx].groupby("video_path").size()
    full = data.meta.groupby("video_path").size()
    assert (per_video == np.ceil(full / 3)).all()


# ------------------------------------------------------------------ reglas
def test_rule_scores_and_explanations():
    X = pd.DataFrame([CALM, DROWSY])[list(FEATURE_COLUMNS)]
    scores = RuleBasedClassifier.score_frame(X)
    assert scores[0] == 0
    assert scores[1] == 2 + 1 + 3 + 1 + 2  # PERCLOS≥.15, ≥.30, microsueño, bostezo, cabeceo
    reasons = RuleBasedClassifier.explain(DROWSY)
    assert any("Microsueño" in r for r in reasons) and any("Cabeceo" in r for r in reasons)


def test_rules_ignore_nan():
    X = pd.DataFrame([{c: np.nan for c in FEATURE_COLUMNS}])
    assert RuleBasedClassifier.score_frame(X)[0] == 0


def test_rules_threshold_is_learned_only_from_training_data():
    X = pd.DataFrame([CALM, CALM | {"yawn_count": 1}, DROWSY, DROWSY])[list(FEATURE_COLUMNS)]
    clf = RuleBasedClassifier().fit(X, np.array([0, 0, 1, 1]))
    assert clf.min_score_ == 2  # 1 punto (un bostezo aislado) no basta para alertar
    assert clf.predict(X).tolist() == [0, 0, 1, 1]
    assert 0 <= clf.decision_function(X).max() <= 1 and MAX_SCORE > 0


# ------------------------------------------------------------------ evaluación
def test_outer_folds_never_share_subjects(synthetic_csv):
    data = load_features([synthetic_csv])
    for train, test in _outer_splitter(data, 5, 0):
        assert not set(data.groups[train]) & set(data.groups[test])


def test_cross_validation_produces_out_of_fold_predictions(synthetic_csv):
    data = load_features([synthetic_csv])
    results, fold_of = cross_validate(data, ["rules", "rf"], n_splits=3, fast=True, n_jobs=1)
    assert (fold_of >= 0).all()
    for res in results.values():
        assert len(res.fold_metrics) == 3
        assert res.summary()["f1"]["mean"] > 0.6  # los datos sintéticos son separables
    assert results["rf"].importance is not None


def test_bootstrap_identical_predictions_give_zero_difference(synthetic_csv):
    data = load_features([synthetic_csv])
    pred = (data.X["perclos"] > 0.15).astype(int).to_numpy()
    out = bootstrap_f1_difference(data, pred, pred, n_boot=200)
    assert out["delta_f1"] == 0 and out["ci95_low"] == 0 and out["ci95_high"] == 0


# ------------------------------------------------------------------ extremo a extremo
def test_train_cli_writes_report_and_usable_models(synthetic_csv, tmp_path):
    out = train_main(["--features", str(synthetic_csv), "--models", "rules", "rf", "--folds", "3", "--fast",
                      "--subsample", "2", "--jobs", "1",
                      "--out-dir", str(tmp_path / "rep"), "--models-dir", str(tmp_path / "mod")])
    for name in ("resumen.md", "metricas.json", "roc.png", "confusion.png", "importancia_rf.png"):
        assert (tmp_path / "rep" / name).exists()
    assert (tmp_path / "mod" / "drowsiness_best.joblib").exists()
    assert out["best"] in {"rules", "rf"}

    # Ventanas típicas (mediana de cada clase) con las claves extra que entrega la Fase 1.
    data = out["data"]
    typical = {k: data.X[data.y == k].median().to_dict() | {"valid": 1.0, "t_start": 0.0, "t_end": 15.0}
               for k in (0, 1)}
    for name in ("rules", "rf"):
        predictor = DrowsinessPredictor.load(tmp_path / "mod" / f"drowsiness_{name}.joblib")
        assert predictor.predict(typical[1]).is_drowsy
        assert not predictor.predict(typical[0]).is_drowsy
        hidden = predictor.predict(typical[1] | {"valid": 0.0})
        assert not hidden.is_drowsy and not hidden.valid


def test_predictor_rejects_mismatched_columns(tmp_path):
    import joblib
    bundle = {"feature_columns": ["otra"], "model": None, "threshold": 0.5, "name": "x"}
    joblib.dump(bundle, tmp_path / "m.joblib")
    with pytest.raises(ValueError, match="no coinciden"):
        DrowsinessPredictor.load(tmp_path / "m.joblib")
