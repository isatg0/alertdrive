"""Pruebas del histórico SQLite (no usan cámara ni MediaPipe)."""
import math
import sqlite3
from types import SimpleNamespace

from alertdrive.storage.history import HistoryDB, SessionLogger
from alertdrive.web.alerts import AlertManager


def _pred(drowsy, perclos=0.1):
    return SimpleNamespace(is_drowsy=drowsy, score=0.9 if drowsy else 0.1,
                           features={"perclos": perclos, "blink_rate_per_min": 12.0,
                                     "yawn_count": 0.0, "nod_count": math.nan})


def _simulate(db, seconds=40, drowsy_from=25):
    """Sesión simulada: calibración 15 s, luego alerta y somnolencia desde drowsy_from."""
    alerts = AlertManager(persistence=3, cooldown_s=8)
    logger = SessionLogger(db, "Webcam 0", "webcam", "Random Forest", {"persistence": 3}, flush_every=5)
    for t in range(seconds):
        p = _pred(t >= drowsy_from, perclos=0.4 if t >= drowsy_from else 0.05)
        d = alerts.update(t, ready=t >= 15, valid=True, is_drowsy=p.is_drowsy, reasons=["PERCLOS alto"] if p.is_drowsy else [])
        logger.log(t, d, p)
    logger.close()
    return logger.session_id


def test_sesion_completa(tmp_path):
    db = HistoryDB(tmp_path / "h.db")
    sid = _simulate(db)

    s = db.sessions().set_index("id").loc[sid]
    assert s["status"] == "terminada"
    assert s["duration_s"] == 39
    # Somnolencia desde t=25 con persistencia 3 → alarmas en 27 y 35 (enfriamiento de 8 s)
    assert s["n_alarms"] == 2
    assert math.isclose(s["frac_alarma"], 13 / 40)

    ev = db.events([sid])
    assert ev["t_s"].tolist() == [27.0, 35.0]
    assert ev["reasons"].iloc[0] == ["PERCLOS alto"]
    feats = db.event_features(int(ev["id"].iloc[0]))
    assert feats["perclos"] == 0.4 and feats["nod_count"] is None  # NaN se guarda como nulo

    samples = db.samples(sid)
    assert len(samples) == 40
    assert samples["level"].iloc[0] == "calibrando"


def test_filtro_por_fechas_y_borrado(tmp_path):
    db = HistoryDB(tmp_path / "h.db")
    sid = _simulate(db, seconds=20)
    assert len(db.sessions("2000-01-01", "2000-01-02")) == 0
    assert len(db.sessions()) == 1
    db.delete_session(sid)
    assert db.sessions().empty
    assert db.samples(sid).empty  # se borran en cascada


def test_sesion_cortada_se_marca_interrumpida(tmp_path):
    path = tmp_path / "h.db"
    db = HistoryDB(path)
    sid = db.start_session("Webcam 0", "webcam", "SVM", {})
    with sqlite3.connect(path) as conn:  # simula que la última escritura fue hace una hora
        conn.execute("UPDATE sessions SET updated_at = '2000-01-01T00:00:00' WHERE id = ?", (sid,))
    assert db.mark_stale_sessions(max_idle_s=120) == 1
    assert db.sessions().set_index("id").loc[sid, "status"] == "interrumpida"


def test_sesion_recien_iniciada_no_se_marca(tmp_path):
    db = HistoryDB(tmp_path / "h.db")
    db.start_session("Webcam 0", "webcam", "SVM", {})
    assert db.mark_stale_sessions(max_idle_s=120) == 0
