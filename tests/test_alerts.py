import wave
import io

import pytest

from alertdrive.web.alerts import AlertManager, make_beep_wav


def test_alarma_requiere_persistencia_y_respeta_enfriamiento():
    am = AlertManager(persistence=3, cooldown_s=5)
    decisions = [am.update(t, ready=True, valid=True, is_drowsy=True) for t in range(3)]
    assert [d.level for d in decisions] == ["atencion", "atencion", "alarma"]
    assert decisions[2].new_alarm
    assert not am.update(3, True, True, True).new_alarm   # dentro del enfriamiento
    assert am.update(7, True, True, True).new_alarm       # ya pasaron 5 s


def test_una_prediccion_alerta_reinicia_la_racha():
    am = AlertManager(persistence=2)
    am.update(0, True, True, True)
    assert am.update(1, True, True, False).level == "ok"
    assert am.update(2, True, True, True).level == "atencion"


def test_calibracion_y_rostro_ausente():
    am = AlertManager(persistence=1)
    assert am.update(0, ready=False, valid=True, is_drowsy=True).level == "calibrando"
    assert am.update(1, ready=True, valid=False, is_drowsy=True).level == "sin_rostro"
    assert am.streak == 0


def test_motivos_se_propagan():
    am = AlertManager(persistence=1)
    d = am.update(0, True, True, True, ["Microsueño"])
    assert d.reasons == ("Microsueño",)


def test_persistencia_invalida():
    with pytest.raises(ValueError):
        AlertManager(persistence=0)


def test_wav_valido():
    data = make_beep_wav(repeats=2)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getnframes() > 0
