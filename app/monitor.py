"""Página de monitoreo en vivo (Fase 3) con registro en el histórico (Fase 4)."""
from __future__ import annotations

import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from alertdrive.pipeline import video_fps
from alertdrive.storage.history import SessionLogger
from alertdrive.web.catalog import available_models
from alertdrive.web.alerts import AlertManager, make_beep_wav, play_system_beep
from alertdrive.web.engine import DrowsinessEngine
from alertdrive.web.overlay import draw_overlay
from alertdrive.web.ui import get_db, get_predictor, status_html

SAMPLE_FPS = 5.0  # debe coincidir con --sample-fps de build_features
SPEEDS = {"Tiempo real": 1.0, "2x": 2.0, "4x": 4.0, "Lo más rápido posible": 0.0}


def fmt(v: object, nd: int = 2) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "--"
    return "--" if f != f else f"{f:.{nd}f}"


ss = st.session_state
ss.setdefault("events", [])
ss.setdefault("beep_count", 0)
ss.setdefault("last_session_id", None)

# ---------------- Barra lateral ----------------
available = available_models()
with st.sidebar:
    st.header("Configuración")
    if not available:
        st.error("No hay modelos en models/. Ejecuta primero el entrenamiento de la Fase 2.")
        st.stop()
    model_name = st.selectbox("Modelo", list(available))

    source = st.radio("Fuente de video", ["Webcam", "Archivo de video"])
    cam_index, upload, speed = 0, None, 1.0
    if source == "Webcam":
        cam_index = int(st.number_input("Número de cámara", min_value=0, max_value=5, value=0,
                                        help="0 es la cámara integrada; prueba 1 si usas una externa."))
    else:
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        speed_name = st.selectbox("Velocidad de reproducción", list(SPEEDS))
        speed = SPEEDS[speed_name]

    st.subheader("Alertas")
    persistence = st.slider("Segundos seguidos con fatiga para alarmar", 1, 10, 3,
                            help="El modelo evalúa cada segundo la ventana de los últimos 15 s.")
    cooldown = st.slider("Segundos mínimos entre alarmas", 2, 30, 8)
    threshold = st.number_input(
        "Umbral de puntaje (opcional)", min_value=0.0, value=None, step=0.05, format="%.2f",
        placeholder="El del modelo",
        help="Puntaje mínimo para contar un segundo como fatiga. Vacío: decide el modelo. "
             "Más alto: menos falsas alarmas, pero más tarde. El valor recomendado para cada "
             "modelo está en reports/fase5/resumen.md, sección «Umbral de puntaje».")
    sound = st.checkbox("Alarma sonora", value=True)
    save = st.checkbox("Guardar la sesión en el historial", value=True)

# ---------------- Encabezado y controles ----------------
st.title("AlertDrive IA")
st.caption("Monitoreo de somnolencia del conductor por cámara. Evalúa cada segundo los últimos 15 s.")
run = st.toggle("Monitorear", value=False)

col_video, col_panel = st.columns([3, 2], gap="large")
video_ph = col_video.empty()
with col_panel:
    status_ph = st.empty()
    st.write("")
    row1, row2 = st.columns(3), st.columns(3)
    metric_ph = [c.empty() for c in (*row1, *row2)]
    reasons_ph = st.empty()
    chart_ph = st.empty()
audio_ph = st.empty()

st.subheader("Alertas de esta sesión")
events_ph = st.empty()


def render_events(final: bool = False) -> None:
    if not ss.events:
        events_ph.info("Todavía no hay alertas en esta sesión.")
        return
    df = pd.DataFrame(ss.events)
    events_ph.dataframe(df, hide_index=True, width="stretch")
    if final:
        st.download_button("Descargar alertas (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="alertas_alertdrive.csv", mime="text/csv")


def render_metrics(feats: dict, ear_now: float, score: float) -> None:
    values = [
        ("PERCLOS (15 s)", fmt(feats.get("perclos"))),
        ("Parpadeos por minuto", fmt(feats.get("blink_rate_per_min"), 0)),
        ("Bostezos (15 s)", fmt(feats.get("yawn_count"), 0)),
        ("EAR actual", fmt(ear_now)),
        ("Cabeceos (15 s)", fmt(feats.get("nod_count"), 0)),
        ("Puntaje del modelo", fmt(score)),
    ]
    for ph, (label, value) in zip(metric_ph, values):
        ph.metric(label, value)


# ---------------- Modo detenido ----------------
if not run:
    status_ph.markdown(status_html("listo"), unsafe_allow_html=True)
    render_events(final=True)
    if ss.last_session_id is not None:
        st.caption(f"La última sesión quedó guardada con el número {ss.last_session_id}. "
                   "Consúltala en la página Historial.")
    st.stop()

# ---------------- Abrir la fuente ----------------
is_file = source == "Archivo de video"
if is_file:
    if upload is None:
        st.warning("Sube un video en la barra lateral.")
        st.stop()
    tmp_path = Path(tempfile.gettempdir()) / f"alertdrive_{Path(upload.name).name}"
    tmp_path.write_bytes(upload.getvalue())
    cap = cv2.VideoCapture(str(tmp_path))
    source_label = upload.name
else:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_index, backend)
    source_label = f"Webcam {cam_index}"

if not cap.isOpened():
    st.error("No se pudo abrir la fuente. Si es la webcam, ciérrala en otras apps "
             "(Teams, Zoom, live_preview.py) o prueba otro número de cámara.")
    st.stop()

# ---------------- Bucle principal ----------------
ss.events = []  # la tabla de abajo muestra sólo la sesión actual
alerts = AlertManager(persistence=persistence, cooldown_s=float(cooldown))
logger = None
if save:
    logger = SessionLogger(
        get_db(), source_label, "video" if is_file else "webcam", model_name,
        {"persistence": persistence, "cooldown_s": cooldown, "sample_fps": SAMPLE_FPS,
         "speed": speed if is_file else 1.0, "threshold": threshold},
    )
    ss.last_session_id = logger.session_id

fps = video_fps(cap)
step = max(1, round(fps / SAMPLE_FPS))
scores: deque[tuple[float, float]] = deque(maxlen=180)  # últimos 3 minutos
level = "calibrando"
idx, last_t = -1, -1e9
wall0 = time.monotonic()
end_status = "terminada"
render_events()

engine = DrowsinessEngine(predictor=get_predictor(available[model_name]), threshold=threshold)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                st.success("Fin del video. Desactiva «Monitorear» para ver el resumen.")
            else:
                st.error("La cámara dejó de enviar imágenes.")
                end_status = "interrumpida"
            break
        idx += 1

        if is_file:  # tiempo del video, muestreado como en el entrenamiento
            if idx % step:
                continue
            t = idx / fps
            if speed > 0:
                ahead = t / speed - (time.monotonic() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
        else:  # reloj real; se leen todos los frames para no acumular retraso
            t = time.monotonic() - wall0
            if t - last_t < 1.0 / SAMPLE_FPS:
                continue
        last_t = t

        out = engine.process(frame, t)
        p = out.prediction
        if out.new_prediction:
            decision = alerts.update(t, p.ready, p.valid, p.is_drowsy, p.reasons)
            level = decision.level
            if logger is not None:
                logger.log(t, decision, p)

            status_ph.markdown(status_html(level), unsafe_allow_html=True)
            render_metrics(p.features, out.frame_features.ear, p.score)
            reasons_ph.markdown("**Motivos:** " + (", ".join(p.reasons) if p.is_drowsy and p.reasons else "ninguno"))
            if p.ready and p.valid:
                scores.append((round(t, 1), p.score))
                chart_ph.line_chart(pd.DataFrame(scores, columns=["t (s)", "Puntaje"]).set_index("t (s)"),
                                    height=160)

            if decision.new_alarm:
                ss.events.append({
                    "Hora": time.strftime("%H:%M:%S"),
                    "Fuente": source_label,
                    "Segundo": round(t, 1),
                    "Modelo": model_name,
                    "Puntaje": round(p.score, 3),
                    "PERCLOS": round(float(p.features.get("perclos", float("nan"))), 3),
                    "Motivos": ", ".join(decision.reasons),
                })
                render_events()
                if sound and not play_system_beep():
                    ss.beep_count += 1  # el WAV cambia un poco para que el navegador lo vuelva a sonar
                    audio_ph.audio(make_beep_wav(repeats=3 + ss.beep_count % 2), format="audio/wav", autoplay=True)

        shown = draw_overlay(out.frame, out.obs, out.frame_features, level)
        video_ph.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
finally:
    cap.release()
    engine.close()
    if logger is not None:
        logger.close(end_status)
