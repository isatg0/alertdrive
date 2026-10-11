"""Recursos compartidos por las páginas de Streamlit."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from alertdrive.classify.predictor import DrowsinessPredictor
from alertdrive.web.engine import load_predictor
from alertdrive.storage.history import DEFAULT_DB_PATH, HistoryDB

# nivel: (fondo, color de texto, título, indicación)
STATUS = {
    "listo": ("#37474F", "#FFFFFF", "Listo para monitorear",
              "Activa «Monitorear». Con webcam, siéntate de frente y con buena luz."),
    "calibrando": ("#1565C0", "#FFFFFF", "Calibrando", "Mira al frente con naturalidad durante 15 segundos."),
    "sin_rostro": ("#546E7A", "#FFFFFF", "No se ve el rostro", "Revisa la cámara, la luz o la posición."),
    "ok": ("#2E7D32", "#FFFFFF", "Conductor alerta", "Sin signos de somnolencia."),
    "atencion": ("#F9A825", "#1A1A1A", "Signos de fatiga", "Se están acumulando indicios. Considera una pausa."),
    "alarma": ("#C62828", "#FFFFFF", "Alerta de somnolencia", "Detente en un lugar seguro y descansa."),
}
# Nombres cortos para gráficas y tablas, en orden de gravedad.
LEVEL_NAMES = {
    "calibrando": "Calibrando",
    "sin_rostro": "Sin rostro",
    "ok": "Alerta",
    "atencion": "Fatiga",
    "alarma": "Alarma",
}
LEVEL_COLORS = {k: STATUS[k][0] for k in LEVEL_NAMES}


def status_html(level: str) -> str:
    bg, fg, title, hint = STATUS[level]
    return (
        f'<div style="background:{bg};color:{fg};padding:1.1rem 1.3rem;border-radius:12px">'
        f'<div style="font-size:1.6rem;font-weight:700;line-height:1.2">{title}</div>'
        f'<div style="font-size:0.95rem;margin-top:0.3rem;opacity:0.92">{hint}</div></div>'
    )


@st.cache_resource(show_spinner="Cargando modelo…")
def get_predictor(path: str) -> DrowsinessPredictor:
    return load_predictor(path)


@st.cache_resource
def get_db(path: str = str(DEFAULT_DB_PATH)) -> HistoryDB:
    """Una sola conexión SQLite compartida por todas las páginas y pestañas."""
    return HistoryDB(Path(path))


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"
