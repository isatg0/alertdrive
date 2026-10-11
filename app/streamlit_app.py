"""AlertDrive IA: punto de entrada de la plataforma web.

Ejecutar desde la raíz del repositorio:

    streamlit run app/streamlit_app.py

Páginas:
- Monitoreo (Fase 3): webcam o video en vivo con alertas.
- Historial (Fase 4): sesiones, alarmas y estadísticas guardadas en SQLite.
- Modelos (Fase 5): métricas de los algoritmos y de la plataforma completa.
"""
import os
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR.parent)  # las rutas del proyecto (models/, data/, reports/) son relativas a la raíz

st.set_page_config(page_title="AlertDrive IA", page_icon="🚗", layout="wide")

pages = st.navigation([
    st.Page(APP_DIR / "monitor.py", title="Monitoreo", icon="🎥", default=True),
    st.Page(APP_DIR / "historial.py", title="Historial", icon="📊"),
    st.Page(APP_DIR / "modelos.py", title="Modelos", icon="🧠"),
])
pages.run()
