"""Página de métricas de los modelos (Fase 2) y de la plataforma (Fase 5)."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

FASE2 = Path("reports/fase2")
FASE5 = Path("reports/fase5")


def show_markdown(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    # Baja dos niveles los títulos para que no compitan con el de la página.
    text = re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", text, flags=re.M)
    st.markdown(text)
    return True


def show_images(items: list[tuple[Path, str]]) -> None:
    present = [(p, c) for p, c in items if p.exists()]
    for col, (p, caption) in zip(st.columns(len(present)) if present else [], present):
        col.image(str(p), caption=caption, width="stretch")


st.title("Modelos")
st.caption("Cómo se evaluaron los algoritmos y cómo se comporta la plataforma completa.")

tab2, tab5, tab_real = st.tabs(["Comparación de algoritmos", "Plataforma en videos", "Pruebas reales"])

with tab2:
    if show_markdown(FASE2 / "resumen.md"):
        show_images([(FASE2 / "roc.png", "Curvas ROC"), (FASE2 / "confusion.png", "Matrices de confusión"),
                     (FASE2 / "importancia_rf.png", "Importancia de características (Random Forest)")])
    else:
        st.info("No hay resultados de la Fase 2. Corre `python -m alertdrive.classify.train ...`.")

with tab5:
    if show_markdown(FASE5 / "resumen.md"):
        show_images([(FASE5 / "persistencia.png", "Detección, latencia y falsas alarmas según la persistencia"),
                     (FASE5 / "umbral.png", "Compromiso entre detección y falsas alarmas según el umbral")])
    else:
        st.info("Aún no hay evaluación de la plataforma. Corre `alertdrive.evaluation.platform_eval` y "
                "después `alertdrive.evaluation.analyze` (ver FASE5.md).")

with tab_real:
    if not show_markdown(FASE5 / "pruebas_reales.md"):
        st.info("Aún no hay pruebas reales resumidas. Corre `alertdrive.evaluation.real_tests` (ver FASE5.md).")
