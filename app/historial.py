"""Página de historial y dashboard (Fase 4)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from alertdrive.web.ui import LEVEL_COLORS, LEVEL_NAMES, fmt_duration, get_db

db = get_db()
db.mark_stale_sessions()

st.title("Historial")
st.caption("Sesiones de monitoreo guardadas, alarmas y estadísticas. "
           "Cada sesión registra una muestra por segundo y cada alarma con sus motivos.")

# ---------------- Filtros ----------------
with st.sidebar:
    st.header("Filtros")
    today = date.today()
    period = st.date_input("Periodo", value=(today - timedelta(days=30), today), max_value=today,
                           format="DD/MM/YYYY")
    d_from, d_to = (period if isinstance(period, tuple) and len(period) == 2 else (period, period))

sessions = db.sessions(str(d_from), str(d_to))
if sessions.empty:
    st.info("No hay sesiones guardadas en este periodo. Monitorea desde la página Monitoreo "
            "con «Guardar la sesión en el historial» activado, o corre "
            "`python scripts\\monitor_video.py --video ... --save`.")
    st.stop()

with st.sidebar:
    kinds = st.multiselect("Fuente", ["webcam", "video"], default=["webcam", "video"],
                           format_func={"webcam": "Webcam", "video": "Archivo de video"}.get)
    models = sorted(sessions["model"].unique())
    model_sel = st.multiselect("Modelo", models, default=models)
    min_minutes = st.number_input("Duración mínima (min)", min_value=0.0, value=0.0, step=0.5,
                                  help="Descarta pruebas muy cortas.")

sessions = sessions[sessions["source_kind"].isin(kinds) & sessions["model"].isin(model_sel)
                    & (sessions["duration_s"] >= min_minutes * 60)]
if sessions.empty:
    st.info("Ninguna sesión cumple los filtros.")
    st.stop()
events = db.events(sessions["id"].tolist())

# ---------------- Indicadores generales ----------------
total_s = float(sessions["duration_s"].sum())
alarm_time = float((sessions["frac_alarma"].fillna(0) * sessions["duration_s"]).sum())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sesiones", len(sessions))
k2.metric("Tiempo monitoreado", fmt_duration(total_s))
k3.metric("Alarmas", int(sessions["n_alarms"].sum()))
k4.metric("Alarmas por hora", f"{sessions['n_alarms'].sum() / max(total_s / 3600, 1 / 60):.1f}")
k5.metric("Tiempo en alarma", f"{alarm_time / total_s:.1%}" if total_s else "--")

# ---------------- Gráficas generales ----------------
level_domain = list(LEVEL_NAMES.values())
level_range = [LEVEL_COLORS[k] for k in LEVEL_NAMES]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Alarmas por día**")
    if events.empty:
        st.caption("Sin alarmas en el periodo.")
    else:
        per_day = events.assign(dia=events["occurred_at"].dt.normalize()).groupby("dia").size()
        per_day = per_day.reset_index(name="alarmas")
        st.altair_chart(
            alt.Chart(per_day).mark_bar(color=LEVEL_COLORS["alarma"]).encode(
                x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m")),
                y=alt.Y("alarmas:Q", title="Alarmas"),
                tooltip=[alt.Tooltip("dia:T", title="Día", format="%d/%m/%Y"), "alarmas:Q"],
            ).properties(height=220),
            width="stretch",
        )
with c2:
    st.markdown("**Alarmas por hora del día**")
    if events.empty:
        st.caption("Sin alarmas en el periodo.")
    else:
        per_hour = (events["occurred_at"].dt.hour.value_counts()
                    .reindex(range(24), fill_value=0).rename_axis("hora").reset_index(name="alarmas"))
        st.altair_chart(
            alt.Chart(per_hour).mark_bar(color=LEVEL_COLORS["alarma"]).encode(
                x=alt.X("hora:O", title="Hora"),
                y=alt.Y("alarmas:Q", title="Alarmas"),
                tooltip=["hora:O", "alarmas:Q"],
            ).properties(height=220),
            width="stretch",
        )

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Motivos más frecuentes en las alarmas**")
    reasons = events["reasons"].explode().dropna() if not events.empty else pd.Series(dtype=str)
    if reasons.empty:
        st.caption("Sin motivos registrados.")
    else:
        top = reasons.value_counts().head(10).rename_axis("motivo").reset_index(name="veces")
        st.altair_chart(
            alt.Chart(top).mark_bar(color="#455A64").encode(
                x=alt.X("veces:Q", title="Alarmas en que aparece"),
                y=alt.Y("motivo:N", sort="-x", title=None),
                tooltip=["motivo:N", "veces:Q"],
            ).properties(height=220),
            width="stretch",
        )
with c4:
    st.markdown("**Tiempo en cada estado**")
    shares = {
        "Alerta": (sessions["frac_ok"].fillna(0) * sessions["duration_s"]).sum(),
        "Fatiga": (sessions["frac_atencion"].fillna(0) * sessions["duration_s"]).sum(),
        "Alarma": alarm_time,
        "Sin rostro": (sessions["frac_sin_rostro"].fillna(0) * sessions["duration_s"]).sum(),
    }
    used = sum(shares.values())
    shares["Calibrando"] = max(total_s - used, 0)
    share_df = pd.DataFrame({"estado": list(shares), "segundos": list(shares.values())})
    share_df["porcentaje"] = share_df["segundos"] / max(total_s, 1e-9)
    st.altair_chart(
        alt.Chart(share_df).mark_bar().encode(
            x=alt.X("porcentaje:Q", title="Porcentaje del tiempo", axis=alt.Axis(format="%")),
            y=alt.Y("estado:N", sort=level_domain[::-1], title=None),
            color=alt.Color("estado:N", scale=alt.Scale(domain=level_domain, range=level_range), legend=None),
            tooltip=["estado:N", alt.Tooltip("porcentaje:Q", format=".1%")],
        ).properties(height=220),
        width="stretch",
    )

# ---------------- Tabla de sesiones ----------------
st.subheader("Sesiones")
table = pd.DataFrame({
    "N.º": sessions["id"],
    "Inicio": sessions["started_at"].dt.strftime("%d/%m/%Y %H:%M"),
    "Fuente": sessions["source"],
    "Modelo": sessions["model"],
    "Duración": sessions["duration_s"].map(fmt_duration),
    "Alarmas": sessions["n_alarms"],
    "Alarmas por hora": sessions["alarms_per_hour"].round(1),
    "En alarma": (sessions["frac_alarma"].fillna(0) * 100).round(1).astype(str) + " %",
    "PERCLOS medio": sessions["perclos_mean"].round(3),
    "Estado": sessions["status"].str.capitalize(),
})
selection = st.dataframe(table, hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="sessions_table")
rows = selection.selection.rows if selection else []
st.caption("Selecciona una fila para ver la sesión en detalle.")

d1, d2 = st.columns(2)
d1.download_button("Descargar sesiones (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                   file_name="sesiones_alertdrive.csv", mime="text/csv")
if not events.empty:
    ev_export = events.assign(reasons=events["reasons"].map(", ".join))
    d2.download_button("Descargar alarmas (CSV)", ev_export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="alarmas_alertdrive.csv", mime="text/csv")

if not rows:
    st.stop()

# ---------------- Detalle de una sesión ----------------
sess = sessions.iloc[rows[0]]
sid = int(sess["id"])
params = json.loads(sess["params"] or "{}")
st.divider()
st.subheader(f"Sesión {sid}: {sess['source']}")
st.caption(f"Inicio {sess['started_at']:%d/%m/%Y %H:%M}, modelo {sess['model']}, "
           f"alarma tras {params.get('persistence', '?')} s seguidos de fatiga, "
           f"mínimo {params.get('cooldown_s', '?')} s entre alarmas, "
           f"umbral {params['threshold'] if params.get('threshold') is not None else 'del modelo'}")

samples = db.samples(sid)
sess_events = events[events["session_id"] == sid]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Duración", fmt_duration(sess["duration_s"]))
m2.metric("Alarmas", int(sess["n_alarms"]))
m3.metric("Tiempo en alarma", f"{(sess['frac_alarma'] or 0):.1%}")
m4.metric("PERCLOS medio", f"{sess['perclos_mean']:.3f}" if pd.notna(sess["perclos_mean"]) else "--")

if samples.empty:
    st.info("Esta sesión no tiene muestras (se detuvo antes del primer segundo).")
else:
    samples = samples.sort_values("t_s")
    samples["estado"] = samples["level"].map(LEVEL_NAMES)
    # Tramos continuos con el mismo estado, para pintar el fondo de la línea de tiempo.
    samples["t_end"] = samples["t_s"].shift(-1).fillna(samples["t_s"] + 1)
    block = (samples["level"] != samples["level"].shift()).cumsum()
    segments = samples.groupby(block).agg(inicio=("t_s", "first"), fin=("t_end", "last"),
                                          estado=("estado", "first")).reset_index(drop=True)

    base = alt.Chart(samples).encode(x=alt.X("t_s:Q", title="Segundo de la sesión"))
    bands = alt.Chart(segments).mark_rect(opacity=0.22).encode(
        x="inicio:Q", x2="fin:Q",
        color=alt.Color("estado:N", title="Estado", scale=alt.Scale(domain=level_domain, range=level_range)),
        tooltip=["estado:N", alt.Tooltip("inicio:Q", title="Desde (s)"), alt.Tooltip("fin:Q", title="Hasta (s)")],
    )
    score_line = base.mark_line(color="#263238").encode(
        y=alt.Y("score:Q", title="Puntaje del modelo"),
        tooltip=[alt.Tooltip("t_s:Q", title="Segundo"), alt.Tooltip("score:Q", format=".2f"), "estado:N"],
    )
    layers = [bands, score_line]
    if not sess_events.empty:
        layers.append(alt.Chart(sess_events).mark_rule(color=LEVEL_COLORS["alarma"], strokeDash=[4, 3]).encode(
            x="t_s:Q", tooltip=[alt.Tooltip("t_s:Q", title="Alarma en el segundo")]))
    st.markdown("**Línea de tiempo**: fondo según el estado, línea del puntaje y alarmas punteadas")
    st.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")

    st.markdown("**PERCLOS de la ventana (15 s)**")
    st.altair_chart(
        base.mark_area(color="#1565C0", opacity=0.5, line=True).encode(
            y=alt.Y("perclos:Q", title="PERCLOS"),
            tooltip=[alt.Tooltip("t_s:Q", title="Segundo"), alt.Tooltip("perclos:Q", format=".2f")],
        ).properties(height=160),
        width="stretch",
    )

if not sess_events.empty:
    st.markdown("**Alarmas de la sesión**")
    ev_table = pd.DataFrame({
        "N.º": sess_events["id"],
        "Hora": sess_events["occurred_at"].dt.strftime("%H:%M:%S"),
        "Segundo": sess_events["t_s"].round(1),
        "Puntaje": sess_events["score"].round(3),
        "PERCLOS": sess_events["perclos"].round(3),
        "Motivos": sess_events["reasons"].map(", ".join),
    })
    st.dataframe(ev_table, hide_index=True, width="stretch")
    with st.expander("Características completas de una alarma"):
        ev_id = st.selectbox("Alarma", ev_table["N.º"].tolist(),
                             format_func=lambda i: f"N.º {i} (segundo {ev_table.set_index('N.º').loc[i, 'Segundo']})")
        feats = db.event_features(int(ev_id))
        st.dataframe(pd.DataFrame({"Característica": list(feats), "Valor": list(feats.values())}),
                     hide_index=True, width="stretch")

with st.expander("Borrar esta sesión"):
    st.write("Se borran la sesión, sus muestras y sus alarmas. No se puede deshacer.")
    if st.checkbox("Entiendo, quiero borrarla", key=f"confirm_{sid}"):
        if st.button("Borrar sesión", type="primary"):
            db.delete_session(sid)
            st.rerun()
