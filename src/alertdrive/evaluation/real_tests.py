"""Pruebas en condiciones reales: cruza los escenarios con las sesiones guardadas.

1. Genera la plantilla de escenarios:

       python -m alertdrive.evaluation.real_tests --template reports/fase5/escenarios.csv

2. Graba cada escenario con la plataforma (página Monitoreo, guardando en el
   historial) y anota en la plantilla el número de sesión que muestra la app.
3. Resume:

       python -m alertdrive.evaluation.real_tests --escenarios reports/fase5/escenarios.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from alertdrive.storage.history import DEFAULT_DB_PATH, HistoryDB

WINDOW_S = 15.0

# (escenario, ¿debe alarmar?, qué hacer después de los 15 s de calibración)
SCENARIOS = [
    ("Alerta, buena luz", "no", "Mirar al frente con normalidad 2 min, parpadeando natural"),
    ("Alerta hablando", "no", "Hablar o cantar 2 min (pone a prueba los falsos bostezos)"),
    ("Alerta mirando a los lados", "no", "Mirar a izquierda y derecha cada 10 s, como revisando espejos, 2 min"),
    ("Alerta con lentes", "no", "Como el primero, con lentes formulados"),
    ("Alerta con poca luz", "no", "Como el primero, con la luz del cuarto apagada y sólo la pantalla"),
    ("Microsueños", "si", "Cerrar los ojos 2 a 3 s, abrir 10 s, repetir durante 1 min"),
    ("Ojos entrecerrados", "si", "Mantener los ojos casi cerrados y parpadeo lento 45 s"),
    ("Bostezos", "si", "Bostezar 3 o 4 veces en 1 min, con la boca bien abierta"),
    ("Cabeceos", "si", "Dejar caer la cabeza hacia adelante y recuperarla, 4 o 5 veces en 1 min"),
    ("Fatiga combinada", "si", "Parpadeo lento, un bostezo y un cabeceo, repetir 1 min"),
    ("Lentes de sol", "no", "Alerta con lentes de sol 1 min (limitación conocida: puede no ver los ojos)"),
    ("Otra persona, alerta", "no", "Un compañero repite el escenario 1"),
    ("Otra persona, microsueños", "si", "Un compañero repite el escenario de microsueños"),
]


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"sesion": "", "escenario": s, "alarma_esperada": e, "instrucciones": i, "notas": ""}
                       for s, e, i in SCENARIOS])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def evaluate(scenarios: pd.DataFrame, db: HistoryDB) -> pd.DataFrame:
    sessions = db.sessions().set_index("id")
    rows = []
    for r in scenarios.itertuples(index=False):
        if pd.isna(r.sesion) or str(r.sesion).strip() == "":
            continue
        sid = int(float(r.sesion))
        if sid not in sessions.index:
            print(f"Aviso: la sesión {sid} ({r.escenario}) no está en la base de datos", file=sys.stderr)
            continue
        s = sessions.loc[sid]
        ev = db.events([sid])
        expected = str(r.alarma_esperada).strip().lower() in {"si", "sí", "s", "1", "true", "yes"}
        n_alarms = int(s["n_alarms"])
        first = float(ev["t_s"].min()) - WINDOW_S if n_alarms else math.nan
        reasons = sorted({x for lst in ev["reasons"] for x in lst}) if not ev.empty else []
        rows.append({
            "Sesión": sid,
            "Escenario": r.escenario,
            "Esperado": "Alarma" if expected else "Sin alarma",
            "Duración (s)": round(float(s["duration_s"])),
            "Alarmas": n_alarms,
            "Primera alarma (s tras calibrar)": round(max(first, 0)) if n_alarms else None,
            "Sin rostro": f"{(s['frac_sin_rostro'] or 0):.0%}",
            "Motivos": ", ".join(reasons),
            "Resultado": "Acierto" if (n_alarms > 0) == expected else
                         ("Falsa alarma" if n_alarms > 0 else "No detectado"),
            "Modelo": s["model"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Primera alarma (s tras calibrar)"] = df["Primera alarma (s tras calibrar)"].astype("Int64")
    return df


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Pruebas en condiciones reales\n\nNo hay escenarios con número de sesión.\n"
    hits = (df["Resultado"] == "Acierto").sum()
    exp_yes = df[df["Esperado"] == "Alarma"]
    exp_no = df[df["Esperado"] == "Sin alarma"]
    lines = [
        "# Pruebas en condiciones reales",
        "",
        f"**Aciertos:** {hits} de {len(df)} escenarios. "
        f"Escenarios de fatiga detectados: {(exp_yes['Alarmas'] > 0).sum()} de {len(exp_yes)}. "
        f"Escenarios alerta sin falsas alarmas: {(exp_no['Alarmas'] == 0).sum()} de {len(exp_no)}.",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))
                                       else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", help="Escribe la plantilla de escenarios en esta ruta y termina")
    ap.add_argument("--escenarios", default="reports/fase5/escenarios.csv")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--out", default="reports/fase5/pruebas_reales.md")
    args = ap.parse_args(argv)

    if args.template:
        write_template(Path(args.template))
        print(f"Plantilla escrita en {args.template}. Llena la columna 'sesion' con el número de cada sesión.")
        return 0
    if not Path(args.escenarios).exists():
        print(f"No existe {args.escenarios}. Créala con --template.", file=sys.stderr)
        return 1
    df = evaluate(pd.read_csv(args.escenarios, encoding="utf-8-sig"), HistoryDB(args.db))
    text = to_markdown(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
