"""Histórico de monitoreo en SQLite.

Tres tablas:

- ``sessions``: una fila por sesión de monitoreo (webcam o video).
- ``samples``: una fila por predicción (una por segundo): estado, puntaje e
  indicadores principales. Permite reconstruir la línea de tiempo de la sesión
  y calcular cuánto tiempo pasó el conductor en cada estado.
- ``events``: una fila por alarma, con sus motivos y el vector completo de
  características de la ventana que la disparó.

Se usa sólo la biblioteca estándar (``sqlite3``). Las consultas devuelven
``pandas.DataFrame`` para alimentar directamente el dashboard.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path("data/alertdrive.db")
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,              -- ISO 8601, hora local
    updated_at  TEXT    NOT NULL,              -- última escritura (detecta sesiones cortadas)
    ended_at    TEXT,
    source      TEXT    NOT NULL,              -- 'Webcam 0' o nombre del video
    source_kind TEXT    NOT NULL,              -- 'webcam' | 'video'
    model       TEXT    NOT NULL,
    params      TEXT    NOT NULL DEFAULT '{}', -- JSON con los parámetros de alerta
    duration_s  REAL    NOT NULL DEFAULT 0,
    n_alarms    INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'activa'  -- activa | terminada | interrumpida
);

CREATE TABLE IF NOT EXISTS samples (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_s         REAL    NOT NULL,              -- segundo dentro de la sesión
    level       TEXT    NOT NULL,              -- calibrando | sin_rostro | ok | atencion | alarma
    is_drowsy   INTEGER NOT NULL,
    score       REAL,
    perclos     REAL,
    blink_rate  REAL,
    yawn_count  REAL,
    nod_count   REAL,
    PRIMARY KEY (session_id, t_s)
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    occurred_at TEXT    NOT NULL,              -- ISO 8601, hora local
    t_s         REAL    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'alarma',
    score       REAL,
    perclos     REAL,
    reasons     TEXT    NOT NULL DEFAULT '[]', -- JSON: lista de reglas activadas
    features    TEXT                           -- JSON: características de la ventana
);

CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_time     ON events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_sessions_start  ON sessions(started_at);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean(v: Any) -> float | None:
    """Convierte a float apto para SQLite/JSON (NaN e infinitos → None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _features_json(features: Mapping[str, Any] | None) -> str | None:
    if not features:
        return None
    return json.dumps({k: _clean(v) for k, v in features.items()}, ensure_ascii=False)


class HistoryDB:
    """Acceso al archivo SQLite. Seguro para usar desde varios hilos de Streamlit."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    # ------------------------------------------------------------ escritura
    def _write(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def start_session(self, source: str, source_kind: str, model: str,
                      params: Mapping[str, Any] | None = None) -> int:
        now = _now()
        cur = self._write(
            "INSERT INTO sessions (started_at, updated_at, source, source_kind, model, params) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, now, source, source_kind, model, json.dumps(dict(params or {}), ensure_ascii=False)),
        )
        return int(cur.lastrowid)

    def add_samples(self, session_id: int, rows: list[tuple]) -> None:
        """Inserta varias muestras de una vez: (t_s, level, is_drowsy, score, perclos, blink, yawn, nod)."""
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO samples (session_id, t_s, level, is_drowsy, score, perclos, "
                "blink_rate, yawn_count, nod_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(session_id, *r) for r in rows],
            )
            self._conn.execute("UPDATE sessions SET updated_at = ?, duration_s = MAX(duration_s, ?) "
                               "WHERE id = ?", (_now(), max(r[0] for r in rows), session_id))
            self._conn.commit()

    def add_event(self, session_id: int, t_s: float, score: float | None, reasons: Iterable[str],
                  features: Mapping[str, Any] | None = None, kind: str = "alarma") -> int:
        features = features or {}
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (session_id, occurred_at, t_s, kind, score, perclos, reasons, features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, _now(), float(t_s), kind, _clean(score), _clean(features.get("perclos")),
                 json.dumps(list(reasons), ensure_ascii=False), _features_json(features)),
            )
            self._conn.execute("UPDATE sessions SET n_alarms = n_alarms + 1, updated_at = ? WHERE id = ?",
                               (_now(), session_id))
            self._conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int, duration_s: float | None = None, status: str = "terminada") -> None:
        self._write(
            "UPDATE sessions SET ended_at = ?, updated_at = ?, status = ?, "
            "duration_s = COALESCE(?, duration_s) WHERE id = ?",
            (_now(), _now(), status, _clean(duration_s), session_id),
        )

    def mark_stale_sessions(self, max_idle_s: float = 120.0) -> int:
        """Marca como 'interrumpida' toda sesión activa sin escrituras recientes.

        Pasa si se cierra la terminal o el navegador sin detener el monitoreo.
        """
        limit = (datetime.now() - timedelta(seconds=max_idle_s)).isoformat(timespec="seconds")
        cur = self._write(
            "UPDATE sessions SET status = 'interrumpida', ended_at = updated_at "
            "WHERE status = 'activa' AND updated_at < ?", (limit,))
        return cur.rowcount

    def delete_session(self, session_id: int) -> None:
        self._write("DELETE FROM sessions WHERE id = ?", (session_id,))

    def delete_all(self) -> None:
        with self._lock:
            self._conn.executescript("DELETE FROM events; DELETE FROM samples; DELETE FROM sessions;")
            self._conn.commit()

    # ------------------------------------------------------------- lectura
    def _query(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=tuple(params))

    def sessions(self, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
        """Sesiones con su resumen. Fechas en formato 'AAAA-MM-DD' (incluidas)."""
        where, params = self._date_filter("s.started_at", date_from, date_to)
        df = self._query(
            f"""
            SELECT s.id, s.started_at, s.ended_at, s.source, s.source_kind, s.model, s.params,
                   s.duration_s, s.n_alarms, s.status,
                   AVG(CASE WHEN p.level = 'alarma'   THEN 1.0 ELSE 0 END) AS frac_alarma,
                   AVG(CASE WHEN p.level = 'atencion' THEN 1.0 ELSE 0 END) AS frac_atencion,
                   AVG(CASE WHEN p.level = 'ok'       THEN 1.0 ELSE 0 END) AS frac_ok,
                   AVG(CASE WHEN p.level = 'sin_rostro' THEN 1.0 ELSE 0 END) AS frac_sin_rostro,
                   AVG(p.perclos) AS perclos_mean
            FROM sessions s LEFT JOIN samples p ON p.session_id = s.id
            {where}
            GROUP BY s.id ORDER BY s.started_at DESC
            """, params)
        if not df.empty:
            df["started_at"] = pd.to_datetime(df["started_at"])
            df["alarms_per_hour"] = df["n_alarms"] / (df["duration_s"].clip(lower=60) / 3600)
        return df

    def events(self, session_ids: Iterable[int] | None = None, date_from: str | None = None,
               date_to: str | None = None) -> pd.DataFrame:
        where, params = self._date_filter("e.occurred_at", date_from, date_to)
        ids = list(session_ids) if session_ids is not None else None
        if ids is not None:
            if not ids:
                return pd.DataFrame(columns=["id", "session_id", "occurred_at", "t_s", "kind",
                                             "score", "perclos", "reasons", "source", "model"])
            where += (" AND " if where else "WHERE ") + f"e.session_id IN ({','.join('?' * len(ids))})"
            params += ids
        df = self._query(
            f"""
            SELECT e.id, e.session_id, e.occurred_at, e.t_s, e.kind, e.score, e.perclos, e.reasons,
                   s.source, s.model
            FROM events e JOIN sessions s ON s.id = e.session_id
            {where} ORDER BY e.occurred_at
            """, params)
        if not df.empty:
            df["occurred_at"] = pd.to_datetime(df["occurred_at"])
            df["reasons"] = df["reasons"].map(lambda s: json.loads(s) if s else [])
        return df

    def event_features(self, event_id: int) -> dict[str, float | None]:
        with self._lock:
            row = self._conn.execute("SELECT features FROM events WHERE id = ?", (event_id,)).fetchone()
        return json.loads(row["features"]) if row and row["features"] else {}

    def samples(self, session_id: int) -> pd.DataFrame:
        return self._query("SELECT * FROM samples WHERE session_id = ? ORDER BY t_s", (session_id,))

    @staticmethod
    def _date_filter(column: str, date_from: str | None, date_to: str | None) -> tuple[str, list[Any]]:
        conds, params = [], []
        if date_from:
            conds.append(f"{column} >= ?")
            params.append(f"{date_from}T00:00:00")
        if date_to:
            conds.append(f"{column} <= ?")
            params.append(f"{date_to}T23:59:59")
        return ("WHERE " + " AND ".join(conds)) if conds else "", params

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SessionLogger:
    """Registra una sesión de monitoreo: muestras cada segundo y alarmas.

    Las muestras se escriben en lotes (``flush_every``) para no hacer una
    transacción por segundo; las alarmas se escriben de inmediato.

    Ejemplo::

        logger = SessionLogger(db, "Webcam 0", "webcam", "Random Forest", {"persistence": 3})
        logger.log(t, decision, prediction)    # una vez por predicción
        logger.close()
    """

    def __init__(self, db: HistoryDB, source: str, source_kind: str, model: str,
                 params: Mapping[str, Any] | None = None, flush_every: int = 10) -> None:
        self.db = db
        self.session_id = db.start_session(source, source_kind, model, params)
        self.flush_every = flush_every
        self._buffer: list[tuple] = []
        self._last_t = 0.0
        self._closed = False

    def log(self, t: float, decision: Any, prediction: Any) -> int | None:
        """Guarda la muestra y, si la decisión es una alarma nueva, el evento.

        ``decision`` necesita ``level``, ``new_alarm`` y ``reasons``;
        ``prediction`` necesita ``is_drowsy``, ``score`` y ``features``.
        Devuelve el id del evento si se registró una alarma.
        """
        f = prediction.features or {}
        self._last_t = float(t)
        self._buffer.append((
            round(float(t), 2), decision.level, int(bool(prediction.is_drowsy)), _clean(prediction.score),
            _clean(f.get("perclos")), _clean(f.get("blink_rate_per_min")),
            _clean(f.get("yawn_count")), _clean(f.get("nod_count")),
        ))
        event_id = None
        if decision.new_alarm:
            self.flush()
            event_id = self.db.add_event(self.session_id, t, prediction.score, decision.reasons, f)
        elif len(self._buffer) >= self.flush_every:
            self.flush()
        return event_id

    def flush(self) -> None:
        self.db.add_samples(self.session_id, self._buffer)
        self._buffer = []

    def close(self, status: str = "terminada") -> None:
        if self._closed:
            return
        self.flush()
        self.db.end_session(self.session_id, self._last_t, status)
        self._closed = True
