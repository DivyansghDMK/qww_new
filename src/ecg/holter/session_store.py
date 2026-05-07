"""
Session storage helpers for Comprehensive ECG Analysis.

Provides a layered store alongside the legacy JSONL workflow:
- session.json for metadata
- events.db for indexed metrics, events, and annotations
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional


SCHEMA_VERSION = 1


def session_json_path(session_dir: str) -> str:
    return os.path.join(session_dir, "session.json")


def events_db_path(session_dir: str) -> str:
    return os.path.join(session_dir, "events.db")


def write_session_metadata(session_dir: str, metadata: Dict[str, object]) -> str:
    os.makedirs(session_dir, exist_ok=True)
    payload = dict(metadata or {})
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
    path = session_json_path(session_dir)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_session_metadata(session_dir: str) -> Dict[str, object]:
    path = session_json_path(session_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_events_db(session_dir: str) -> str:
    os.makedirs(session_dir, exist_ok=True)
    db_path = events_db_path(session_dir)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                t REAL NOT NULL,
                duration REAL,
                hr_mean REAL,
                hr_min REAL,
                hr_max REAL,
                beat_count INTEGER,
                quality REAL,
                json_payload TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t REAL NOT NULL,
                label TEXT NOT NULL,
                event_type TEXT,
                source TEXT DEFAULT 'analysis',
                confidence REAL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                beat_id TEXT NOT NULL,
                auto_label TEXT,
                clinician_label TEXT,
                confidence REAL,
                edited_by TEXT,
                timestamp REAL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
    return db_path


def append_metric(session_dir: str, metric: Dict[str, object]) -> None:
    db_path = ensure_events_db(session_dir)
    payload = json.dumps(metric, sort_keys=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO metrics (t, duration, hr_mean, hr_min, hr_max, beat_count, quality, json_payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(metric.get("t", 0.0) or 0.0),
                float(metric.get("duration", 0.0) or 0.0),
                float(metric.get("hr_mean", 0.0) or 0.0),
                float(metric.get("hr_min", 0.0) or 0.0),
                float(metric.get("hr_max", 0.0) or 0.0),
                int(metric.get("beat_count", metric.get("n_beats", 0)) or 0),
                float(metric.get("quality", 0.0) or 0.0),
                payload,
            ),
        )


def append_events(session_dir: str, events: Iterable[Dict[str, object]]) -> None:
    db_path = ensure_events_db(session_dir)
    rows = []
    for event in events or []:
        rows.append((
            float(event.get("timestamp", event.get("t", 0.0)) or 0.0),
            str(event.get("label", event.get("type", "Event"))),
            str(event.get("event_type", event.get("template_label", event.get("type", "")))),
            str(event.get("source", "analysis")),
            float(event.get("confidence", 0.0) or 0.0) if event.get("confidence") is not None else None,
            json.dumps(event, sort_keys=True),
        ))
    if not rows:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO events (t, label, event_type, source, confidence, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def append_annotation(session_dir: str, annotation: Dict[str, object]) -> None:
    db_path = ensure_events_db(session_dir)
    payload = json.dumps(annotation, sort_keys=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO annotations (beat_id, auto_label, clinician_label, confidence, edited_by, timestamp, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(annotation.get("beat_id", "")),
                str(annotation.get("auto_label", "")),
                str(annotation.get("clinician_label", "")),
                float(annotation.get("confidence", 0.0) or 0.0) if annotation.get("confidence") is not None else None,
                str(annotation.get("edited_by", "")),
                float(annotation.get("timestamp", 0.0) or 0.0),
                payload,
            ),
        )


def load_metrics(session_dir: str) -> List[Dict[str, object]]:
    db_path = events_db_path(session_dir)
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT json_payload FROM metrics ORDER BY seq ASC").fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["json_payload"]))
        except Exception:
            continue
    return result


def load_events(session_dir: str) -> List[Dict[str, object]]:
    db_path = events_db_path(session_dir)
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT payload FROM events ORDER BY t ASC, id ASC").fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["payload"]))
        except Exception:
            continue
    return result


def load_annotations(session_dir: str) -> List[Dict[str, object]]:
    db_path = events_db_path(session_dir)
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT payload FROM annotations ORDER BY timestamp ASC, id ASC").fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["payload"]))
        except Exception:
            continue
    return result
