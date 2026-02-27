from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


class SQLiteEventStore:
    """Thread-safe SQLite event store with schema v2 migration support."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "ts INTEGER, run_id TEXT, service TEXT, exchange TEXT, market TEXT, symbol TEXT, timeframe TEXT, "
                "stage TEXT, level TEXT, event_id TEXT PRIMARY KEY, payload_json TEXT)"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            self._migrate_v2_if_needed()
            self.conn.commit()

    def _migrate_v2_if_needed(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)").fetchall()}
        if "schema_version" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN schema_version INTEGER DEFAULT 2")
        if "source" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN source TEXT DEFAULT 'runtime'")
        if "tags_json" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN tags_json TEXT DEFAULT '{}' ")

    def append(self, event: Dict[str, Any]) -> None:
        ts = int(event.get("ts") or 0)
        run_id = str(event.get("run_id") or "")
        service = str(event.get("service") or "")
        exchange = None if event.get("exchange") is None else str(event.get("exchange"))
        market = None if event.get("market") is None else str(event.get("market"))
        symbol = None if event.get("symbol") is None else str(event.get("symbol"))
        timeframe = None if event.get("timeframe") is None else str(event.get("timeframe"))
        stage = str(event.get("stage") or "")
        level = str(event.get("level") or "INFO")
        event_id = str(event.get("event_id") or "")
        payload_json = json.dumps(event.get("payload", {}) or {}, ensure_ascii=False)
        schema_version = int(event.get("schema_version", 2) or 2)
        source = str(event.get("source", "runtime") or "runtime")
        tags_json = json.dumps(event.get("tags", {}) or {}, ensure_ascii=False)

        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO events("
                "ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json, schema_version, source, tags_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json, schema_version, source, tags_json),
            )
            self.conn.commit()

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json, "
                "COALESCE(schema_version,2), COALESCE(source,'runtime'), COALESCE(tags_json,'{}') "
                "FROM events ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json, schema_version, source, tags_json in rows:
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except Exception:
                payload = {}
            try:
                tags = json.loads(tags_json) if tags_json else {}
            except Exception:
                tags = {}
            out.append({
                "ts": ts,
                "run_id": run_id,
                "service": service,
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "stage": stage,
                "level": level,
                "event_id": event_id,
                "payload": payload,
                "schema_version": int(schema_version or 2),
                "source": source,
                "tags": tags,
            })
        return out
