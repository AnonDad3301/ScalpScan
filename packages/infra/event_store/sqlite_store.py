from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

class SQLiteEventStore:
    """Thread-safe SQLite event store.

    IMPORTANT: runtime/engine.tick() may run in a background thread (asyncio.to_thread),
    so the sqlite connection must allow cross-thread use and writes must be serialized.
    """
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # allow using connection across threads
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "ts INTEGER, run_id TEXT, service TEXT, exchange TEXT, market TEXT, symbol TEXT, timeframe TEXT, "
            "stage TEXT, level TEXT, event_id TEXT PRIMARY KEY, payload_json TEXT)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self.conn.commit()
        self._lock = threading.Lock()

    def append(self, event: Dict[str, Any]) -> None:
        # Ensure primitive bindings
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

        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO events("
                "ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json),
            )
            self.conn.commit()

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        # newest first
        with self._lock:
            cur = self.conn.execute(
                "SELECT ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json "
                "FROM events ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for ts, run_id, service, exchange, market, symbol, timeframe, stage, level, event_id, payload_json in rows:
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except Exception:
                payload = {}
            out.append(
                {
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
                }
            )
        return out
