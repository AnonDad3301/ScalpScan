from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


class SQLiteTimelineStore:
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS timeline ("
                "ts INTEGER, metric TEXT, value REAL, tags_json TEXT, PRIMARY KEY(ts, metric, tags_json))"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_ts ON timeline(ts)")
            self.conn.commit()

    def append(self, ts: int, metric: str, value: float, tags: Dict[str, Any] | None = None) -> None:
        tj = json.dumps(tags or {}, ensure_ascii=False)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO timeline(ts, metric, value, tags_json) VALUES(?,?,?,?)",
                (int(ts), str(metric), float(value), tj),
            )
            self.conn.commit()

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, metric, value, tags_json FROM timeline ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for ts, metric, value, tj in rows:
            try:
                tags = json.loads(tj) if tj else {}
            except Exception:
                tags = {}
            out.append({"ts": ts, "metric": metric, "value": value, "tags": tags})
        return out
