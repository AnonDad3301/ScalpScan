from __future__ import annotations
import sqlite3, json
from typing import Any, Dict, List

class TradesStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS trades(
            trade_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            market_type TEXT,
            price_source TEXT,
            bar_ts INTEGER,
            decision_ts INTEGER,
            entry_ts INTEGER,
            exit_ts INTEGER,
            entry REAL,
            exit REAL,
            qty REAL,
            fees REAL,
            pnl REAL,
            r_mult REAL,
            reason_exit TEXT,
            payload_json TEXT
        )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_ts ON trades(exit_ts)")
        self.conn.commit()

    def upsert(self, t: Dict[str, Any]) -> None:
        trade_id = str(t.get("trade_id") or t.get("id") or "")
        if not trade_id:
            trade_id = f"{t.get('symbol','')}_{t.get('entry_ts',0)}"
        row = (
            trade_id,
            str(t.get("symbol","")),
            str(t.get("side","")),
            str(t.get("market_type","")),
            str(t.get("price_source","")),
            int(t.get("bar_ts",0) or 0),
            int(t.get("decision_ts",0) or 0),
            int(t.get("entry_ts",0) or 0),
            int(t.get("exit_ts",0) or 0),
            float(t.get("entry",0.0) or 0.0),
            float(t.get("exit",0.0) or 0.0),
            float(t.get("qty",0.0) or 0.0),
            float(t.get("fees",0.0) or 0.0),
            float(t.get("pnl",0.0) or 0.0),
            float(t.get("r_mult",0.0) or 0.0),
            str(t.get("reason_exit","")),
            json.dumps(t, ensure_ascii=False),
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trades(
                trade_id, symbol, side, market_type, price_source,
                bar_ts, decision_ts, entry_ts, exit_ts,
                entry, exit, qty, fees, pnl, r_mult,
                reason_exit, payload_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            row
        )
        self.conn.commit()

    def append_closed(self, t: Dict[str, Any]) -> None:
        """Alias for upsert to maintain compatibility"""
        self.upsert(t)

    def tail(self, limit: int = 200) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT payload_json FROM trades ORDER BY exit_ts DESC LIMIT ?", (int(limit),))
        out=[]
        for (pj,) in cur.fetchall():
            try:
                out.append(json.loads(pj))
            except Exception:
                pass
        return out
