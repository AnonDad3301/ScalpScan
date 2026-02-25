from __future__ import annotations
import sqlite3, time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = '''
CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exchange TEXT NOT NULL,
  market TEXT NOT NULL,
  symbol TEXT NOT NULL,
  base TEXT,
  quote TEXT,
  active INTEGER,
  volume REAL,
  last_updated INTEGER,
  UNIQUE(exchange, market, symbol)
);
CREATE INDEX IF NOT EXISTS idx_symbols_ex_mkt_vol ON symbols(exchange, market, volume DESC);
'''

class SymbolStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_many(self, exchange: str, market: str, rows: List[Dict[str, Any]]) -> int:
        now = int(time.time()*1000)
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO symbols(exchange, market, symbol, base, quote, active, volume, last_updated)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(exchange, market, symbol) DO UPDATE SET "
            " base=excluded.base, quote=excluded.quote, active=excluded.active, volume=excluded.volume, last_updated=excluded.last_updated",
            [(exchange, market, r.get("symbol"), r.get("base"), r.get("quote"),
              1 if r.get("active", True) else 0, float(r.get("volume") or 0.0), now) for r in rows]
        )
        self.conn.commit()
        return len(rows)

    def top_symbols(self, exchange: str, market: str, limit: int, quote: Optional[List[str]]=None, min_volume: float=0.0) -> List[str]:
        q = "SELECT symbol FROM symbols WHERE exchange=? AND market=? AND (active=1 OR active IS NULL)"
        params: List[Any] = [exchange, market]
        if quote:
            q += " AND quote IN (" + ",".join(["?"]*len(quote)) + ")"
            params += quote
        if min_volume > 0:
            q += " AND volume>=?"
            params.append(float(min_volume))
        q += " ORDER BY volume DESC, symbol ASC LIMIT ?"
        params.append(int(limit))
        cur = self.conn.execute(q, tuple(params))
        return [r[0] for r in cur.fetchall()]
