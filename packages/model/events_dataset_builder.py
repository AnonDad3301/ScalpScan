from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from packages.infra.event_store.sqlite_store import SQLiteEventStore


class EventsDatasetBuilder:
    """Builds model dataset from events.sqlite (schema v1/v2 compatible)."""

    def __init__(self, events_db: str, out_csv: str):
        self.store = SQLiteEventStore(events_db)
        self.out_csv = str(out_csv)
        Path(self.out_csv).parent.mkdir(parents=True, exist_ok=True)

    def build_model_b(self, limit: int = 10000) -> int:
        ev = list(reversed(self.store.tail(limit)))
        signals: Dict[str, List[Dict[str, Any]]] = {}
        rows = []

        for e in ev:
            st = e.get("stage")
            sym = str(e.get("symbol") or "")
            if not sym:
                continue
            if st == "MODEL_B_INFERRED":
                signals.setdefault(sym, []).append(e)
            elif st == "TRADE_CLOSED":
                p = e.get("payload") or {}
                pnl = float(p.get("pnl", 0.0) or 0.0)
                label = 1 if pnl > 0 else 0
                cand = None
                for x in reversed(signals.get(sym, [])):
                    if int(x.get("ts") or 0) <= int(e.get("ts") or 0):
                        cand = x
                        break
                if cand is None:
                    continue
                fp = cand.get("payload") or {}
                rows.append([
                    int(cand.get("ts") or 0), sym, label, pnl,
                    fp.get("ret_last", 0.0), fp.get("range_last", 0.0), fp.get("vol_z_last", 0.0),
                    fp.get("imb", 0.0), fp.get("spread_bps", 0.0), fp.get("wall_dist_bps", 0.0),
                    fp.get("wall_age", 0.0), fp.get("wall_touches", 0.0),
                ])

        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts","symbol","label","target",
                "ret_last","range_last","vol_z_last","imb","spread_bps","wall_dist_bps","wall_age","wall_touches"
            ])
            w.writerows(rows)
        return len(rows)
