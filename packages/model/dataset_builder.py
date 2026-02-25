from __future__ import annotations
import os, csv
from typing import Dict, Any

class DatasetBuilder:
    def __init__(self, path: str, horizon_bars: int = 12, **kwargs):
        self.path = path
        self.horizon_bars = int(horizon_bars)
        self.extra = dict(kwargs)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "ts","symbol","label","ret","bar_ts","decision_ts","price_source","market_type",
                    "ret_last","range_last","vol_z_last","imb","spread_bps","wall_dist_bps","wall_age","wall_touches"
                ])

    def append_shadow(self, ts:int, symbol:str, bar_ts:int, decision_ts:int, price_source:str, market_type:str,
                      feats: Dict[str,float], ret: float) -> None:
        label = 1 if ret > 0 else 0
        row = [
            ts, symbol, label, ret, bar_ts, decision_ts, price_source, market_type,
            feats.get("ret_last",0.0), feats.get("range_last",0.0), feats.get("vol_z_last",0.0),
            feats.get("imb",0.0), feats.get("spread_bps",0.0), feats.get("wall_dist_bps",0.0),
            feats.get("wall_age",0.0), feats.get("wall_touches",0.0),
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"rows": 0, "path": self.path}
        try:
            with open(self.path,"r",encoding="utf-8") as f:
                n=max(0, len(f.read().splitlines())-1)
            return {"rows": n, "path": self.path}
        except Exception:
            return {"rows": 0, "path": self.path}
