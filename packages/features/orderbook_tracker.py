from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional
import time
import math

@dataclass
class Wall:
    side: str   # BID/ASK
    price: float
    size: float
    age_sec: float
    touch_count: int
    dist_bps: float

class OrderBookTracker:
    """Tracks orderbook snapshots to estimate wall age/touches.

    Snapshot-based (no tape). Works with ccxt fetch_order_book output.
    """
    def __init__(self, max_snapshots: int = 120, price_bin_bps: float = 2.0, touch_bps: float = 3.0):
        self.max_snapshots=max_snapshots
        self.price_bin_bps=float(price_bin_bps)
        self.touch_bps=float(touch_bps)
        self.snaps: Dict[str, Deque[dict]] = {}
        self.wall_state: Dict[str, Dict[Tuple[str,float], dict]] = {}

    def update(self, symbol: str, ob: dict, mid: float) -> None:
        now_ms=int(time.time()*1000)
        dq=self.snaps.setdefault(symbol, deque(maxlen=self.max_snapshots))
        dq.append({"ts": now_ms, "ob": ob, "mid": mid})
        # update wall states based on top levels
        st=self.wall_state.setdefault(symbol, {})
        for side_key, levels in (("BID", ob.get("bids") or []), ("ASK", ob.get("asks") or [])):
            for price, size in (levels[:25] if levels else []):
                try:
                    p=float(price); sz=float(size)
                except Exception:
                    continue
                if p<=0 or sz<=0: 
                    continue
                # bin by bps around mid
                bps = abs(p-mid)/mid*10000 if mid>0 else 0.0
                bin_bps = round(bps/self.price_bin_bps)*self.price_bin_bps
                key=(side_key, bin_bps if side_key=="ASK" else -bin_bps)
                rec=st.get(key)
                if rec is None:
                    st[key]={"first_ms": now_ms, "last_ms": now_ms, "price": p, "size": sz, "touches": 0}
                else:
                    rec["last_ms"]=now_ms
                    # keep last observed best price/size for that bin
                    rec["price"]=p
                    rec["size"]=max(rec.get("size",0.0), sz)

        # decay old entries
        for k in list(st.keys()):
            if now_ms - int(st[k]["last_ms"]) > 60_000:  # 60s without seeing -> drop
                st.pop(k, None)

        # update touches: if mid got within touch_bps of a wall price
        for k, rec in st.items():
            p=float(rec["price"])
            bps=abs(p-mid)/mid*10000 if mid>0 else 0.0
            if bps <= self.touch_bps:
                rec["touches"]=int(rec.get("touches",0))+1

    def strongest_walls(self, symbol: str, mid: float, topk: int = 2) -> List[Wall]:
        st=self.wall_state.get(symbol, {})
        now_ms=int(time.time()*1000)
        walls=[]
        for (side, _bin), rec in st.items():
            p=float(rec.get("price",0.0)); sz=float(rec.get("size",0.0))
            if p<=0 or sz<=0: 
                continue
            dist_bps=abs(p-mid)/mid*10000 if mid>0 else 0.0
            age_sec=(now_ms - int(rec.get("first_ms", now_ms)))/1000.0
            walls.append(Wall(side=side, price=p, size=sz, age_sec=age_sec, touch_count=int(rec.get("touches",0)), dist_bps=dist_bps))
        walls.sort(key=lambda w: (w.dist_bps, -w.size))
        return walls[:topk]
