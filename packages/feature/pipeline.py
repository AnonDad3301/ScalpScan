from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .indicators import rsi, atr, adx_wilder

def compute(ohlcv: List[Tuple[int,float,float,float,float,float]], ob: Dict[str, Any]) -> Dict[str, float]:
    arr = np.array([[o,h,l,c,v] for _,o,h,l,c,v in ohlcv], dtype=float)
    if arr.size == 0:
        return {
            "rsi": 50.0,
            "atr": 0.0,
            "adx": 0.0,
            "spread": 0.0,
            "spread_bps": 0.0,
            "ob_imb": 0.0,
            "volz": 0.0,
            "ret_last": 0.0,
            "range_last": 0.0,
            "vol_z_last": 0.0,
        }

    high=arr[:,1]; low=arr[:,2]; close=arr[:,3]; vol=arr[:,4]
    ret=np.zeros_like(close)
    if len(close) > 1:
        ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    range_last = float((high[-1] - low[-1]) / max(1e-12, close[-1]))

    win=min(30, len(vol))
    vol_ref=vol[-win:]
    vol_z=(vol[-1]-float(np.mean(vol_ref))) / max(1e-12, float(np.std(vol_ref)))

    spread = float(ob.get("spread", 0.0))
    spread_bps = spread / max(1e-12, float(close[-1])) * 10000.0

    return {
        "rsi": rsi(close, 14),
        "atr": atr(high, low, close, 14),
        "adx": adx_wilder(high, low, close, 14),
        "spread": spread,
        "spread_bps": spread_bps,
        "ob_imb": float(ob.get("imbalance", 0.0)),
        "volz": float(vol_z),
        "ret_last": float(ret[-1]),
        "range_last": range_last,
        "vol_z_last": float(vol_z),
    }
