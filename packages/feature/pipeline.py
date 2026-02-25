from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .indicators import rsi, atr, adx_wilder

def compute(ohlcv: List[Tuple[int,float,float,float,float,float]], ob: Dict[str, Any]) -> Dict[str, float]:
    arr = np.array([[o,h,l,c,v] for _,o,h,l,c,v in ohlcv], dtype=float)
    high=arr[:,1]; low=arr[:,2]; close=arr[:,3]
    return {
        "rsi": rsi(close, 14),
        "atr": atr(high, low, close, 14),
        "adx": adx_wilder(high, low, close, 14),
        "spread": float(ob.get("spread", 0.0)),
        "ob_imb": float(ob.get("imbalance", 0.0)),
    }
