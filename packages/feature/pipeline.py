from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .indicators import rsi, atr, adx_wilder

def compute(ohlcv: List[Tuple[int,float,float,float,float,float]], ob: Dict[str, Any]) -> Dict[str, float]:
    arr = np.array([[o,h,l,c,v] for _,o,h,l,c,v in ohlcv], dtype=float)
    high=arr[:,1]; low=arr[:,2]; close=arr[:,3]; vol=arr[:,4]
    last=float(close[-1]) if len(close) else 0.0
    ret_last=float((close[-1]-close[-2])/max(1e-12,close[-2])) if len(close)>=2 else 0.0
    range_last=float((high[-1]-low[-1])/max(1e-12,last)) if len(close) else 0.0
    rets=np.diff(close)/np.maximum(1e-12,close[:-1]) if len(close)>=3 else np.array([],dtype=float)
    vol_z_last=0.0
    if len(rets)>=10:
        mu=float(np.mean(rets[-10:]))
        sd=float(np.std(rets[-10:]))
        vol_z_last=float((ret_last-mu)/max(1e-12,sd))

    spread_raw=float(ob.get("spread", 0.0) or 0.0)
    spread_bps=float(ob.get("spread_bps", 0.0) or 0.0)
    if spread_bps<=0.0 and last>0.0:
        spread_bps=abs(spread_raw)/last*10000.0

    return {
        "rsi": rsi(close, 14),
        "atr": atr(high, low, close, 14),
        "adx": adx_wilder(high, low, close, 14),
        "spread": spread_raw,
        "spread_bps": spread_bps,
        "ob_imb": float(ob.get("imbalance", 0.0) or 0.0),
        "ret_last": ret_last,
        "range_last": range_last,
        "vol_z_last": vol_z_last,
        "volz": abs(vol_z_last),
    }
