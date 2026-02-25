from __future__ import annotations
from typing import Any, Dict, List
import math

def check(features: Dict[str, float], inv_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rsi_min, rsi_max = inv_cfg.get("rsi_range", [0,100])
    adx_min, adx_max = inv_cfg.get("adx_range", [0,100])
    atr_min = float(inv_cfg.get("atr_min", 1e-12))
    forbid_nan = bool(inv_cfg.get("forbid_nan", True))

    def is_nan(x: float) -> bool:
        return x is None or (isinstance(x, float) and math.isnan(x))

    out=[]
    rsi=float(features.get("rsi", float("nan")))
    adx=float(features.get("adx", float("nan")))
    atr=float(features.get("atr", float("nan")))

    out.append({"name":"RSI_RANGE","pass": (rsi_min<=rsi<=rsi_max), "value": rsi})
    out.append({"name":"ADX_RANGE","pass": (adx_min<=adx<=adx_max), "value": adx})
    out.append({"name":"ATR_MIN","pass": (atr>atr_min), "value": atr})
    if forbid_nan:
        has_nan = any(is_nan(float(v)) for v in features.values())
        out.append({"name":"NO_NAN","pass": (not has_nan)})
    return out

def ok(results: List[Dict[str, Any]]) -> bool:
    return all(bool(r.get("pass")) for r in results)
