from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def platt_calibration(p: float, a: float = 1.0, b: float = 0.0) -> float:
    """Simple logistic calibration over raw probability/logit-like score."""
    import math

    x = _clip(p, 1e-6, 1.0 - 1e-6)
    logit = math.log(x / (1.0 - x))
    return _clip(1.0 / (1.0 + math.exp(-(a * logit + b))))


@dataclass
class EnsembleOutput:
    direction: str
    signal_strength: float
    p_up_3m: float
    p_up_5m: float
    p_tp_first: float
    p_sl_first: float
    confidence: float
    no_trade_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "signal_strength": self.signal_strength,
            "p_up_3m": self.p_up_3m,
            "p_up_5m": self.p_up_5m,
            "p_tp_first": self.p_tp_first,
            "p_sl_first": self.p_sl_first,
            "confidence": self.confidence,
            "no_trade_reason": self.no_trade_reason,
        }


def meta_decide(
    model_a: Dict[str, float],
    model_b: Dict[str, float],
    regime: str,
    cfg: Dict[str, Any],
) -> EnsembleOutput:
    """Stacking-like deterministic meta-model for online inference.

    Combines Model-A/Model-B probabilities, applies calibration + uncertainty gray-zone.
    """
    c = cfg or {}
    w_a = float(c.get("w_a", 0.55))
    w_b = float(c.get("w_b", 0.45))
    gray_lo, gray_hi = c.get("gray_zone", [0.45, 0.55])

    p_a = _clip(model_a.get("p_up_3m", 0.5))
    p_b = _clip(model_b.get("p_up_3m", 0.5))
    p5_a = _clip(model_a.get("p_up_5m", p_a))
    p5_b = _clip(model_b.get("p_up_5m", p_b))

    p_up_3m_raw = _clip(w_a * p_a + w_b * p_b)
    p_up_5m_raw = _clip(w_a * p5_a + w_b * p5_b)

    platt = c.get("platt", {})
    p_up_3m = platt_calibration(p_up_3m_raw, float(platt.get("a3", 1.0)), float(platt.get("b3", 0.0)))
    p_up_5m = platt_calibration(p_up_5m_raw, float(platt.get("a5", 1.0)), float(platt.get("b5", 0.0)))

    p_tp_first = _clip(0.6 * p_up_3m + 0.4 * p_up_5m)
    p_sl_first = _clip(1.0 - p_tp_first)
    signal_strength = abs(p_up_3m - 0.5) * 2.0
    confidence = _clip(max(p_up_3m, 1.0 - p_up_3m))

    no_trade_reason = ""
    direction = "NEUTRAL"
    if gray_lo < p_up_3m < gray_hi:
        no_trade_reason = "gray_zone"
    elif regime == "low_liquidity":
        no_trade_reason = "low_liquidity"
    else:
        direction = "LONG" if p_up_3m >= 0.5 else "SHORT"

    return EnsembleOutput(
        direction=direction,
        signal_strength=float(signal_strength),
        p_up_3m=float(p_up_3m),
        p_up_5m=float(p_up_5m),
        p_tp_first=float(p_tp_first),
        p_sl_first=float(p_sl_first),
        confidence=float(confidence),
        no_trade_reason=no_trade_reason,
    )

