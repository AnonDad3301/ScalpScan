from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RegimeState:
    name: str
    volatility: float
    trend_strength: float

    def as_dict(self):
        return {
            "name": self.name,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
        }


class MarketRegimeDetector:
    """Simple regime detector: calm / trend / volatile."""

    def __init__(self, vol_threshold: float = 0.004, trend_threshold: float = 0.25):
        self.vol_threshold = float(vol_threshold)
        self.trend_threshold = float(trend_threshold)

    def detect(self, closes: Sequence[float]) -> RegimeState:
        arr = np.asarray(closes, dtype=float)
        if arr.size < 8:
            return RegimeState(name="unknown", volatility=0.0, trend_strength=0.0)

        rets = np.diff(np.log(np.clip(arr, 1e-12, None)))
        vol = float(np.std(rets))

        x = np.arange(arr.size, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        fit = slope * x + intercept
        ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
        ss_res = float(np.sum((arr - fit) ** 2))
        trend = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)

        if vol >= self.vol_threshold:
            name = "volatile"
        elif trend >= self.trend_threshold:
            name = "trend"
        else:
            name = "calm"
        return RegimeState(name=name, volatility=vol, trend_strength=trend)
