from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class PatternScanResult:
    volume_profile_poc: float
    support: float
    resistance: float
    volatility_breakout: bool
    breakout_strength: float

    def as_dict(self) -> Dict[str, float | bool]:
        return {
            "volume_profile_poc": self.volume_profile_poc,
            "support": self.support,
            "resistance": self.resistance,
            "volatility_breakout": self.volatility_breakout,
            "breakout_strength": self.breakout_strength,
        }


class MarketPatternScanner:
    """Fast market pattern scanner for 1-15m scalping horizons."""

    def __init__(self, bins: int = 24, breakout_window: int = 20):
        self.bins = max(8, int(bins))
        self.breakout_window = max(5, int(breakout_window))

    def scan(self, ohlcv: Sequence[Sequence[float]]) -> PatternScanResult:
        if not ohlcv:
            return PatternScanResult(0.0, 0.0, 0.0, False, 0.0)
        arr = np.asarray(ohlcv, dtype=float)
        highs = arr[:, 2]
        lows = arr[:, 3]
        closes = arr[:, 4]
        volumes = arr[:, 5] if arr.shape[1] > 5 else np.ones_like(closes)

        support = float(np.nanpercentile(lows, 20))
        resistance = float(np.nanpercentile(highs, 80))
        poc = self._volume_profile_poc(closes, volumes)

        window = min(self.breakout_window, len(closes))
        recent_close = float(closes[-1])
        rolling_high = float(np.max(highs[-window:]))
        rolling_low = float(np.min(lows[-window:]))
        band = max(1e-9, rolling_high - rolling_low)
        breakout_up = recent_close > rolling_high
        breakout_down = recent_close < rolling_low
        breakout_strength = abs(recent_close - np.clip(recent_close, rolling_low, rolling_high)) / band

        return PatternScanResult(
            volume_profile_poc=poc,
            support=support,
            resistance=resistance,
            volatility_breakout=bool(breakout_up or breakout_down),
            breakout_strength=float(breakout_strength),
        )

    def cross_asset_correlation(
        self,
        asset_closes: Dict[str, Sequence[float]],
        max_lag: int = 3,
    ) -> Dict[str, float]:
        """Returns average absolute rolling correlation for each asset."""
        prepared: Dict[str, np.ndarray] = {}
        for symbol, closes in asset_closes.items():
            arr = np.asarray(closes, dtype=float)
            if arr.size < 5:
                continue
            rets = np.diff(np.log(np.clip(arr, 1e-12, None)))
            prepared[symbol] = rets

        symbols = list(prepared.keys())
        if len(symbols) < 2:
            return {k: 0.0 for k in symbols}

        out: Dict[str, List[float]] = {k: [] for k in symbols}
        for i, a in enumerate(symbols):
            for b in symbols[i + 1 :]:
                corr = self._lagged_corr(prepared[a], prepared[b], max_lag=max_lag)
                out[a].append(abs(corr))
                out[b].append(abs(corr))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}

    def _volume_profile_poc(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        if prices.size < 2:
            return float(prices[-1])
        lo, hi = float(np.min(prices)), float(np.max(prices))
        if abs(hi - lo) < 1e-12:
            return float(prices[-1])
        bins = np.linspace(lo, hi, self.bins + 1)
        hist, edges = np.histogram(prices, bins=bins, weights=volumes)
        idx = int(np.argmax(hist))
        return float((edges[idx] + edges[idx + 1]) / 2)

    @staticmethod
    def _lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> float:
        n = min(a.size, b.size)
        if n < 5:
            return 0.0
        a = a[-n:]
        b = b[-n:]
        best = 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x, y = a[-lag:], b[: n + lag]
            elif lag > 0:
                x, y = a[: n - lag], b[lag:]
            else:
                x, y = a, b
            if x.size < 5 or y.size < 5:
                continue
            c = np.corrcoef(x, y)[0, 1]
            if np.isfinite(c) and abs(c) > abs(best):
                best = float(c)
        return best
