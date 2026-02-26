from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


class MultiHorizonForecaster:
    """Lightweight multi-horizon forecaster with uncertainty estimate."""

    def __init__(self, horizons: Sequence[int] = (1, 3, 5, 15), mc_samples: int = 64):
        self.horizons = tuple(sorted({max(1, int(h)) for h in horizons}))
        self.mc_samples = max(8, int(mc_samples))

    def predict(self, closes: Sequence[float]) -> Dict[str, Dict[str, float]]:
        arr = np.asarray(closes, dtype=float)
        if arr.size < 4:
            return {f"m{h}": {"ret": 0.0, "uncertainty": 1.0} for h in self.horizons}

        x = np.arange(arr.size, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        residuals = arr - (slope * x + intercept)
        residual_std = float(np.std(residuals))
        cur = float(arr[-1])
        out: Dict[str, Dict[str, float]] = {}

        for h in self.horizons:
            next_x = float(arr.size - 1 + h)
            point = float(slope * next_x + intercept)
            ret = (point - cur) / max(1e-12, cur)
            unc = self._mc_uncertainty(cur, point, residual_std)
            out[f"m{h}"] = {"ret": float(ret), "uncertainty": float(unc)}
        return out

    def _mc_uncertainty(self, cur: float, point: float, residual_std: float) -> float:
        noise = np.random.normal(0.0, residual_std, size=self.mc_samples)
        samples = (point + noise - cur) / max(1e-12, cur)
        return float(np.std(samples))
