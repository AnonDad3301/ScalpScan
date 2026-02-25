from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

def lorentzian_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.log1p(np.abs(b - a)), axis=-1)

@dataclass
class LorentzianKNN:
    k: int = 35
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)

    def predict_proba(self, x: np.ndarray) -> Tuple[float, float]:
        if self.X is None or self.y is None or len(self.X) < 10:
            return 0.5, 0.5
        x = np.asarray(x, dtype=float)
        d = lorentzian_distance(x, self.X)
        idx = np.argsort(d)[: max(5, min(self.k, len(self.X)))]
        ny = self.y[idx]
        p_pos = float(np.mean((ny > 0).astype(float)))
        return 1.0 - p_pos, p_pos

    def predict(self, x: np.ndarray) -> float:
        p_neg, p_pos = self.predict_proba(x)
        return float(p_pos - p_neg)

    def confidence(self, x: np.ndarray) -> float:
        p_neg, p_pos = self.predict_proba(x)
        return float(max(p_neg, p_pos))
