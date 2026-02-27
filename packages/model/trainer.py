from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from .lorentzian_knn import LorentzianKNN

@dataclass
class Metrics:
    samples: int = 0
    auc: float = 0.0
    accuracy: float = 0.0
    version: str = "lorentzian_knn_local"

class OnlineTrainer:
    """Online trainer with throttled retraining.

    Problem seen in production: retrain() per symbol per tick can stall the scanner thread for
    tens of seconds (O(n log n) per validation point due to argsort), starving the system.
    Fix: retrain at most once per interval, and cap train/val sizes.
    """
    def __init__(
        self,
        k: int = 35,
        min_samples: int = 300,
        retrain_interval_sec: float = 60.0,
        max_train_samples: int = 3000,
        max_val_samples: int = 200,
    ):
        self.model = LorentzianKNN(k=k)
        self.min_samples = int(min_samples)
        self.retrain_interval_ms = int(max(1.0, float(retrain_interval_sec)) * 1000)
        self.max_train_samples = int(max_train_samples)
        self.max_val_samples = int(max_val_samples)

        self.X: List[np.ndarray] = []
        self.y: List[int] = []
        self.metrics = Metrics()
        self._last_retrain_ms: int = 0
        self._since_last_added: int = 0

    def add_samples(self, X: np.ndarray, y: np.ndarray) -> int:
        added=0
        for xi, yi in zip(X, y):
            yi=int(yi)
            if yi==0:
                continue
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(yi)
            added += 1
        if added:
            self._since_last_added += added
            # cap memory
            if len(self.X) > self.max_train_samples:
                drop = len(self.X) - self.max_train_samples
                self.X = self.X[drop:]
                self.y = self.y[drop:]
            # keep health() and UI counters live even when retrain is throttled
            self.metrics.samples = len(self.X)
        return added

    def maybe_retrain(self, now_ms: Optional[int]=None, force: bool=False) -> Tuple[Metrics, bool]:
        if now_ms is None:
            now_ms = int(time.time()*1000)
        n = len(self.X)
        self.metrics.samples = n
        if n < self.min_samples:
            return self.metrics, False
        if not force:
            if (now_ms - self._last_retrain_ms) < self.retrain_interval_ms:
                return self.metrics, False
            # don't retrain if we barely added anything
            if self._since_last_added < max(50, int(0.02 * n)):
                return self.metrics, False
        self._last_retrain_ms = now_ms
        self._since_last_added = 0
        return self.retrain(), True

    def retrain(self) -> Metrics:
        n = len(self.X)
        self.metrics.samples = n
        if n < self.min_samples:
            return self.metrics

        X = np.vstack(self.X)
        y = np.asarray(self.y, dtype=int)

        # train/val split
        split = int(n * 0.8)
        X_tr, y_tr = X[:split], y[:split]
        X_va, y_va = X[split:], y[split:]

        # cap val size for speed
        if len(X_va) > self.max_val_samples:
            idx = np.linspace(0, len(X_va)-1, num=self.max_val_samples).astype(int)
            X_va = X_va[idx]
            y_va = y_va[idx]

        self.model.fit(X_tr, y_tr)

        probs=[]; preds=[]
        for xv in X_va:
            _, p = self.model.predict_proba(xv)
            probs.append(p)
            preds.append(1 if p>=0.5 else -1)

        y01 = (y_va > 0).astype(int)
        try:
            self.metrics.auc = float(roc_auc_score(y01, np.asarray(probs)))
        except Exception:
            self.metrics.auc = 0.0
        try:
            self.metrics.accuracy = float(accuracy_score(y_va, np.asarray(preds)))
        except Exception:
            self.metrics.accuracy = 0.0
        return self.metrics

    def infer(self, x: np.ndarray) -> Dict[str, Any]:
        pred = float(self.model.predict(x))
        conf = float(self.model.confidence(x))
        return {"pred": pred, "confidence": conf}

    def health(self) -> Dict[str, Any]:
        samples = len(self.X)
        self.metrics.samples = samples
        return {"samples": samples, "auc": self.metrics.auc, "accuracy": self.metrics.accuracy, "version": self.metrics.version}

    def on_trade_closed(self, trade: dict) -> None:
        # placeholder for future: supervised updates on closed trades
        return
