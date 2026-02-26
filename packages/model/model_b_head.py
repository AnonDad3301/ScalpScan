from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class Metrics:
    samples:int=0
    auc:float=0.0
    pr_auc:float=0.0
    acc:float=0.0

class ModelBHead:
    def __init__(self):
        from sklearn.linear_model import SGDClassifier
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4)
        self.fitted=False
        self.metrics=Metrics()

    def fit_eval(self, X: np.ndarray, y: np.ndarray) -> Metrics:
        self.metrics.samples=int(len(y))
        if len(y)<200 or len(np.unique(y))<2:
            return self.metrics
        split=int(len(y)*0.8)
        Xtr,ytr=X[:split],y[:split]
        Xva,yva=X[split:],y[split:]
        self.clf.fit(Xtr,ytr)
        self.fitted=True
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
            p=self.proba(Xva)
            self.metrics.auc=float(roc_auc_score(yva,p))
            self.metrics.pr_auc=float(average_precision_score(yva,p))
            self.metrics.acc=float(accuracy_score(yva,(p>=0.5).astype(int)))
        except Exception:
            pass
        return self.metrics

    def proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full((len(X),), 0.5, dtype=float)
        z=self.clf.decision_function(X)
        z=np.clip(z, -60.0, 60.0)
        return 1.0/(1.0+np.exp(-z))

    def infer_one(self, feats: Dict[str,Any]) -> float:
        X=np.asarray([[float(feats.get(k,0.0)) for k in (
            "ret_last","range_last","vol_z_last","imb","spread_bps","wall_dist_bps","wall_age","wall_touches"
        )]], dtype=float)
        return float(self.proba(X)[0])
