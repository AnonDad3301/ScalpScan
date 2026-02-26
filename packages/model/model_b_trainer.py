from __future__ import annotations
import os, csv, time
import numpy as np
from typing import Any, Dict, Optional, Tuple
from .model_b_head import ModelBHead, Metrics
from .model_registry import Registry

class Trainer:
    def __init__(self, dataset_path: str, registry: Registry, retrain_sec: float = 1800.0, retrain_interval_sec: float | None = None, **kwargs):
        self.dataset_path=dataset_path
        self.registry=registry
        interval = retrain_interval_sec if retrain_interval_sec is not None else retrain_sec
        self.retrain_ms=int(float(interval)*1000)
        self.last_train_ms=0
        self.head=self.registry.load_current("model_b_head") or ModelBHead()
        self.last_metrics: Optional[Metrics]=None
        self.last_retrain_reason: str = "init"

    def _load(self)->Optional[Tuple[np.ndarray,np.ndarray]]:
        if not os.path.exists(self.dataset_path):
            return None
        X=[]; y=[]
        with open(self.dataset_path,"r",encoding="utf-8") as f:
            r=csv.reader(f); _=next(r,None)
            for row in r:
                if len(row)<16:
                    continue
                try:
                    y.append(int(float(row[2])))
                    X.append([float(row[i]) for i in range(8,16)])
                except Exception:
                    continue
        if len(y)<200 or len(set(y))<2:
            return None
        return np.asarray(X,dtype=float), np.asarray(y,dtype=int)

    def maybe_retrain(self, now_ms:int, force:bool=False)->Tuple[Optional[Metrics], bool]:
        if (not force) and (now_ms-self.last_train_ms)<self.retrain_ms:
            self.last_retrain_reason = "cooldown"
            return self.last_metrics, False
        data=self._load()
        if data is None:
            self.last_retrain_reason = "insufficient_dataset"
            return self.last_metrics, False
        X,y=data
        head=ModelBHead()
        m=head.fit_eval(X,y)
        self.last_metrics=m
        self.last_train_ms=now_ms
        promoted=False
        if m.samples>=400 and m.auc>=0.55:
            info=self.registry.save("model_b_head", head, {"auc":m.auc,"pr_auc":m.pr_auc,"acc":m.acc,"samples":m.samples})
            self.registry.promote(info)
            self.head=head
            promoted=True
        self.last_retrain_reason = "trained"
        return m, promoted

    def infer_prob(self, feats: Dict[str,Any])->float:
        try:
            return float(self.head.infer_one(feats))
        except Exception:
            return 0.5


    def dataset_stats(self) -> Dict[str, Any]:
        rows = 0
        pos = 0
        neg = 0
        if not os.path.exists(self.dataset_path):
            return {"rows": 0, "pos": 0, "neg": 0, "pos_rate": 0.0}
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                _ = next(r, None)
                for row in r:
                    if len(row) < 3:
                        continue
                    rows += 1
                    y = int(float(row[2]))
                    if y == 1:
                        pos += 1
                    else:
                        neg += 1
        except Exception:
            return {"rows": 0, "pos": 0, "neg": 0, "pos_rate": 0.0}
        return {"rows": rows, "pos": pos, "neg": neg, "pos_rate": (float(pos) / max(1, rows))}
