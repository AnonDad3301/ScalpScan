from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class SimilarCase:
    index: int
    score: float
    ret_1m: float
    ret_2m: float
    ret_4m: float


class PatternSimilarityEngine:
    def __init__(self, top_k: int = 5):
        self.top_k = max(1, int(top_k))

    def find_topk(
        self,
        current: Sequence[float],
        historical: Sequence[Sequence[float]],
        outcomes: Sequence[Dict[str, float]],
        method: str = "cosine",
    ) -> List[SimilarCase]:
        cur = np.asarray(current, dtype=float)
        if cur.size == 0:
            return []
        rows = []
        for idx, v in enumerate(historical):
            arr = np.asarray(v, dtype=float)
            if arr.shape != cur.shape:
                continue
            rows.append((idx, arr))
        if not rows:
            return []

        scored: List[SimilarCase] = []
        for idx, vec in rows:
            if method == "euclidean":
                dist = float(np.linalg.norm(cur - vec))
                score = 1.0 / (1.0 + dist)
            else:
                denom = float(np.linalg.norm(cur) * np.linalg.norm(vec))
                score = 0.0 if denom <= 1e-12 else float(np.dot(cur, vec) / denom)
            out = outcomes[idx] if idx < len(outcomes) else {}
            scored.append(
                SimilarCase(
                    index=idx,
                    score=score,
                    ret_1m=float(out.get("ret_1m", 0.0)),
                    ret_2m=float(out.get("ret_2m", 0.0)),
                    ret_4m=float(out.get("ret_4m", 0.0)),
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.top_k]

    def summarize(self, cases: Iterable[SimilarCase]) -> Dict[str, float]:
        rows = list(cases)
        if not rows:
            return {
                "similarity": 0.0,
                "p_up_4m": 0.0,
                "p_down_4m": 0.0,
                "p_flat_4m": 1.0,
                "avg_ret_1m": 0.0,
                "avg_ret_2m": 0.0,
                "avg_ret_4m": 0.0,
            }
        r4 = np.asarray([x.ret_4m for x in rows], dtype=float)
        eps = 5e-4
        return {
            "similarity": float(np.mean([x.score for x in rows])),
            "p_up_4m": float(np.mean(r4 > eps)),
            "p_down_4m": float(np.mean(r4 < -eps)),
            "p_flat_4m": float(np.mean(np.abs(r4) <= eps)),
            "avg_ret_1m": float(np.mean([x.ret_1m for x in rows])),
            "avg_ret_2m": float(np.mean([x.ret_2m for x in rows])),
            "avg_ret_4m": float(np.mean([x.ret_4m for x in rows])),
        }
