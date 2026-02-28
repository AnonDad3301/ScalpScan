from __future__ import annotations

from typing import Tuple

import numpy as np


def triple_barrier_labels(
    close: np.ndarray,
    atr: np.ndarray,
    horizon_bars: int = 5,
    tp_atr_mult: float = 1.2,
    sl_atr_mult: float = 1.0,
    costs_bps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels based on TP-first / SL-first with time barrier.

    y_dir: 1 (TP first), -1 (SL first), 0 (time barrier / unresolved)
    p_tp_proxy/p_sl_proxy are binary proxies for online compatibility.
    """
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr, dtype=float)
    n = len(c)
    y = np.zeros(n, dtype=int)
    p_tp = np.zeros(n, dtype=float)
    p_sl = np.zeros(n, dtype=float)
    if n == 0:
        return y, p_tp, p_sl

    hb = max(1, int(horizon_bars))
    for i in range(0, max(0, n - hb)):
        px0 = float(c[i])
        atr_i = max(1e-12, float(a[i]) if i < len(a) else 0.0)
        cost_px = px0 * float(max(0.0, costs_bps)) / 10000.0
        up = px0 + tp_atr_mult * atr_i + cost_px
        dn = px0 - sl_atr_mult * atr_i - cost_px
        label = 0
        for j in range(i + 1, min(n, i + hb + 1)):
            px = float(c[j])
            if px >= up:
                label = 1
                break
            if px <= dn:
                label = -1
                break
        y[i] = label
        p_tp[i] = 1.0 if label == 1 else 0.0
        p_sl[i] = 1.0 if label == -1 else 0.0
    return y, p_tp, p_sl

