from __future__ import annotations
import numpy as np

def rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 2:
        return 50.0
    diff = np.diff(close)
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:]) + 1e-12
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(out, 0.0, 100.0))

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 2:
        return float(max(np.mean(high-low), 1e-6))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high-low, np.maximum(np.abs(high-prev_close), np.abs(low-prev_close)))
    out = float(np.mean(tr[-period:]))
    return max(out, 1e-12)

def adx_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    n = len(close)
    if n < period*2 + 2:
        return 10.0
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

    def wilder_smooth(x):
        out = np.zeros_like(x)
        out[period-1] = np.sum(x[:period])
        for i in range(period, len(x)):
            out[i] = out[i-1] - (out[i-1]/period) + x[i]
        return out

    tr_s = wilder_smooth(tr)
    p_s = wilder_smooth(plus_dm)
    m_s = wilder_smooth(minus_dm)
    tr_s = np.where(tr_s==0, 1e-12, tr_s)
    pdi = 100.0 * (p_s / tr_s)
    mdi = 100.0 * (m_s / tr_s)
    dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)

    adx = np.zeros_like(dx)
    start = period*2 - 2
    adx[start] = np.mean(dx[period-1:start+1])
    for i in range(start+1, len(dx)):
        adx[i] = ((adx[i-1]*(period-1)) + dx[i]) / period
    return float(np.clip(adx[-1], 0.0, 100.0))
