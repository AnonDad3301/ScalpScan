from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from .indicators import rsi, atr, adx_wilder


def _safe_pct_rank(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    cur = float(x[-1])
    return float(np.mean(x <= cur))




def _hurst_exponent(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size < 20:
        return 0.5
    lags = np.array([2, 4, 8, 16], dtype=int)
    tau = []
    for lag in lags:
        if lag >= arr.size:
            continue
        diff = arr[lag:] - arr[:-lag]
        tau.append(float(np.std(diff)))
    if len(tau) < 2:
        return 0.5
    poly = np.polyfit(np.log(lags[:len(tau)]), np.log(np.maximum(1e-12, tau)), 1)
    return float(max(0.0, min(1.0, poly[0])))


def _lob_slope(levels, side: str) -> float:
    if not levels:
        return 0.0
    px = np.array([float(x[0]) for x in levels[:10] if len(x) >= 2], dtype=float)
    sz = np.array([float(x[1]) for x in levels[:10] if len(x) >= 2], dtype=float)
    if px.size < 2:
        return 0.0
    y = np.log(np.maximum(1e-9, sz))
    x = np.arange(px.size, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope if side == "bid" else -slope)



def _rolling_vwap(close: np.ndarray, vol: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    pv = np.cumsum(close * np.maximum(1e-12, vol))
    vv = np.cumsum(np.maximum(1e-12, vol))
    return pv / np.maximum(1e-12, vv)

def compute(ohlcv: List[Tuple[int, float, float, float, float, float]], ob: Dict[str, Any]) -> Dict[str, float]:
    arr = np.array([[o, h, l, c, v] for _, o, h, l, c, v in ohlcv], dtype=float)
    high = arr[:, 1]
    low = arr[:, 2]
    close = arr[:, 3]
    vol = arr[:, 4]
    last = float(close[-1]) if len(close) else 0.0
    ret_last = float((close[-1] - close[-2]) / max(1e-12, close[-2])) if len(close) >= 2 else 0.0
    range_last = float((high[-1] - low[-1]) / max(1e-12, last)) if len(close) else 0.0
    rets = np.diff(close) / np.maximum(1e-12, close[:-1]) if len(close) >= 3 else np.array([], dtype=float)
    vol_z_last = 0.0
    if len(rets) >= 10:
        mu = float(np.mean(rets[-10:]))
        sd = float(np.std(rets[-10:]))
        vol_z_last = float((ret_last - mu) / max(1e-12, sd))

    spread_raw = float(ob.get("spread", 0.0) or 0.0)
    spread_bps = float(ob.get("spread_bps", 0.0) or 0.0)
    if spread_bps <= 0.0 and last > 0.0:
        spread_bps = abs(spread_raw) / last * 10000.0

    atr_v = atr(high, low, close, 14)
    atr_roll = np.array([atr(high[max(0, i - 30): i + 1], low[max(0, i - 30): i + 1], close[max(0, i - 30): i + 1], 14) for i in range(len(close))], dtype=float)
    atr_percentile = _safe_pct_rank(atr_roll[-80:]) if len(atr_roll) else 0.0

    vwap = float(np.sum(close * vol) / max(1e-12, np.sum(vol))) if len(close) else last
    vwap_dist = float((last - vwap) / max(1e-12, vwap))
    vwap_slope = float((vwap - float(np.mean(close[-10:]))) / max(1e-12, vwap)) if len(close) >= 10 else 0.0

    body = float(close[-1] - arr[-1, 0]) if len(close) else 0.0
    up_wick = float(high[-1] - max(arr[-1, 0], close[-1])) if len(close) else 0.0
    low_wick = float(min(arr[-1, 0], close[-1]) - low[-1]) if len(close) else 0.0
    wick_asym = float((up_wick - low_wick) / max(1e-12, abs(body) + up_wick + low_wick))

    realized_vol = float(np.std(rets[-30:])) if len(rets) >= 5 else 0.0
    hurst = _hurst_exponent(np.log(np.maximum(1e-12, close))) if len(close) else 0.5
    entropy = 0.0
    if len(rets) >= 20:
        bins = np.histogram(rets[-60:], bins=10)[0].astype(float)
        probs = bins / max(1e-12, bins.sum())
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log(probs)).sum())

    # microstructure from provided snapshot (fallback to defaults)
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []

    def lvl_imb(n: int) -> float:
        b = sum(float(x[1]) for x in bids[:n] if len(x) >= 2)
        a = sum(float(x[1]) for x in asks[:n] if len(x) >= 2)
        return float((b - a) / max(1e-12, b + a))

    l1_imb = float(ob.get("imbalance_l1", lvl_imb(1)))
    l5_imb = float(ob.get("imbalance_l5", lvl_imb(5)))
    l10_imb = float(ob.get("imbalance_l10", lvl_imb(10)))
    best_bid = float(bids[0][0]) if bids else (last - spread_raw / 2.0)
    best_ask = float(asks[0][0]) if asks else (last + spread_raw / 2.0)
    microprice = float((best_ask * max(1e-12, (1 + l1_imb)) + best_bid * max(1e-12, (1 - l1_imb))) / 2.0)
    microprice_delta_bps = float((microprice - last) / max(1e-12, last) * 10000.0)
    lob_slope_bid = _lob_slope(bids, "bid")
    lob_slope_ask = _lob_slope(asks, "ask")
    queue_imbalance_dynamics = float(ob.get("queue_imbalance_dynamics", l1_imb - l5_imb))
    volume_delta = float(ob.get("volume_delta", ret_last * float(vol[-1] if len(vol) else 0.0)))
    volume_imbalance = float(ob.get("volume_imbalance", (volume_delta / max(1e-12, abs(volume_delta) + float(vol[-1] if len(vol) else 1.0)))))

    # EPIC features: AVWAP / squeeze / sweep / CVD proxy
    avwap = _rolling_vwap(close, vol)
    avwap_dist = float((last - float(avwap[-1])) / max(1e-12, float(avwap[-1]))) if avwap.size else 0.0

    squeeze_on = 0.0
    if len(close) >= 22:
        mid = float(np.mean(close[-20:]))
        std = float(np.std(close[-20:]))
        bb_w = max(1e-12, (mid + 2.0 * std) - (mid - 2.0 * std))
        tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
        kc_w = max(1e-12, 2.0 * float(np.mean(tr[-20:])))
        squeeze_on = 1.0 if bb_w < kc_w else 0.0

    sweep = 0.0
    if len(close) >= 22:
        prev_h = float(np.max(high[-21:-1]))
        prev_l = float(np.min(low[-21:-1]))
        if float(high[-1]) > prev_h and float(close[-1]) < prev_h:
            sweep = -1.0
        elif float(low[-1]) < prev_l and float(close[-1]) > prev_l:
            sweep = 1.0

    bid_sz = sum(float(x[1]) for x in bids[:20] if len(x) >= 2)
    ask_sz = sum(float(x[1]) for x in asks[:20] if len(x) >= 2)
    cvd_proxy = float((bid_sz - ask_sz) / max(1e-12, bid_sz + ask_sz))

    return {
        "rsi": rsi(close, 14),
        "atr": atr_v,
        "adx": adx_wilder(high, low, close, 14),
        "spread": spread_raw,
        "spread_bps": spread_bps,
        "spread_pct_rank": float(ob.get("spread_pct_rank", 0.0) or 0.0),
        "ob_imb": float(ob.get("imbalance", 0.0) or 0.0),
        "ob_imb_l1": l1_imb,
        "ob_imb_l5": l5_imb,
        "ob_imb_l10": l10_imb,
        "microprice_delta_bps": microprice_delta_bps,
        "wall_persistence": float(ob.get("wall_persistence", 0.0) or 0.0),
        "cancel_add_ratio": float(ob.get("cancel_add_ratio", 0.0) or 0.0),
        "orderflow_delta": float(ob.get("orderflow_delta", 0.0) or 0.0),
        "lob_slope_bid": lob_slope_bid,
        "lob_slope_ask": lob_slope_ask,
        "queue_imbalance_dynamics": queue_imbalance_dynamics,
        "volume_delta": volume_delta,
        "volume_imbalance": volume_imbalance,
        "ret_last": ret_last,
        "range_last": range_last,
        "vol_z_last": vol_z_last,
        "volz": abs(vol_z_last),
        "vwap_dist": vwap_dist,
        "vwap_slope": vwap_slope,
        "atr_percentile": atr_percentile,
        "wick_asymmetry": wick_asym,
        "realized_vol": realized_vol,
        "entropy_returns": entropy,
        "hurst": hurst,
        "avwap_dist": avwap_dist,
        "squeeze_on": squeeze_on,
        "liquidity_sweep": sweep,
        "cvd_proxy": cvd_proxy,
    }
