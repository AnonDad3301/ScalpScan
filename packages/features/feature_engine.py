from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

def _safe(arr, default=0.0):
    try:
        a=np.asarray(arr, dtype=float)
        a[np.isnan(a)]=default
        return a
    except Exception:
        return np.asarray([default], dtype=float)

def ohlcv_to_channels(ohlcv: List[List[float]], lookback: int = 128) -> Dict[str, np.ndarray]:
    """Return dict of channels length L (lookback)."""
    a=np.asarray(ohlcv, dtype=float)
    if a.ndim!=2 or a.shape[0]<max(lookback, 10):
        # pad
        L=min(a.shape[0], lookback)
        close=a[-L:,4] if a.shape[0]>0 else np.zeros(lookback)
    # select last lookback
    a=a[-lookback:]
    o,h,l,c,v=a[:,1],a[:,2],a[:,3],a[:,4],a[:,5]
    # returns
    ret=np.zeros_like(c)
    ret[1:]=np.log(np.maximum(c[1:],1e-12)/np.maximum(c[:-1],1e-12))
    # range/body/wicks
    rng=(h-l)/np.maximum(c,1e-12)
    body=(c-o)/np.maximum(o,1e-12)
    uw=(h-np.maximum(o,c))/np.maximum(c,1e-12)
    lw=(np.minimum(o,c)-l)/np.maximum(c,1e-12)
    # vol zscore
    vol=v
    n=20
    ma=np.convolve(vol, np.ones(n)/n, mode='same')
    sd=np.sqrt(np.convolve((vol-ma)**2, np.ones(n)/n, mode='same')+1e-12)
    vol_z=(vol-ma)/sd
    # dollar volume
    dvol=c*vol
    return {
        "ret": _safe(ret),
        "range": _safe(rng),
        "body": _safe(body),
        "uw": _safe(uw),
        "lw": _safe(lw),
        "vol_z": _safe(vol_z),
        "dvol": _safe(dvol/np.maximum(np.nanmean(dvol),1e-12)),
        "close": _safe(c),
        "vol": _safe(vol/np.maximum(np.nanmean(vol),1e-12)),
    }

def pack_model_b_tensor(ch: Dict[str, np.ndarray], ob: Dict[str, float], lookback: int = 128) -> np.ndarray:
    """Return tensor shape [C, L] for PatchTST."""
    # Basic channels + orderbook scalar channels broadcast
    keys=["ret","range","body","uw","lw","vol_z","dvol"]
    mats=[ch[k][-lookback:] for k in keys]
    L=len(mats[0])
    # broadcast scalars
    for sk in ["imb","spread_bps","wall_dist_bps","wall_age","wall_touches"]:
        val=float(ob.get(sk,0.0))
        mats.append(np.full(L, val, dtype=float))
    X=np.vstack(mats).astype(np.float32)
    return X
