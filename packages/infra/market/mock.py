from __future__ import annotations
import random, time, math
from typing import Any, Dict, List, Tuple

def now_ms() -> int:
    return int(time.time()*1000)

class MockMarket:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def fetch_symbols(self) -> List[Dict[str, Any]]:
        syms = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","BNB/USDT","ADA/USDT","DOGE/USDT","AVAX/USDT","LINK/USDT","TRX/USDT"]
        return [{"symbol": s, "base": s.split('/')[0], "quote": s.split('/')[1], "active": True, "volume": random.random()*1e9} for s in syms]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Tuple[int,float,float,float,float,float]]:
        now = int(time.time()//60)*60
        base = 1000.0 + (abs(hash(symbol)) % 50000)
        price = base
        out=[]
        for i in range(limit):
            ts=(now-(limit-1-i)*60)*1000
            drift=math.sin((ts/1000)/300.0)*5.0
            shock=(random.random()-0.5)*12.0
            close=max(1.0, price+drift+shock)
            high=close+random.random()*5.0
            low=close-random.random()*5.0
            open_=price
            vol=random.random()*100.0
            out.append((ts, open_, high, low, close, vol))
            price=close
        return out

    def fetch_orderbook(self, symbol: str, depth: int=50) -> Dict[str, Any]:
        mid = 1000.0 + (abs(hash(symbol)) % 50000) + random.random()*5.0
        spread = max(0.01, random.random()*2.0)
        bid = mid - spread/2.0
        ask = mid + spread/2.0
        imb = (random.random()-0.5)*0.6
        bids = [[bid - i*0.5, 1000.0/(i+1)] for i in range(depth)]
        asks = [[ask + i*0.5, 900.0/(i+1)] for i in range(depth)]
        spread_bps = (spread / max(1e-12, mid)) * 10000.0
        return {"ts": now_ms(), "depth": depth, "spread": spread, "spread_bps": spread_bps, "imbalance": imb, "bid_vol": 1000.0, "ask_vol": 900.0, "bids": bids, "asks": asks}

    def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        # live-ish prices for mock: last close of short fetch
        out={}
        for s in symbols:
            out[s] = 1000.0 + (abs(hash(s)) % 50000) + random.random()*10.0
        return out
