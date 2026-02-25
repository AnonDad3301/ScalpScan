from __future__ import annotations
from typing import Any, Dict, List, Tuple
import time

class CCXTMarket:
    def set_ccxt_options(self, options: Dict[str, Any]) -> None:
        try:
            # ccxt uses ex.options dict
            if isinstance(options, dict):
                self.ex.options.update(options)
        except Exception:
            pass

    def __init__(self, exchange_id: str, market: str="perp"):
        try:
            import ccxt  # type: ignore
        except Exception as e:
            raise RuntimeError("ccxt is not installed. Run: pip install ccxt") from e
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unknown exchange: {exchange_id}")
        self.exchange_id = exchange_id
        self.market = market
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, 'timeout': 10000})

    def fetch_symbols(self) -> List[Dict[str, Any]]:
        markets = self.ex.load_markets()
        tickers = {}
        try:
            if getattr(self.ex, "has", {}).get("fetchTickers", False):
                tickers = self.ex.fetch_tickers()
        except Exception:
            tickers = {}
        out=[]
        for sym, m in markets.items():
            if not isinstance(m, dict): 
                continue
            if not m.get("active", True):
                continue
            is_spot = bool(m.get("spot"))
            is_swap = bool(m.get("swap"))
            if self.market == "perp" and not is_swap:
                continue
            if self.market == "spot" and not is_spot:
                continue
            t = tickers.get(sym) if isinstance(tickers, dict) else None
            vol = float(t.get("quoteVolume") or t.get("baseVolume") or 0.0) if isinstance(t, dict) else 0.0
            out.append({"symbol": sym, "base": m.get("base"), "quote": m.get("quote"), "active": True, "volume": vol})
        return out

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Tuple[int,float,float,float,float,float]]:
        data = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])) for x in data]

    def fetch_orderbook(self, symbol: str, depth: int=50) -> Dict[str, Any]:
        ob = self.ex.fetch_order_book(symbol, limit=depth)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        spread = float(asks[0][0]) - float(bids[0][0]) if bids and asks else 0.0
        bid_vol = sum(float(v) for _, v in bids) if bids else 0.0
        ask_vol = sum(float(v) for _, v in asks) if asks else 0.0
        denom = bid_vol + ask_vol if (bid_vol+ask_vol)!=0 else 1.0
        imb = (bid_vol - ask_vol) / denom
        ts = int(ob.get("timestamp") or (time.time()*1000))
        return {"ts": ts, "depth": depth, "spread": spread, "imbalance": imb, "bid_vol": bid_vol, "ask_vol": ask_vol}

    def fetch_prices(self, symbols: List[str], price_source: str = 'last') -> Dict[str, float]:
        out={}
        try:
            if getattr(self.ex, "has", {}).get("fetchTickers", False):
                tickers = self.ex.fetch_tickers(symbols)
                if isinstance(tickers, dict):
                    for s,t in tickers.items():
                        if isinstance(t, dict) and t.get("last") is not None:
                            out[s]=float(t["last"])
                    if out:
                        return out
        except Exception:
            try:
                self._err_count += 1
                if self._err_count >= 3:
                    self._reset_exchange()
            except Exception:
                pass
            pass
        for s in symbols[:50]:
            try:
                t = self.ex.fetch_ticker(s)
                if isinstance(t, dict) and t.get("last") is not None:
                    out[s]=float(t["last"])
            except Exception:
                continue
        return out

    def _reset_exchange(self):
        try:
            self.ex = getattr(self._ccxt, self.exchange_id)(self._params)
            self._err_count = 0
        except Exception:
            try:
                self._err_count += 1
                if self._err_count >= 3:
                    self._reset_exchange()
            except Exception:
                pass
            pass

    def fetch_last_close(self, symbol: str, timeframe: str='1m') -> float:
        data = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
        if not data:
            return 0.0
        return float(data[-1][4])


def fetch_mark_price(self, symbol: str) -> float:
    """Fetch mark/last price fallback via ticker."""
    t = self.ex.fetch_ticker(symbol)
    # bybit returns info; try common fields
    for k in ("mark", "markPrice", "last", "close"):
        v = t.get(k) if isinstance(t, dict) else None
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # ccxt standard
    v = t.get("last") if isinstance(t, dict) else None
    return float(v) if v is not None else 0.0
