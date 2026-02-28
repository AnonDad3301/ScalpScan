from __future__ import annotations
import multiprocessing as mp
import traceback
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _worker_main(req_q: mp.Queue, resp_q: mp.Queue, cfg: Dict[str, Any]) -> None:
    """Runs in a separate process to isolate ccxt hard-hangs on Windows."""
    try:
        from packages.infra.market.ccxt_market import CCXTMarket
        rt = cfg.get("runtime", {})
        ex_id = rt.get("exchange", "bybit")
        mkt = rt.get("market", "perp")
        market = CCXTMarket(ex_id, market=mkt)
        resp_q.put({"type":"ready","ts": int(time.time()*1000), "exchange": ex_id, "market": mkt})
    except Exception as e:
        resp_q.put({"type":"fatal","err":repr(e),"tb":traceback.format_exc()})
        return

    while True:
        try:
            msg = req_q.get()
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        cmd = msg.get("cmd")
        if cmd == "stop":
            resp_q.put({"type":"stopped","ts": int(time.time()*1000)})
            return
        if cmd == "ping":
            resp_q.put({"type":"pong","ts": int(time.time()*1000)})
            continue
        if cmd == "fetch_prices":
            syms = msg.get("symbols") or []
            try:
                prices = market.fetch_prices(list(syms))
                resp_q.put({"type":"prices","ts": int(time.time()*1000), "prices": prices})
            except Exception as e:
                try:
                    market._reset_exchange()  # type: ignore
                except Exception:
                    pass
                resp_q.put({"type":"error","ts": int(time.time()*1000), "err":repr(e),"tb":traceback.format_exc()})
        elif cmd == "fetch_mark_prices":
            syms = msg.get("symbols") or []
            out={}
            try:
                for s in syms:
                    try:
                        px = market.fetch_mark_price(s)
                        if px:
                            out[s]=float(px)
                    except Exception:
                        continue
                resp_q.put({"type":"mark_prices","ts": int(time.time()*1000), "prices": out})
            except Exception as e:
                resp_q.put({"type":"error","ts": int(time.time()*1000), "err":repr(e),"tb":traceback.format_exc()})
        
        elif cmd == "fetch_last_close":
            sym = msg.get("symbol")
            tf = msg.get("timeframe","1m")
            try:
                px = market.fetch_last_close(sym, timeframe=tf)
                resp_q.put({"type":"last_close","ts": int(time.time()*1000), "symbol":sym,"price":px})
            except Exception as e:
                resp_q.put({"type":"error","ts": int(time.time()*1000), "err":repr(e),"tb":traceback.format_exc()})

class PriceWorker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.req_q: mp.Queue = mp.Queue()
        self.resp_q: mp.Queue = mp.Queue()
        self.proc: mp.Process | None = None

    def start(self) -> None:
        if self.proc and self.proc.is_alive():
            return
        self.proc = mp.Process(target=_worker_main, args=(self.req_q, self.resp_q, self.cfg), daemon=True)
        self.proc.start()

    def stop(self) -> None:
        try:
            self.req_q.put({"cmd":"stop"})
        except Exception:
            pass
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=2)

    def restart(self) -> None:
        self.stop()
        self.start()

    def wait_ready(self, timeout_sec: float = 3.0) -> Dict[str, Any]:
        self.start()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                msg = self.resp_q.get(timeout=0.1)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") in ("ready","fatal"):
                return msg
        return {"type":"timeout"}

    def fetch_prices(self, symbols: List[str], timeout_sec: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        self.start()
        try:
            self.req_q.put({"cmd":"fetch_prices","symbols":symbols})
        except Exception as e:
            return {}, {"status":"ERROR","err":repr(e)}
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                msg = self.resp_q.get(timeout=0.05)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "prices":
                return (msg.get("prices") or {}), {"status":"OK","ts":msg.get("ts")}
            if msg.get("type") == "fatal":
                return {}, {"status":"FATAL","err":msg.get("err"),"tb":msg.get("tb")}
            if msg.get("type") == "error":
                return {}, {"status":"ERROR","err":msg.get("err"),"tb":msg.get("tb")}
            # ignore other messages (ready/pong)
        # timeout: restart to kill potential hang
        self.restart()
        return {}, {"status":"TIMEOUT"}

    def fetch_mark_prices(self, symbols: List[str], timeout_sec: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        self.start()
        try:
            self.req_q.put({"cmd":"fetch_mark_prices","symbols":symbols})
        except Exception as e:
            return {}, {"status":"ERROR","err":repr(e)}
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                msg = self.resp_q.get(timeout=0.05)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "mark_prices":
                return (msg.get("prices") or {}), {"status":"OK","ts":msg.get("ts")}
            if msg.get("type") == "fatal":
                return {}, {"status":"FATAL","err":msg.get("err"),"tb":msg.get("tb")}
            if msg.get("type") == "error":
                return {}, {"status":"ERROR","err":msg.get("err"),"tb":msg.get("tb")}
        self.restart()
        return {}, {"status":"TIMEOUT"}
