from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import socket

@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str

def _port_status(host: str, port: int) -> Tuple[bool, str]:
    """Return (ok, details).
    OK if:
      - port is free (bind works), OR
      - port is already listening (connect works) — this is acceptable when hub is already running.
    FAIL if:
      - port is in use but NOT listening (bind fails and connect fails).
    """
    # 1) if something is listening, connect succeeds => OK
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True, "listening"
    except Exception:
        pass
    # 2) else check if we can bind => free => OK
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.bind((host, port))
            return True, "free"
    except OSError as e:
        return False, f"in_use_not_listening:{e}"

def validate_config(cfg: Dict[str, Any]) -> List[CheckResult]:
    res: List[CheckResult] = []
    rt = cfg.get("runtime", {})
    ex = cfg.get("execution", {})
    exits = ex.get("exits", {})

    tick = float(rt.get("tick_interval_sec", 2.0))
    price = float(rt.get("price_interval_sec", 1.0))
    res.append(CheckResult("tick_interval_sec>=0.1", tick >= 0.1, f"{tick}"))
    res.append(CheckResult("price_interval_sec>=0.1", price >= 0.1, f"{price}"))

    host = str(rt.get("ws_host", "127.0.0.1"))
    port = int(rt.get("ws_port", 8765))
    ok, details = _port_status(host, port)
    res.append(CheckResult("ws_port_ready", ok, f"{host}:{port} ({details})"))

    n = int(rt.get("max_pairs", 50))
    res.append(CheckResult("max_pairs>=1", n >= 1, f"{n}"))

    tp1_frac = float(exits.get("tp1_fraction", 0.5))
    res.append(CheckResult("tp1_fraction_in_(0,1]", 0.0 < tp1_frac <= 1.0, f"{tp1_frac}"))
    for k in ("sl_atr_mult","tp1_atr_mult","tp2_atr_mult"):
        v = float(exits.get(k, 1.0))
        res.append(CheckResult(f"{k}>0", v > 0.0, f"{v}"))

    dep = float(ex.get("deposit_usd", 0.0))
    risk = float(ex.get("risk_usd", 0.0))
    res.append(CheckResult("deposit_usd>0", dep > 0.0, f"{dep}"))
    res.append(CheckResult("risk_usd>0", risk > 0.0, f"{risk}"))
    return res

def preflight(cfg: Dict[str, Any], market, symbol_store) -> Tuple[bool, List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    for r in validate_config(cfg):
        results.append({"name": r.name, "ok": r.ok, "details": r.details})

    rt = cfg.get("runtime", {})
    if str(rt.get("adapter","mock")) == "ccxt":
        try:
            import ccxt  # noqa: F401
            results.append({"name":"ccxt_import", "ok": True, "details":"ok"})
        except Exception as e:
            results.append({"name":"ccxt_import", "ok": False, "details": str(e)})

    # symbols
    try:
        rows = market.fetch_symbols()
        ok = isinstance(rows, list) and len(rows) > 0
        results.append({"name":"fetch_symbols", "ok": ok, "details": f"count={len(rows) if isinstance(rows,list) else 'n/a'}"})
        if ok:
            exchange = str(rt.get("exchange",""))
            mkt = str(rt.get("market",""))
            n = symbol_store.upsert_many(exchange, mkt, rows)
            results.append({"name":"symbols_db_upsert", "ok": True, "details": f"upserted={n}"})
    except Exception as e:
        results.append({"name":"fetch_symbols", "ok": False, "details": str(e)})

    # ohlcv sample
    try:
        exchange = str(rt.get("exchange",""))
        mkt = str(rt.get("market",""))
        top = symbol_store.top_symbols(exchange, mkt, 1, quote=cfg.get("universe",{}).get("quote"))
        sym = top[0] if top else (cfg.get("universe",{}).get("custom_symbols") or ["BTC/USDT"])[0]
        ohlcv = market.fetch_ohlcv(sym, str(rt.get("timeframe","1m")), int(rt.get("lookback",300)))
        ok = isinstance(ohlcv, list) and len(ohlcv) >= 50
        results.append({"name":"fetch_ohlcv_sample", "ok": ok, "details": f"{sym} rows={len(ohlcv) if isinstance(ohlcv,list) else 'n/a'}"})
    except Exception as e:
        results.append({"name":"fetch_ohlcv_sample", "ok": False, "details": str(e)})

    ok_all = all(bool(x["ok"]) for x in results)
    return ok_all, results
