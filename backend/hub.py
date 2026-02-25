from __future__ import annotations
import asyncio, json, time, logging, datetime, faulthandler, uuid
from typing import Any, Dict, Set

import websockets
from websockets.legacy.server import serve as legacy_serve
from websockets.exceptions import ConnectionClosed

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.common.config import Config
from packages.common.preflight import preflight
from packages.infra.event_store.sqlite_store import SQLiteEventStore
from packages.infra.trades_store.sqlite_trades import TradesStore
from packages.infra.snapshot_store.files import FileSnapshotStore
from packages.infra.symbol_store.sqlite_symbols import SymbolStore
from packages.infra.market.mock import MockMarket
from packages.infra.market.ccxt_market import CCXTMarket
from apps.runtime.engine import Engine
from backend.bybit_ws_prices import BybitPublicWS
from backend.price_worker import PriceWorker
from packages.obs.telemetry import Telemetry


logging.getLogger("websockets").disabled = True
logging.getLogger("websockets.server").disabled = True
logging.getLogger("websockets.http11").disabled = True

clients: Set[Any] = set()
STATE: Dict[str, Any] = {"running": False, "preflight_ok": False, "preflight_report": []}

tele: Telemetry | None = None


def now_ms() -> int:
    return int(time.time() * 1000)



def log_line(path: str, line: str) -> None:
    try:
        from pathlib import Path as _P
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass

def _safe_json(obj: Any) -> str:

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


async def broadcast(msg: Dict[str, Any]) -> None:
    if not clients:
        return
    data = _safe_json(msg)
    dead = []
    for ws in list(clients):
        try:
            await ws.send(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


class CommandQueue:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()

    async def enqueue(self, msg: Dict[str, Any]):
        await self.q.put(msg)

    async def pump(self, eng: Engine, cfg: Dict[str, Any], market, symdb: SymbolStore):
        while True:
            msg = await self.q.get()
            cmd = str(msg.get("cmd", ""))
            try:
                if cmd == "run_preflight":
                    ok, rep = preflight(cfg, market, symdb)
                    STATE["preflight_ok"] = ok
                    STATE["preflight_report"] = rep
                    await broadcast({"type": "preflight", "ts": now_ms(), "ok": ok, "report": rep})
                elif cmd == "start":
                    if not STATE.get("preflight_ok", False):
                        await broadcast({"type": "status", "ts": now_ms(), "status": "start_blocked:preflight_not_ok"})
                        continue
                    STATE["running"] = True
                    await broadcast({"type": "status", "ts": now_ms(), "status": "running"})
                elif cmd == "stop":
                    STATE["running"] = False
                    await broadcast({"type": "status", "ts": now_ms(), "status": "stopped"})
                elif cmd == "sync_symbols":
                    n = eng.sync_symbols()
                    await broadcast({"type": "status", "ts": now_ms(), "status": f"symbols_synced:{n}"})
                elif cmd == "get_symbols":
                    rt = cfg.get("runtime", {})
                    uni = cfg.get("universe", {})
                    exchange = str(rt.get("exchange", ""))
                    mkt = str(rt.get("market", ""))
                    limit = int(msg.get("limit", 200))
                    top = symdb.top_symbols(exchange, mkt, limit, quote=uni.get("quote"), min_volume=float(uni.get("min_volume", 0)))
                    await broadcast({"type": "symbols", "ts": now_ms(), "exchange": exchange, "market": mkt, "count": len(top), "symbols": top})
            except Exception as e:
                await broadcast({"type": "status", "ts": now_ms(), "status": f"cmd_error:{cmd}:{e}"})


COMMANDS = CommandQueue()


async def handler(ws):
    clients.add(ws)
    try:
        await ws.send(_safe_json({"type": "hello", "ts": now_ms()}))
        await ws.send(_safe_json({"type": "preflight", "ts": now_ms(), "ok": STATE["preflight_ok"], "report": STATE["preflight_report"]}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "cmd":
                await COMMANDS.enqueue(msg)
    except ConnectionClosed:
        pass
    except Exception:
        pass
    finally:
        clients.discard(ws)


def append_event(es, rt: Dict[str, Any], stage: str, payload: Dict[str, Any], run_id: str="sys", level: str="INFO", symbol: str="*"):
    if stage == "TRADE_CLOSED":
        try:
            trades_store.upsert(payload)
        except Exception:
            pass
    es.append({
        "event_id": str(uuid.uuid4()),
        "ts": now_ms(),
        "run_id": run_id,
        "service": "hub",
        "exchange": rt.get("exchange"),
        "market": rt.get("market"),
        "symbol": "*",
        "timeframe": rt.get("timeframe"),
        "stage": stage,
        "level": level,
        "payload": payload,
    })


async def loops(cfg: Dict[str, Any], eng: Engine, es: SQLiteEventStore, market, tele, price_worker, ws_client, trades_store) -> None:
    rt = cfg.get("runtime", {})
    tick_int = float(rt.get("tick_interval_sec", 2.0))
    price_int = float(rt.get("price_interval_sec", 1.0))
    tick_timeout = float(rt.get("tick_timeout_sec", 25.0))
    price_timeout = float(rt.get("price_timeout_sec", 1.0))
    ws_resub_ms = int(rt.get("ws_stale_resub_ms", 3000))
    ws_reconnect_ms = int(rt.get("ws_stale_reconnect_ms", 10000))
    ws_zero_ticks_limit = int(rt.get("ws_zero_ticks_limit", 5))
    ws_zero_ticks = 0

    stall_sec = float(rt.get("watchdog_stall_sec", 20.0))
    wd_int = float(rt.get("watchdog_interval_sec", 1.0))
    liveness = {"tick": time.monotonic(), "price": time.monotonic(), "events": time.monotonic()}
    last_prices: Dict[str, float] = {}

    async def tick_loop():
        while True:
            if STATE.get("running", False) and STATE.get("preflight_ok", False):
                t0 = time.time()
                try:

                    try:

                        eng.runtime_flags = {

                            "force_model_b_retrain": bool(STATE.get("force_model_b_retrain", False)),

                            "disable_model_b_training": bool(STATE.get("disable_model_b_training", False)),

                            "rollback_model_b": bool(STATE.get("rollback_model_b", False)),

                        }

                    except Exception:

                        pass

                    run_id = await asyncio.wait_for(asyncio.to_thread(eng.tick), timeout=tick_timeout)


                    STATE['force_model_b_retrain']=False


                    STATE['rollback_model_b']=False
                    liveness["tick"] = time.monotonic()
                    append_event(es, rt, "TICK_DONE", {"run_id": run_id, "dt_ms": int((time.time()-t0)*1000)}, run_id="tick")
                except asyncio.TimeoutError:
                    liveness["tick"] = time.monotonic()
                    append_event(es, rt, "TICK_TIMEOUT", {"timeout_sec": tick_timeout}, run_id="tick", level="ERROR")
                except Exception as e:
                    liveness["tick"] = time.monotonic()
                    append_event(es, rt, "TICK_ERROR", {"err": str(e)}, run_id="tick", level="ERROR")
            await asyncio.sleep(max(0.05, tick_int))

    async def price_loop():
        while True:
            t0 = time.time()
            syms = list(getattr(eng.portfolio, "positions", {}).keys())
            try:
                await ws_client.set_desired(syms)
            except Exception:
                pass
            # auto-fix WS pong-only
            try:
                if (ws_ticker_1s is not None) and int(ws_ticker_1s) == 0:
                    ws_zero_ticks += 1
                else:
                    ws_zero_ticks = 0
                if (ws_stale_ms is not None) and int(ws_stale_ms) >= ws_resub_ms:
                    await ws_client.request_resubscribe()
                if (ws_stale_ms is not None) and int(ws_stale_ms) >= ws_reconnect_ms:
                    await ws_client.request_reconnect()
                if ws_zero_ticks >= ws_zero_ticks_limit:
                    await ws_client.request_reconnect()
                    ws_zero_ticks = 0
            except Exception:
                pass
            try:
                await ws_client.set_desired(syms)
            except Exception:
                pass
            prices: Dict[str, float] = {}
            status = "OK"
            try:
                if syms:
                    # primary: WS cache
                    try:
                        ws_prices, ws_state = await ws_client.get_prices()
                        for s in syms:
                            if s in ws_prices:
                                prices[s] = float(ws_prices[s])
                    except Exception:
                        ws_state = None
                    stale = True
                    try:
                        if ws_state and ws_state.connected:
                            stale = (int(time.time()*1000) - int(ws_state.last_msg_ms)) > 3000
                    except Exception:
                        stale = True
                    missing = [s for s in syms if s not in prices]
                    if missing or stale:
                        prices2, pm = price_worker.fetch_prices(syms, timeout_sec=price_timeout)
                        if isinstance(prices2, dict):
                            for k,v in prices2.items():
                                try:
                                    prices[k]=float(v)
                                except Exception:
                                    pass
                        if pm.get('status') != 'OK':
                            append_event(es, rt, 'PRICE_WORKER', pm, run_id='price', level='ERROR')
                    if ws_state is not None:
                        append_event(es, rt, 'WS_STATE', {
                            'connected': ws_state.connected,
                            'last_msg_ms': ws_state.last_msg_ms,
                            'last_pong_ms': ws_state.last_pong_ms,
                            'url': ws_state.conn_url,
                            'stale': stale,
                            'missing_n': len(missing),
                        }, run_id='price')
                    prices, pm = price_worker.fetch_prices(syms, timeout_sec=price_timeout)
                if not isinstance(prices, dict):
                    prices = {}
                if pm.get("status") != "OK":
                    append_event(es, rt, "PRICE_WORKER", pm, run_id="price", level="ERROR")


                # fallback: if some symbols missing, use last_prices or last 1m close
                missing = [s for s in syms if s not in prices]
                if missing:
                    for s in missing:
                        if s in last_prices:
                            prices[s] = float(last_prices[s])
                    still = [s for s in missing if s not in prices]
                    if still:
                        for s in still:
                            try:
                                px = await asyncio.wait_for(asyncio.to_thread(market.fetch_last_close, s, "1m"), timeout=1.5)
                                if px:
                                    prices[s] = float(px)
                            except Exception:
                                continue
                for k, v in prices.items():
                    try:
                        last_prices[k] = float(v)
                    except Exception:
                        pass

                # exit manager + equity
                try:
                    updates, closed = eng.portfolio.manage_positions(now_ms(), prices)
                    for u in updates:
                        append_event(es, rt, "POSITION_UPDATE", u, run_id="price")
                    for c in closed:
                        append_event(es, rt, "TRADE_CLOSED", c, run_id="price")
                        try:
                            trades_store.append_closed(c)
                        except Exception as e:
                            append_event(es, rt, "TRADES_STORE_ERROR", {"err": str(e)}, run_id="price", level="ERROR")
                        try:
                            eng.trainer.on_trade_closed(c)
                        except Exception:
                            pass
                except Exception as e:
                    append_event(es, rt, "ERROR", {"where": "manage_positions", "err": str(e)}, run_id="price", level="ERROR")

                try:
                    eng.portfolio.update_equity(prices)
                except Exception:
                    pass

                # snapshots for UI
                append_event(es, rt, "PRICES_SNAPSHOT", {"prices": prices}, run_id="price")
                acc = eng.portfolio.account
                append_event(es, rt, "ACCOUNT", {"deposit": acc.deposit, "cash": acc.cash, "equity": acc.equity, "open_positions": len(eng.portfolio.positions)}, run_id="price")
                pos_payload = {"positions": {k: {"symbol": v.symbol, "side": v.side, "entry": v.entry, "qty": v.qty, "sl": v.sl, "tp1": v.tp1, "tp2": v.tp2, "open_ts": v.open_ts} for k, v in eng.portfolio.positions.items()}}
                append_event(es, rt, "POSITIONS_SNAPSHOT", pos_payload, run_id="price")
            except asyncio.TimeoutError:
                status = "TIMEOUT"
                append_event(es, rt, "PRICE_TIMEOUT", {"timeout_sec": price_timeout, "open_positions": len(syms)}, run_id="price", level="ERROR")
            except Exception as e:
                status = "ERROR"
                append_event(es, rt, "PRICE_ERROR", {"err": str(e)}, run_id="price", level="ERROR")

            dt_ms = int((time.time()-t0)*1000)
            liveness["price"] = time.monotonic()
            # WS diagnostics
            ws_connected=False; ws_stale_ms=None; ws_ticker_1s=None; ws_sub_ok=None; ws_sub_err=None
            try:
                _, st = await ws_client.get_prices()
                ws_connected = bool(st.connected)
                ws_ticker_1s = int(st.ticker_updates_1s)
                ws_sub_ok = int(st.sub_ok)
                ws_sub_err = int(st.sub_err)
                ws_stale_ms = int(time.time()*1000) - int(st.last_msg_ms)
            except Exception:
                pass
            append_event(es, rt, "PRICE_TICK", {"open_positions": len(syms), "prices_n": len(prices), "dt_ms": dt_ms, "status": status, "symbols": syms[:50], "ws_connected": ws_connected, "ws_stale_ms": ws_stale_ms, "ws_ticker_1s": ws_ticker_1s, "ws_sub_ok": ws_sub_ok, "ws_sub_err": ws_sub_err}, run_id="price")

            try:
                sp = tele.span_start("PRICE_TICK", trace_id="price_tick", symbol="*", open_positions=len(syms))
                tele.span_end(sp, status=status, prices_n=len(prices), dt_ms=dt_ms)
            except Exception:
                pass

            await asyncio.sleep(max(0.25, price_int))

    async def events_push_loop():
        while True:
            tail = es.tail(500)
            interesting = [e for e in tail if e["stage"] in (
                "SIGNAL","TRADE_OPEN","TRADE_CLOSED","POSITION_UPDATE","ERROR",
                "PRICE_TICK","PRICE_TIMEOUT","PRICE_ERROR",
                "TICK_TIMEOUT","TICK_ERROR","TICK_DONE",
                "ACCOUNT","PRICES_SNAPSHOT","POSITIONS_SNAPSHOT",
                "MODEL_INFERRED","GATE_DECISION","STALL_DETECTED"
            )]
            liveness["events"] = time.monotonic()
            await broadcast({"type": "events", "ts": now_ms(), "events": list(reversed(interesting))[-350:]})
            await asyncio.sleep(1.0)


    async def cmd_loop():
        """Poll data/ui_commands.jsonl and set one-shot or sticky flags."""
        cmd_path = str(Path(__file__).resolve().parents[1] / "data" / "ui_commands.jsonl")
        last_pos = 0
        while True:
            try:
                if os.path.exists(cmd_path):
                    with open(cmd_path, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            line=line.strip()
                            if not line:
                                continue
                            try:
                                cmd=json.loads(line)
                            except Exception:
                                continue
                            cid=int(cmd.get("id",0) or 0)
                            if cid <= int(STATE.get("last_cmd_id",0) or 0):
                                continue
                            STATE["last_cmd_id"]=cid
                            c=cmd.get("cmd")
                            if c=="force_model_b_retrain":
                                STATE["force_model_b_retrain"]=True
                                append_event(es, rt, "CMD", {"cmd": c}, run_id="cmd", level="INFO")
                            elif c=="disable_model_b_training":
                                STATE["disable_model_b_training"]=True
                                append_event(es, rt, "CMD", {"cmd": c}, run_id="cmd", level="INFO")
                            elif c=="enable_model_b_training":
                                STATE["disable_model_b_training"]=False
                                append_event(es, rt, "CMD", {"cmd": c}, run_id="cmd", level="INFO")
                            elif c=="rollback_model_b":
                                STATE["rollback_model_b"]=True
                                append_event(es, rt, "CMD", {"cmd": c}, run_id="cmd", level="INFO")
                        last_pos=f.tell()
            except Exception as e:
                append_event(es, rt, "CMD_ERROR", {"err": str(e)}, run_id="cmd", level="ERROR")
            await asyncio.sleep(1.0)


    async def watchdog_loop():
        log_path = str(Path(__file__).resolve().parents[1] / "data" / "watchdog.log")
        dump_path = str(Path(__file__).resolve().parents[1] / "data" / "watchdog_dump.txt")
        while True:
            nowm = time.monotonic()
            ages = {k: nowm - float(v) for k, v in liveness.items()}
            log_line(log_path, _safe_json({"ts": now_ms(), "kind": "WD_HEARTBEAT", "ages": ages}))
            stalled = {k: a for k, a in ages.items() if a > stall_sec}
            if stalled:
                append_event(es, rt, "STALL_DETECTED", {"stall_sec": stall_sec, "stalled": stalled}, run_id="wd", level="ERROR")
                log_line(log_path, _safe_json({"ts": now_ms(), "kind": "STALL", "stalled": stalled}))
                try:
                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write("\n=== STALL %s stalled=%s ===\n" % (now_ms(), stalled))
                        faulthandler.dump_traceback(file=f, all_threads=True)
                except Exception:
                    pass
            await asyncio.sleep(max(0.2, wd_int))

    await asyncio.gather(tick_loop(), price_loop(), events_push_loop(), cmd_loop(), watchdog_loop())

async def main():
    print('HUB_MAIN_ENTER')
    _boot_log(f"{datetime.datetime.now().isoformat()} HUB_MAIN_ENTER")

    cfg = Config.load("config.yaml").raw
    rt = cfg["runtime"]
    es = SQLiteEventStore(cfg["storage"]["events_db"])
    trades_store = TradesStore(cfg["storage"].get("trades_db","data/trades.sqlite"))
    ss = FileSnapshotStore(cfg["storage"]["snapshots_dir"])
    symdb = SymbolStore(cfg["storage"]["symbols_db"])

    market = CCXTMarket(rt.get("exchange", "bybit"), market=rt.get("market", "perp")) if rt.get("adapter", "mock") == "ccxt" else MockMarket()
    global tele
    tele = Telemetry(path=str(Path(cfg.get('storage',{}).get('data_dir','data'))/ 'telemetry_hub.jsonl'), service='hub', es=es, base_event={'exchange':rt.get('exchange'),'market':rt.get('market'),'timeframe':rt.get('timeframe')})
    eng = Engine(cfg, market, es, ss, symdb)
    eng.telemetry = tele

    ws_client = BybitPublicWS(channel='linear', testnet=bool(cfg.get('runtime',{}).get('testnet', False)))
    ws_stop = asyncio.Event()
    ws_task = asyncio.create_task(ws_client.run(ws_stop))

    price_worker = PriceWorker(cfg)
    price_worker.start()
    try:
        rdy = price_worker.wait_ready(timeout_sec=3.0)
        append_event(es, rt, "PRICE_WORKER_READY", rdy, run_id="sys", level="INFO")
    except Exception as e:
        append_event(es, rt, "PRICE_WORKER_READY", {"type":"error","err":str(e)}, run_id="sys", level="ERROR")



    ok, rep = preflight(cfg, market, symdb)
    STATE["preflight_ok"] = ok
    STATE["preflight_report"] = rep

    host = rt.get("ws_host", "127.0.0.1")
    port = int(rt.get("ws_port", 8765))
    asyncio.create_task(COMMANDS.pump(eng, cfg, market, symdb))

    async with legacy_serve(handler, host, port, ping_interval=None, ping_timeout=None, logger=None):
        print(f'HUB_LISTENING ws://{host}:{port}')
        _boot_log(f"{datetime.datetime.now().isoformat()} HUB_LISTENING ws://{host}:{port}")
        await broadcast({"type": "status", "ts": now_ms(), "status": "hub_started"})
        await broadcast({"type": "preflight", "ts": now_ms(), "ok": STATE["preflight_ok"], "report": STATE["preflight_report"]})
        try:
            await loops(cfg, eng, es, market, tele, price_worker, ws_client, trades_store)
        except Exception as e:
            import traceback
            print('HUB_LOOPS_FATAL:', e)
            _boot_log('HUB_LOOPS_FATAL: '+repr(e))
            _boot_log(traceback.format_exc())
            raise
        print('HUB_LOOPS_RETURNED_UNEXPECTEDLY')
        _boot_log(f"{datetime.datetime.now().isoformat()} HUB_LOOPS_RETURNED_UNEXPECTEDLY")
        while True:
            await asyncio.sleep(60)



# --- robust boot wrapper (writes to data/hub_run.log) ---
def _boot_log(msg: str) -> None:
    try:
        from pathlib import Path as _P
        fp = _P(__file__).resolve().parents[1] / "data" / "hub_run.log"
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    print("HUB_BOOT: starting...")
    _boot_log(f"{datetime.datetime.now().isoformat()} HUB_BOOT starting")
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print("HUB_FATAL:", e)
        _boot_log("HUB_FATAL: " + repr(e))
        _boot_log(traceback.format_exc())
        time.sleep(3)
        raise
