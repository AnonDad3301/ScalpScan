from __future__ import annotations
import asyncio, json, time, logging, datetime, faulthandler, uuid, os
import urllib.parse, urllib.request
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
from backend.monitoring_api import start_monitoring_api
from packages.obs.telemetry import Telemetry


logging.getLogger("websockets").disabled = True
logging.getLogger("websockets.server").disabled = True
logging.getLogger("websockets.http11").disabled = True

clients: Set[Any] = set()
STATE: Dict[str, Any] = {"running": False, "preflight_ok": False, "preflight_report": []}
telegram_notifier = None

tele: Telemetry | None = None


def now_ms() -> int:
    return int(time.time() * 1000)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst



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


# Global variable to store trades_store reference
trades_store_global = None

def append_event(es, rt: Dict[str, Any], stage: str, payload: Dict[str, Any], run_id: str="sys", level: str="INFO", symbol: str="*"):
    if stage == "TRADE_CLOSED":
        try:
            if trades_store_global is not None:
                trades_store_global.upsert(payload)
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
    try:
        if telegram_notifier is not None:
            r = telegram_notifier.notify(stage=stage, payload=payload, symbol=symbol, timeframe=str(rt.get("timeframe", "")))
            if isinstance(r, dict) and (not r.get("ok", False)) and r.get("reason") not in ("filtered_or_disabled", "empty_message"):
                es.append({
                    "event_id": str(uuid.uuid4()), "ts": now_ms(), "run_id": run_id, "service": "hub",
                    "exchange": rt.get("exchange"), "market": rt.get("market"), "symbol": symbol,
                    "timeframe": rt.get("timeframe"), "stage": "TELEGRAM_ERROR", "level": "ERROR", "payload": {"stage": stage, **r}
                })
    except Exception:
        pass


class TelegramNotifier:
    def __init__(self, cfg: Dict[str, Any]):
        tg = (cfg.get("notifications", {}) or {}).get("telegram", {}) or {}
        self.enabled = bool(tg.get("enabled", False))
        self.bot_token = str(tg.get("bot_token", "") or "")
        self.chat_id = str(tg.get("chat_id", "") or "")
        self.send_signal = bool(tg.get("send_signal", True))
        self.send_trade_open = bool(tg.get("send_trade_open", True))
        self.send_trade_closed = bool(tg.get("send_trade_closed", True))
        self.send_on_gate_fail = bool(tg.get("send_on_gate_fail", False))
        self.min_interval_sec = float(tg.get("min_interval_sec", 2.0) or 0.0)
        self.include_volatility = bool(tg.get("include_volatility", True))
        self.include_positions = bool(tg.get("include_positions", True))
        self.include_levels = bool(tg.get("include_levels", True))
        self.include_probabilities = bool(tg.get("include_probabilities", True))
        self.include_timing = bool(tg.get("include_timing", True))
        self._last_sent_ts: Dict[str, int] = {}
        self._sent_event_ids: set[str] = set()
        self._sent_event_order: list[str] = []
        self._sent_event_limit = int(tg.get("dedupe_cache_size", 4000) or 4000)

    def reconfigure(self, cfg: Dict[str, Any]) -> None:
        self.__init__(cfg)

    def _should_send(self, stage: str, payload: Dict[str, Any]) -> bool:
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False
        if stage == "SIGNAL":
            if not self.send_signal:
                return False
            g = payload.get("gate", {}) if isinstance(payload.get("gate"), dict) else {}
            if (g.get("decision") == "FAIL") and (not self.send_on_gate_fail):
                return False
        elif stage == "TRADE_OPEN":
            if not self.send_trade_open:
                return False
            if str(payload.get("result", "OK")) != "OK":
                return False
        elif stage == "TRADE_CLOSED":
            if not self.send_trade_closed:
                return False
        else:
            return False
        now = now_ms()
        prev = int(self._last_sent_ts.get(stage, 0))
        if self.min_interval_sec > 0 and (now - prev) < int(self.min_interval_sec * 1000):
            return False
        self._last_sent_ts[stage] = now
        return True

    def notify(self, stage: str, payload: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        if not self._should_send(stage, payload):
            return {"ok": False, "reason": "filtered_or_disabled"}
        text = self._format_message(stage, payload, symbol, timeframe)
        if not text:
            return {"ok": False, "reason": "empty_message"}
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                body = resp.read().decode("utf-8", errors="ignore")[:800]
            return {"ok": True, "status": "sent", "response": body}
        except Exception as e:
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def notify_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event_id = str(event.get("event_id") or "")
        if event_id:
            if event_id in self._sent_event_ids:
                return {"ok": False, "reason": "already_sent"}
            self._sent_event_ids.add(event_id)
            self._sent_event_order.append(event_id)
            if len(self._sent_event_order) > self._sent_event_limit:
                old = self._sent_event_order.pop(0)
                self._sent_event_ids.discard(old)
        return self.notify(
            stage=str(event.get("stage", "")),
            payload=(event.get("payload") if isinstance(event.get("payload"), dict) else {}),
            symbol=str(event.get("symbol", "*") or "*"),
            timeframe=str(event.get("timeframe", "") or ""),
        )

    def _fmt_num(self, v: Any, digits: int = 6) -> str:
        try:
            fv = float(v)
            return f"{fv:.{digits}f}".rstrip("0").rstrip(".")
        except Exception:
            return "н/д"

    def _fmt_pct(self, v: Any, digits: int = 1) -> str:
        try:
            fv = float(v)
            if abs(fv) <= 1.0:
                fv *= 100.0
            return f"{fv:.{digits}f}%"
        except Exception:
            return "н/д"

    def _gate_reason_ru(self, code: str) -> str:
        mp = {
            "DIRECTIONAL_AGREEMENT": "нет согласия моделей",
            "UNCERTAINTY_MAX": "слишком высокая неопределенность",
            "EV_POSITIVE": "ожидаемая доходность ниже порога",
            "SPREAD_MAX": "слишком большой спред",
            "RR_MIN": "низкое риск/прибыль",
            "AUC_MIN": "низкое качество модели",
        }
        c = str(code or "").strip().upper()
        return mp.get(c, c or "-")

    def _format_message(self, stage: str, p: Dict[str, Any], symbol: str, timeframe: str) -> str:
        if stage == "SIGNAL":
            g = p.get("gate", {}) if isinstance(p.get("gate"), dict) else {}
            reasons = [self._gate_reason_ru(x) for x in (g.get("reasons", []) or [])]
            lines = [
                f"📡 <b>Сигнал</b> {symbol} ({timeframe})",
                f"Направление: <b>{p.get('direction')}</b>",
            ]
            if self.include_probabilities:
                lines.append(f"Вероятность LONG (3m/5m): <b>{self._fmt_pct(p.get('p_up_3m'))}</b> / <b>{self._fmt_pct(p.get('p_up_5m'))}</b>")
                lines.append(f"Вероятность SHORT (3m/5m): <b>{self._fmt_pct(p.get('p_down_3m'))}</b> / <b>{self._fmt_pct(p.get('p_down_5m'))}</b>")
                lines.append(f"TP раньше SL: {self._fmt_pct(p.get('p_tp_first'))} | SL раньше TP: {self._fmt_pct(p.get('p_sl_first'))}")
                lines.append(f"Пробой вверх/вниз: {self._fmt_pct(p.get('p_breakout_up'))} / {self._fmt_pct(p.get('p_breakout_down'))}")
            lines.append(f"Сила сигнала: {self._fmt_pct(p.get('signal_strength'))} | Уверенность: {self._fmt_pct(p.get('confidence'))}")
            if self.include_volatility:
                lines.append(f"Режим: {p.get('market_regime')} | Волатильность: {self._fmt_pct(p.get('volatility_pct'))}")
                lines.append(f"ATR percentile: {self._fmt_pct(p.get('atr_percentile'))}")
            lines.append(f"Gate: <b>{g.get('decision')}</b> | Причины: {', '.join(reasons) if reasons else '-'}")
            if self.include_levels:
                lines.append(f"Вход: {self._fmt_num(p.get('entry'))} | SL: {self._fmt_num(p.get('sl'))} | TP1: {self._fmt_num(p.get('tp1'))} | TP2: {self._fmt_num(p.get('tp2'))}")
                lines.append(f"Поддержка/сопротивление: {self._fmt_num(p.get('support_level'))} / {self._fmt_num(p.get('resistance_level'))}")
            if self.include_timing:
                lines.append(f"ts={now_ms()}")
            return "\n".join(lines)
        if stage == "TRADE_OPEN":
            lines = [f"🟢 <b>Позиция открыта</b> {symbol}"]
            lines.append(f"Сторона: {p.get('side')} | Entry: {self._fmt_num(p.get('entry'))}")
            if self.include_positions:
                lines.append(f"Объем: {self._fmt_num(p.get('qty'))} | Номинал: {self._fmt_num(p.get('notional'), 2)}")
                lines.append(f"SL: {self._fmt_num(p.get('sl'))} | TP1: {self._fmt_num(p.get('tp1'))} | TP2: {self._fmt_num(p.get('tp2'))}")
            if self.include_timing:
                lines.append(f"ts={now_ms()}")
            return "\n".join(lines)
        if stage == "TRADE_CLOSED":
            lines = [f"🔴 <b>Позиция закрыта</b> {symbol}"]
            if self.include_positions:
                lines.append(f"Причина: {p.get('reason')}")
                lines.append(f"Entry: {p.get('entry')} | Exit: {p.get('exit')} | PnL: <b>{p.get('pnl')}</b>")
            if self.include_timing:
                lines.append(f"ts={now_ms()}")
            return "\n".join(lines)
        return ""


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
            open_syms = list(getattr(eng.portfolio, "positions", {}).keys())
            if STATE.get("running", False):
                try:
                    uni_syms = list(eng.universe())
                except Exception:
                    uni_syms = []
            else:
                uni_syms = []
            syms = list(dict.fromkeys(open_syms + uni_syms))[: int(rt.get("max_pairs", 50) or 50)]
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
            pm: Dict[str, Any] = {"status": "SKIP", "reason": "no_symbols"}
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
                if not isinstance(prices, dict):
                    prices = {}
                if syms and pm.get("status") != "OK":
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
                append_event(es, rt, "PRICE_TIMEOUT", {"timeout_sec": price_timeout, "open_positions": len(open_syms), "tracked_symbols": len(syms)}, run_id="price", level="ERROR")
            except Exception as e:
                status = "ERROR"
                append_event(es, rt, "PRICE_ERROR", {"err": str(e)}, run_id="price", level="ERROR")

            dt_ms = int((time.time()-t0)*1000)
            liveness["price"] = time.monotonic()
            append_event(es, rt, "PRICE_TICK", {"open_positions": len(open_syms), "tracked_symbols": len(syms), "prices_n": len(prices), "dt_ms": dt_ms, "status": status, "symbols": syms[:50], "ws_connected": ws_connected, "ws_stale_ms": ws_stale_ms, "ws_ticker_1s": ws_ticker_1s, "ws_sub_ok": ws_sub_ok, "ws_sub_err": ws_sub_err}, run_id="price")

            try:
                sp = tele.span_start("PRICE_TICK", trace_id="price_tick", symbol="*", open_positions=len(open_syms), tracked_symbols=len(syms))
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
                "MODEL_INFERRED","MODEL_TRADE_SAMPLE","MODEL_RETRAIN_SKIPPED","GATE_DECISION","STALL_DETECTED",
                "SHORT_TERM_LEVEL_PROB","TRADE_OUTCOME_STATS",
                "MODEL_B_LOAD_START","MODEL_B_LOAD_END","MODEL_B_INFERRED",
                "MODEL_B_DATASET_APPEND","MODEL_B_STATUS","MODEL_B_RETRAIN","MODEL_B_RETRAIN_SKIPPED","MODEL_B_TRAINING_STATUS","MODEL_B_PROMOTED"
            )]
            try:
                if telegram_notifier is not None:
                    for e in reversed(interesting):
                        if e.get("stage") not in ("SIGNAL", "TRADE_OPEN", "TRADE_CLOSED"):
                            continue
                        r = telegram_notifier.notify_event(e)
                        if isinstance(r, dict) and (not r.get("ok", False)) and r.get("reason") not in ("filtered_or_disabled", "empty_message", "already_sent"):
                            append_event(es, rt, "TELEGRAM_ERROR", {"stage": e.get("stage"), "event_id": e.get("event_id"), **r}, run_id="events", level="ERROR", symbol=str(e.get("symbol", "*")))
            except Exception:
                pass
            liveness["events"] = time.monotonic()
            await broadcast({"type": "events", "ts": now_ms(), "events": list(reversed(interesting))[-350:]})
            await asyncio.sleep(1.0)


    async def cmd_loop():
        """Poll data/ui_commands.jsonl and set one-shot or sticky flags."""
        cmd_path = str(Path(__file__).resolve().parents[1] / "data" / "ui_commands.jsonl")
        last_pos = 0
        if os.path.exists(cmd_path):
            try:
                last_pos = os.path.getsize(cmd_path)
            except Exception:
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
                            elif c=="reload_config":
                                try:
                                    new_cfg = Config.load("config.yaml").raw
                                    _deep_update(cfg, new_cfg)
                                    eng.cfg = cfg
                                    global telegram_notifier
                                    if telegram_notifier is None:
                                        telegram_notifier = TelegramNotifier(cfg)
                                    else:
                                        telegram_notifier.reconfigure(cfg)
                                    append_event(es, rt, "CONFIG_RELOADED", {"status": "OK"}, run_id="cmd", level="INFO")
                                    append_event(es, rt, "CMD", {"cmd": c, "status": "applied"}, run_id="cmd", level="INFO")
                                except Exception as e:
                                    append_event(es, rt, "CMD", {"cmd": c, "status": "error", "err": str(e)}, run_id="cmd", level="ERROR")
                            elif c=="send_test_telegram":
                                test_payload = {
                                    "direction": "LONG",
                                    "pred": 0.77,
                                    "confidence": 0.81,
                                    "p_up_3m": 0.72,
                                    "p_up_5m": 0.69,
                                    "p_tp_first": 0.66,
                                    "p_sl_first": 0.34,
                                    "signal_strength": 0.44,
                                    "market_regime": "trend",
                                    "gate": {"decision": "PASS", "reasons": []},
                                    "entry": 100.0,
                                    "sl": 99.2,
                                    "tp1": 100.8,
                                    "tp2": 101.3,
                                }
                                r = {"ok": False, "reason": "not_initialized"}
                                try:
                                    if telegram_notifier is not None:
                                        r = telegram_notifier.notify(stage="SIGNAL", payload=test_payload, symbol="TEST/USDT", timeframe=str(rt.get("timeframe", "1m")))
                                except Exception as e:
                                    r = {"ok": False, "reason": str(e)}
                                append_event(es, rt, "CMD", {"cmd": c, **r}, run_id="cmd", level="INFO" if r.get("ok") else "ERROR")
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
    global telegram_notifier
    telegram_notifier = TelegramNotifier(cfg)
    rt = cfg["runtime"]
    es = SQLiteEventStore(cfg["storage"]["events_db"])
    trades_store = TradesStore(cfg["storage"].get("trades_db","data/trades.sqlite"))
    global trades_store_global
    trades_store_global = trades_store
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
    mon_host = rt.get("monitoring_host", "127.0.0.1")
    mon_port = int(rt.get("monitoring_port", 8081))
    mon_server = await start_monitoring_api(mon_host, mon_port, STATE)
    if mon_server is not None:
        append_event(es, rt, "MONITORING_API_READY", {"host": mon_host, "port": mon_port}, run_id="sys", level="INFO")
    else:
        append_event(es, rt, "MONITORING_API_DISABLED", {"reason": "aiohttp_missing"}, run_id="sys", level="INFO")

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
