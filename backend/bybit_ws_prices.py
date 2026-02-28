from __future__ import annotations
import asyncio, json, time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from websockets.legacy.client import connect as legacy_connect

@dataclass
class WSState:
    connected: bool = False
    last_msg_ms: int = 0
    last_pong_ms: int = 0
    last_sub_ms: int = 0
    conn_url: str = ""
    sub_ok: int = 0
    sub_err: int = 0
    last_sub_ack: Dict[str, object] = field(default_factory=dict)
    ticker_updates: int = 0
    ticker_updates_1s: int = 0
    last_tick_1s_ms: int = 0
    desired_n: int = 0
    subscribed_n: int = 0
    last_sub_req: Dict[str, object] = field(default_factory=dict)
    last_error: str = ""

def _to_bybit_symbol(sym: str) -> str:
    s = sym.split(":")[0]
    return s.replace("/", "")

class BybitPublicWS:
    def __init__(self, channel: str = "linear", testnet: bool = False):
        host = "wss://stream-testnet.bybit.com" if testnet else "wss://stream.bybit.com"
        self.url = f"{host}/v5/public/{channel}"
        self.state = WSState(conn_url=self.url)
        self._desired: Set[str] = set()
        self._prices: Dict[str, float] = {}
        self._bybit_to_orig: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._subscribed: Set[str] = set()
        self._pending_sub_reqs: Dict[str, List[str]] = {}
        self._force_resub = False
        self._force_reconnect = False

    async def set_desired(self, symbols: Iterable[str]) -> None:
        async with self._lock:
            self._desired = set(symbols)

    async def get_prices(self) -> Tuple[Dict[str, float], WSState]:
        async with self._lock:
            return dict(self._prices), self.state

    async def run(self, stop_evt: asyncio.Event) -> None:
        ping_every = 20
        while not stop_evt.is_set():
            try:
                async with legacy_connect(self.url, ping_interval=None, ping_timeout=None) as ws:
                    now = int(time.time()*1000)
                    self.state.connected = True
                    self.state.last_msg_ms = now
                    self.state.last_pong_ms = now
                    self.state.ticker_updates = 0
                    self.state.ticker_updates_1s = 0
                    self.state.last_tick_1s_ms = now
                    self._subscribed = set()
                    self._pending_sub_reqs = {}

                    async def pinger():
                        while not stop_evt.is_set():
                            try:
                                await ws.send(json.dumps({"op":"ping","req_id":""}))
                            except Exception:
                                return
                            await asyncio.sleep(ping_every)

                    ping_task = asyncio.create_task(pinger())
                    await self._sync_subs(ws)

                    async for raw in ws:
                        if stop_evt.is_set():
                            break
                        now = int(time.time()*1000)
                        self.state.last_msg_ms = now
                        # forced actions
                        do_resub = False
                        do_reconnect = False
                        async with self._lock:
                            if self._force_resub:
                                self._force_resub = False
                                do_resub = True
                            if self._force_reconnect:
                                self._force_reconnect = False
                                do_reconnect = True
                        if do_reconnect:
                            raise RuntimeError('FORCE_RECONNECT')
                        if do_resub:
                            self._subscribed = set()
                            await self._sync_subs(ws)
                        if now - self.state.last_tick_1s_ms >= 1000:
                            self.state.ticker_updates_1s = 0
                            self.state.last_tick_1s_ms = now
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        if msg.get("op") in ("pong","ping") or msg.get("ret_msg") == "pong":
                            self.state.last_pong_ms = int(time.time()*1000)
                            continue

                        if await self._handle_message(msg):
                            continue

                        if (int(time.time()*1000) - self.state.last_sub_ms) > 5000:
                            await self._sync_subs(ws)

                    ping_task.cancel()
            except Exception as e:
                self.state.connected = False
                self.state.last_error = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.0)

    async def _handle_message(self, msg: Dict[str, object]) -> bool:
        # subscribe ack
        if msg.get("op") == "subscribe" or ("success" in msg and "ret_msg" in msg):
            self.state.last_sub_ack = msg
            req_id = str(msg.get("req_id") or "")
            chunk = self._pending_sub_reqs.pop(req_id, []) if req_id else []
            if msg.get("success") is True or msg.get("retCode") in (0, "0"):
                self.state.sub_ok += 1
                for a in chunk:
                    self._subscribed.add(a)
            else:
                self.state.sub_err += 1
            self.state.subscribed_n = len(self._subscribed)
            return True

        topic = msg.get("topic", "")
        if isinstance(topic, str) and topic.startswith("tickers."):
            data = msg.get("data", {})
            rows = data if isinstance(data, list) else [data]
            if not isinstance(rows, list):
                return True
            for row in rows:
                if not isinstance(row, dict):
                    continue
                b_sym = row.get("symbol") or topic.split(".", 1)[1]
                last = row.get("lastPrice")
                if b_sym and last is not None:
                    try:
                        px = float(last)
                    except Exception:
                        continue
                    async with self._lock:
                        orig = self._bybit_to_orig.get(str(b_sym), str(b_sym))
                        self._prices[orig] = px
                    self.state.ticker_updates += 1
                    self.state.ticker_updates_1s += 1
            return True
        return False

    async def _sync_subs(self, ws) -> None:
        async with self._lock:
            desired = set(self._desired)
        bybit_syms: List[str] = []
        bybit_to_orig: Dict[str, str] = {}
        for s in desired:
            b = _to_bybit_symbol(s)
            if b:
                bybit_syms.append(b)
                bybit_to_orig[b] = s
        async with self._lock:
            self._bybit_to_orig = bybit_to_orig
            self.state.desired_n = len(bybit_syms)

        args = [f"tickers.{b}" for b in bybit_syms]
        if not args:
            return
        pending_args = {a for chunk in self._pending_sub_reqs.values() for a in chunk}
        new_args = [a for a in args if a not in self._subscribed and a not in pending_args]
        if not new_args:
            return
        for i in range(0, len(new_args), 50):
            chunk = new_args[i:i+50]
            req_id = f"sub-{int(time.time()*1000)}-{i}"
            try:
                await ws.send(json.dumps({"op":"subscribe","args":chunk, "req_id": req_id}))
                self.state.last_sub_ms = int(time.time()*1000)
                self.state.last_sub_req = {"req_id": req_id, "args": chunk}
                self._pending_sub_reqs[req_id] = chunk
            except Exception:
                return


    async def request_resubscribe(self) -> None:
        async with self._lock:
            self._force_resub = True

    async def request_reconnect(self) -> None:
        async with self._lock:
            self._force_reconnect = True
