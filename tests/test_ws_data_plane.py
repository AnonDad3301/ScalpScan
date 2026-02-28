import asyncio
import json

from backend.bybit_ws_prices import BybitPublicWS


class _DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))


def test_ws_subscribe_ack_tracks_subscribed_count():
    async def _run():
        ws = BybitPublicWS()
        await ws.set_desired(["BTC/USDT:USDT", "ETH/USDT:USDT"])
        dummy = _DummyWS()
        await ws._sync_subs(dummy)

        assert dummy.sent
        req = dummy.sent[0]
        assert req["op"] == "subscribe"
        assert req.get("req_id")

        ok = await ws._handle_message({"op": "subscribe", "req_id": req["req_id"], "success": True})
        assert ok is True
        assert ws.state.sub_ok == 1
        assert ws.state.subscribed_n == len(req["args"])

    asyncio.run(_run())


def test_ws_ticker_parses_dict_and_list_payloads():
    async def _run():
        ws = BybitPublicWS()
        await ws.set_desired(["BTC/USDT:USDT"])
        ws._bybit_to_orig = {"BTCUSDT": "BTC/USDT:USDT"}

        handled = await ws._handle_message(
            {"topic": "tickers.BTCUSDT", "data": {"symbol": "BTCUSDT", "lastPrice": "101.5"}}
        )
        assert handled is True
        prices, _ = await ws.get_prices()
        assert prices["BTC/USDT:USDT"] == 101.5

        handled = await ws._handle_message(
            {"topic": "tickers.BTCUSDT", "data": [{"symbol": "BTCUSDT", "lastPrice": "102.0"}]}
        )
        assert handled is True
        prices, _ = await ws.get_prices()
        assert prices["BTC/USDT:USDT"] == 102.0

    asyncio.run(_run())
