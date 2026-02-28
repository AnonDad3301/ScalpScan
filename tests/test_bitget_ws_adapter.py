import asyncio
import json

from backend.bitget_ws_prices import BitgetPublicWS


class _DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))


def test_bitget_subscribe_and_ticker_parsing():
    async def _run():
        ws = BitgetPublicWS()
        await ws.set_desired(["BTC/USDT:USDT"])
        dummy = _DummyWS()
        await ws._sync_subs(dummy)

        assert dummy.sent
        req = dummy.sent[0]
        assert req["op"] == "subscribe"
        req_id = req["id"]

        handled = await ws._handle_message({"event": "subscribe", "id": req_id})
        assert handled is True
        assert ws.state.sub_ok == 1

        handled = await ws._handle_message(
            {
                "arg": {"channel": "ticker", "instId": "BTCUSDT"},
                "data": [{"lastPr": "43123.5"}],
            }
        )
        assert handled is True
        prices, _ = await ws.get_prices()
        assert prices["BTC/USDT:USDT"] == 43123.5

    asyncio.run(_run())
