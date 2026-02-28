from __future__ import annotations

from backend.price_worker import PriceWorker
from packages.infra.market.ccxt_market import CCXTMarket
from packages.infra.trades_store.sqlite_trades import TradesStore


def test_trades_store_upsert_matches_schema(tmp_path):
    db = tmp_path / "trades.sqlite"
    store = TradesStore(str(db))
    trade = {
        "trade_id": "t-1",
        "symbol": "BTC/USDT:USDT",
        "side": "LONG",
        "entry_ts": 1000,
        "exit_ts": 2000,
        "entry": 100.0,
        "exit": 101.0,
        "qty": 1.0,
        "fees": 0.1,
        "pnl": 0.9,
        "r_mult": 1.2,
        "reason_exit": "TP",
    }
    store.upsert(trade)

    rows = store.tail(limit=1)
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t-1"


def test_ccxt_market_has_mark_price_method_and_reset_fields(monkeypatch):
    assert hasattr(CCXTMarket, "fetch_mark_price")

    class FakeExchange:
        def __init__(self, params):
            self.params = params
            self.options = {}

    class FakeCCXT:
        fake = FakeExchange

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ccxt":
            return FakeCCXT
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    m = CCXTMarket("fake")

    assert m._ccxt is FakeCCXT
    assert isinstance(m._params, dict)
    assert m._err_count == 0


def test_price_worker_has_fetch_mark_prices_method():
    assert hasattr(PriceWorker, "fetch_mark_prices")
