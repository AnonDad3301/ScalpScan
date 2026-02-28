import unittest

from apps.runtime.engine import Engine


class _SymDbStub:
    def __init__(self, strict, relaxed):
        self.strict = strict
        self.relaxed = relaxed

    def top_symbols(self, exchange, market, limit, quote=None, min_volume=0.0):
        if min_volume > 0:
            return self.strict[:limit]
        return self.relaxed[:limit]


class EngineUniverseTests(unittest.TestCase):
    def _mk_engine(self, strict, relaxed, custom, max_pairs=5):
        eng = Engine.__new__(Engine)
        eng.cfg = {
            "runtime": {"exchange": "bybit", "market": "linear", "max_pairs": max_pairs},
            "universe": {
                "mode": "db",
                "quote": "USDT",
                "min_volume": 1_000_000,
                "custom_symbols": custom,
            },
        }
        eng.symdb = _SymDbStub(strict=strict, relaxed=relaxed)
        return eng

    def test_universe_fills_when_strict_db_returns_too_few(self):
        eng = self._mk_engine(
            strict=["BTC/USDT:USDT"],
            relaxed=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
            custom=["XRP/USDT:USDT", "DOGE/USDT:USDT"],
            max_pairs=5,
        )
        out = eng.universe()
        self.assertEqual(
            out,
            [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "SOL/USDT:USDT",
                "XRP/USDT:USDT",
                "DOGE/USDT:USDT",
            ],
        )

    def test_universe_deduplicates_db_and_custom(self):
        eng = self._mk_engine(
            strict=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            relaxed=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
            custom=["ETH/USDT:USDT", "XRP/USDT:USDT"],
            max_pairs=4,
        )
        out = eng.universe()
        self.assertEqual(out, ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"])

    def test_universe_handles_invalid_custom_symbols_type(self):
        eng = self._mk_engine(
            strict=["BTC/USDT:USDT"],
            relaxed=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            custom=0.5,
            max_pairs=3,
        )
        out = eng.universe()
        self.assertEqual(out, ["BTC/USDT:USDT", "ETH/USDT:USDT"])


if __name__ == "__main__":
    unittest.main()
