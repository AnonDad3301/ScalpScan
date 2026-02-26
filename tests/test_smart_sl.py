import unittest

from packages.execution.paper import PaperPortfolio


class SmartSLTests(unittest.TestCase):
    def _cfg(self):
        return {
            "execution": {
                "deposit_usd": 5000,
                "risk_usd": 100,
                "per_trade_alloc_pct": 0.5,
                "total_alloc_pct": 0.9,
                "exits": {
                    "sl_atr_mult": 1.0,
                    "tp1_atr_mult": 1.5,
                    "tp2_atr_mult": 2.0,
                    "tp1_fraction": 0.0,
                    "breakeven_r": 0.8,
                    "time_stop_minutes": 0,
                },
                "smart_sl": {
                    "enabled": True,
                    "arm_r": 0.3,
                    "min_step_bps": 1.0,
                },
            }
        }

    def test_smart_sl_moves_and_emits_update(self):
        p = PaperPortfolio(self._cfg())
        res = p.open(0, "BTC/USDT", "LONG", 100.0, atr=1.0)
        self.assertEqual(res["result"], "OK")
        updates, closed = p.manage_positions(1, {"BTC/USDT": 101.5}, market_ctx={"BTC/USDT": {"bid_wall": 101.0}})
        self.assertFalse(closed)
        self.assertTrue(any(u.get("event") == "SL_MOVE" for u in updates))
        self.assertGreaterEqual(p.positions["BTC/USDT"].sl, 100.0)


if __name__ == "__main__":
    unittest.main()
