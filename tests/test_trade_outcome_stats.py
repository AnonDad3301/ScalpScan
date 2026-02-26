import unittest

from packages.execution.paper import PaperPortfolio


class TradeOutcomeStatsTests(unittest.TestCase):
    def test_stats_count_closed_outcomes(self):
        cfg = {
            "execution": {
                "deposit_usd": 5000,
                "risk_usd": 100,
                "per_trade_alloc_pct": 0.5,
                "total_alloc_pct": 0.9,
                "exits": {"sl_atr_mult": 1.0, "tp1_atr_mult": 1.2, "tp2_atr_mult": 1.5, "tp1_fraction": 0.0, "breakeven_r": 0.0, "time_stop_minutes": 0},
            }
        }
        p = PaperPortfolio(cfg)
        p.open(0, "A", "LONG", 100.0, atr=1.0)
        p.manage_positions(1, {"A": 98.0})  # likely SL
        st = p.stats()
        self.assertEqual(st["total_closed"], 1)
        self.assertEqual(st["sl"], 1)


if __name__ == "__main__":
    unittest.main()
