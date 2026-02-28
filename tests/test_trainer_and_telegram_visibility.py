import unittest
import numpy as np

from packages.model.trainer import OnlineTrainer
from backend.hub import TelegramNotifier


class TrainerVisibilityTests(unittest.TestCase):
    def test_health_samples_updates_without_retrain(self):
        tr = OnlineTrainer(min_samples=1000, retrain_interval_sec=3600)
        X = np.asarray([[1.0, 2.0, 3.0, 0.1, 5.0], [1.1, 2.1, 3.1, 0.2, 5.2]], dtype=float)
        y = np.asarray([1, -1], dtype=int)
        added = tr.add_samples(X, y)
        self.assertEqual(added, 2)
        h = tr.health()
        self.assertEqual(h.get("samples"), 2)


class TelegramFormattingTests(unittest.TestCase):
    def test_trade_closed_contains_trade_and_signal_refs(self):
        n = TelegramNotifier({
            "notifications": {
                "telegram": {
                    "enabled": True,
                    "bot_token": "x",
                    "chat_id": "y",
                    "include_positions": True,
                    "include_timing": False,
                }
            }
        })
        msg = n._format_message(
            "TRADE_CLOSED",
            {
                "side": "LONG",
                "reason": "TP2",
                "entry": 100.0,
                "exit": 101.5,
                "pnl": 12.34,
                "open_trade_event_id": "trade-evt-1",
                "signal_event_id": "sig-evt-2",
                "open_ts": 10,
                "close_ts": 20,
            },
            symbol="BTC/USDT",
            timeframe="1m",
        )
        self.assertIn("TradeRef", msg)
        self.assertIn("SignalRef", msg)
        self.assertIn("trade-evt-1", msg)
        self.assertIn("sig-evt-2", msg)


if __name__ == "__main__":
    unittest.main()
