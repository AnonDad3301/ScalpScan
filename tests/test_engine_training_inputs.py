import unittest

from apps.runtime.engine import Engine


class EngineTrainingInputsTests(unittest.TestCase):
    def test_model_a_matrix_is_not_constant(self):
        eng = Engine.__new__(Engine)
        ohlcv = []
        p = 100.0
        for i in range(40):
            p += (0.3 if i % 2 == 0 else -0.1)
            ohlcv.append((i * 60_000, p - 0.2, p + 0.4, p - 0.5, p, 100 + i))
        ob = {"spread": 0.2, "imbalance": 0.12}

        X = eng._build_model_a_matrix(ohlcv, ob)
        self.assertEqual(X.shape[0], len(ohlcv))
        self.assertEqual(X.shape[1], 5)
        # RSI/ATR/ADX evolve through time (not cloned row anymore)
        self.assertGreater(float((X[:, 0].max() - X[:, 0].min())), 1e-9)


    def test_mtf_trend_context_detects_short_bias(self):
        eng = Engine.__new__(Engine)
        import numpy as np
        close = np.linspace(120.0, 100.0, 80)
        ctx = eng._mtf_trend_context(close, "3m")
        self.assertEqual(ctx["dir_60m"], "SHORT")
        self.assertLess(ctx["bias"], 0.0)

    def test_mtf_trend_context_has_confidence_fields(self):
        eng = Engine.__new__(Engine)
        import numpy as np
        close = np.linspace(100.0, 112.0, 120)
        ctx = eng._mtf_trend_context(close, "1m")
        self.assertIn("conf_15m", ctx)
        self.assertIn("conf_60m", ctx)
        self.assertGreaterEqual(float(ctx["conf_60m"]), 0.0)
        self.assertEqual(ctx["dir_60m"], "LONG")

    def test_debias_prob_pulls_long_drift_toward_center(self):
        eng = Engine.__new__(Engine)
        eng._p_center_ema = 0.70
        out = eng._debias_prob(0.80)
        self.assertLess(out, 0.80)
        self.assertGreaterEqual(out, 0.0)


if __name__ == "__main__":
    unittest.main()
