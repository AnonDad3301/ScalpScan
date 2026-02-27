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


if __name__ == "__main__":
    unittest.main()
