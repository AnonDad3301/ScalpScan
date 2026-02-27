import unittest
import numpy as np

from packages.model.ensemble import meta_decide
from packages.model.labeling import triple_barrier_labels


class EnsembleAndLabelingTests(unittest.TestCase):
    def test_triple_barrier_tp_first(self):
        close = np.array([100.0, 100.3, 100.6, 100.9, 101.2, 101.3])
        atr = np.full(len(close), 0.2)
        y, p_tp, p_sl = triple_barrier_labels(close, atr, horizon_bars=3, tp_atr_mult=1.0, sl_atr_mult=1.0)
        self.assertEqual(int(y[0]), 1)
        self.assertEqual(float(p_tp[0]), 1.0)
        self.assertEqual(float(p_sl[0]), 0.0)

    def test_ensemble_gray_zone(self):
        out = meta_decide(
            model_a={"p_up_3m": 0.51, "p_up_5m": 0.52},
            model_b={"p_up_3m": 0.50, "p_up_5m": 0.51},
            regime="trend",
            cfg={"gray_zone": [0.45, 0.55]},
        )
        self.assertEqual(out.direction, "LONG")
        self.assertEqual(out.no_trade_reason, "gray_zone")


if __name__ == "__main__":
    unittest.main()
