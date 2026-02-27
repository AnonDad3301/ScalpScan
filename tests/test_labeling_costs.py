import unittest
import numpy as np

from packages.model.labeling import triple_barrier_labels


class LabelingCostsTests(unittest.TestCase):
    def test_costs_make_tp_harder(self):
        close = np.array([100.0, 100.05, 100.1, 100.12, 100.15, 100.2])
        atr = np.full_like(close, 0.05)
        y0, _, _ = triple_barrier_labels(close, atr, horizon_bars=3, tp_atr_mult=1.0, sl_atr_mult=1.0, costs_bps=0.0)
        y1, _, _ = triple_barrier_labels(close, atr, horizon_bars=3, tp_atr_mult=1.0, sl_atr_mult=1.0, costs_bps=20.0)
        self.assertGreaterEqual(int((y0 == 1).sum()), int((y1 == 1).sum()))


if __name__ == "__main__":
    unittest.main()
