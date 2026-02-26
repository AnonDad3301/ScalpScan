import unittest
import numpy as np

from packages.model.model_b_head import ModelBHead


class ModelBHeadStabilityTests(unittest.TestCase):
    def test_proba_is_finite_on_large_logits(self):
        h = ModelBHead()
        h.fitted = True

        class Dummy:
            def decision_function(self, X):
                return np.array([1e6, -1e6], dtype=float)

        h.clf = Dummy()
        p = h.proba(np.zeros((2, 8), dtype=float))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(float(p.min()), 0.0)
        self.assertLessEqual(float(p.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
