import unittest

from packages.feature.pipeline import compute


class ModernScalpFeaturesTests(unittest.TestCase):
    def test_pipeline_exposes_epic_features(self):
        ohlcv = []
        p = 100.0
        for i in range(50):
            p += (0.3 if i % 3 else -0.15)
            ohlcv.append((i * 60_000, p - 0.2, p + 0.4, p - 0.5, p, 100 + i))
        ob = {
            "spread": 0.2,
            "imbalance": 0.1,
            "bids": [[p - 0.1, 10], [p - 0.2, 9]],
            "asks": [[p + 0.1, 8], [p + 0.2, 7]],
        }
        feats = compute(ohlcv, ob)
        for k in ("avwap_dist", "squeeze_on", "liquidity_sweep", "cvd_proxy"):
            self.assertIn(k, feats)


if __name__ == "__main__":
    unittest.main()
