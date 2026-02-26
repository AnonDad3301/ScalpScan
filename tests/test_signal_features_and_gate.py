import unittest

from packages.feature.pipeline import compute
from packages.decision.gate import decide


class SignalFeatureGateTests(unittest.TestCase):
    def test_pipeline_computes_nonzero_micro_features(self):
        ohlcv = []
        p = 100.0
        for i in range(40):
            p += 0.2
            ohlcv.append((i * 60000, p - 0.1, p + 0.3, p - 0.3, p, 100 + i))
        ob = {
            "spread": 0.2,
            "spread_bps": 20.0,
            "imbalance": 0.15,
        }
        feats = compute(ohlcv, ob)
        self.assertGreater(feats["spread_bps"], 0)
        self.assertNotEqual(feats["ret_last"], 0.0)
        self.assertNotEqual(feats["range_last"], 0.0)
        self.assertIn("vol_z_last", feats)

    def test_gate_exposes_quality_filters(self):
        features = {
            "adx": 30,
            "spread_bps": 5,
            "volz": 0.8,
            "ret_last": 0.001,
            "range_last": 0.002,
            "vol_z_last": 0.3,
            "ob_imb": 0.1,
            "forecast_uncertainty": 0.005,
            "rr_ratio": 1.8,
            "trend_strength": 0.4,
        }
        out = decide(
            features,
            inv_results=[{"pass": True}],
            model_out={"confidence": 0.8, "pred": 0.2},
            model_health={"samples": 700, "auc": 0.65},
            cfg={
                "profile": "scalp",
                "profiles": {
                    "scalp": {
                        "entry_threshold": 0.06,
                        "confidence_min": 0.52,
                        "adx_min": 8,
                        "spread_max": 20,
                        "volz_max": 3.5,
                        "uncertainty_max": 0.02,
                        "rr_min": 1.2,
                        "trend_strength_min": 0.05,
                    }
                },
                "min_samples": 50,
                "auc_warmup_samples": 600,
                "min_auc": 0.52,
            },
        )
        self.assertEqual(out["decision"], "PASS")
        self.assertIn("forecast_uncertainty", out)
        self.assertIn("rr_ratio", out)


if __name__ == "__main__":
    unittest.main()
